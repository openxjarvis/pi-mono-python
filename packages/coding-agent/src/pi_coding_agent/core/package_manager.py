"""
Package manager for installing, removing, and resolving agent extension packages.

Supports local paths, npm packages (via npm CLI), and git repositories.
Mirrors core/package-manager.ts
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pi_coding_agent.utils.git import GitSource, parse_git_url


SourceScope = Literal["user", "project", "temporary"]
ResourceType = Literal["extensions", "skills", "prompts", "themes"]
MissingSourceAction = Literal["install", "skip", "error"]

RESOURCE_TYPES: list[ResourceType] = ["extensions", "skills", "prompts", "themes"]

_FILE_PATTERNS: dict[ResourceType, str] = {
    "extensions": r"\.(ts|js)$",
    "skills": r"\.md$",
    "prompts": r"\.md$",
    "themes": r"\.json$",
}

_IGNORE_FILE_NAMES = [".gitignore", ".ignore", ".fdignore"]


@dataclass
class PathMetadata:
    source: str
    scope: SourceScope
    origin: Literal["package", "top-level"]
    base_dir: str | None = None


@dataclass
class ResolvedResource:
    path: str
    enabled: bool
    metadata: PathMetadata


@dataclass
class ResolvedPaths:
    extensions: list[ResolvedResource] = field(default_factory=list)
    skills: list[ResolvedResource] = field(default_factory=list)
    prompts: list[ResolvedResource] = field(default_factory=list)
    themes: list[ResolvedResource] = field(default_factory=list)


@dataclass
class ProgressEvent:
    type: Literal["start", "progress", "complete", "error"]
    action: Literal["install", "remove", "update", "clone", "pull"]
    source: str
    message: str | None = None


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class _NpmSource:
    type: Literal["npm"] = "npm"
    spec: str = ""
    name: str = ""
    pinned: bool = False


@dataclass
class _LocalSource:
    type: Literal["local"] = "local"
    path: str = ""


_ParsedSource = _NpmSource | GitSource | _LocalSource


@dataclass
class _PiManifest:
    extensions: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)


def _is_pattern(s: str) -> bool:
    return s.startswith("!") or s.startswith("+") or s.startswith("-") or "*" in s or "?" in s


def _collect_files(dir_path: str, pattern: str) -> list[str]:
    """Recursively collect files matching regex pattern in dir_path."""
    result: list[str] = []
    if not os.path.isdir(dir_path):
        return result
    rx = re.compile(pattern)
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__")]
        for fname in files:
            if rx.search(fname):
                result.append(os.path.join(root, fname))
    return result


def _collect_skill_entries(dir_path: str, include_root_files: bool = True) -> list[str]:
    entries: list[str] = []
    if not os.path.isdir(dir_path):
        return entries
    try:
        for item in sorted(os.scandir(dir_path), key=lambda e: e.name):
            if item.name.startswith(".") or item.name in ("node_modules", "__pycache__"):
                continue
            full = item.path
            is_dir = item.is_dir(follow_symlinks=True)
            is_file = item.is_file(follow_symlinks=True)

            if is_dir:
                entries.extend(_collect_skill_entries(full, False))
            elif is_file:
                if include_root_files and item.name.endswith(".md"):
                    entries.append(full)
                elif not include_root_files and item.name == "SKILL.md":
                    entries.append(full)
    except OSError:
        pass
    return entries


def _collect_resource_files(dir_path: str, resource_type: ResourceType) -> list[str]:
    if resource_type == "skills":
        return _collect_skill_entries(dir_path)
    return _collect_files(dir_path, _FILE_PATTERNS[resource_type])


def _apply_patterns(all_paths: list[str], patterns: list[str], base_dir: str) -> set[str]:
    includes: list[str] = []
    excludes: list[str] = []
    force_includes: list[str] = []
    force_excludes: list[str] = []

    for p in patterns:
        if p.startswith("+"):
            force_includes.append(p[1:])
        elif p.startswith("-"):
            force_excludes.append(p[1:])
        elif p.startswith("!"):
            excludes.append(p[1:])
        else:
            includes.append(p)

    def _matches(path: str, pats: list[str]) -> bool:
        rel = os.path.relpath(path, base_dir)
        name = os.path.basename(path)
        return any(
            fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(path, pat)
            for pat in pats
        )

    result = list(all_paths) if not includes else [p for p in all_paths if _matches(p, includes)]
    if excludes:
        result = [p for p in result if not _matches(p, excludes)]
    if force_includes:
        for fp in all_paths:
            if fp not in result and _matches(fp, force_includes):
                result.append(fp)
    if force_excludes:
        result = [p for p in result if not _matches(p, force_excludes)]
    return set(result)


class DefaultPackageManager:
    """Resolves, installs, updates, and removes agent resource packages."""

    def __init__(
        self,
        cwd: str,
        agent_dir: str,
        settings_manager: Any = None,
    ) -> None:
        from pi_coding_agent.config import CONFIG_DIR_NAME

        self._cwd = cwd
        self._agent_dir = agent_dir
        self._settings_manager = settings_manager
        self._config_dir_name = CONFIG_DIR_NAME
        self._progress_callback: ProgressCallback | None = None

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        self._progress_callback = callback

    def _emit(self, event: ProgressEvent) -> None:
        if self._progress_callback:
            self._progress_callback(event)

    async def _with_progress(
        self,
        action: str,
        source: str,
        message: str,
        operation: Any,
    ) -> None:
        self._emit(ProgressEvent(type="start", action=action, source=source, message=message))
        try:
            await operation()
            self._emit(ProgressEvent(type="complete", action=action, source=source))
        except Exception as e:
            self._emit(ProgressEvent(type="error", action=action, source=source, message=str(e)))
            raise

    def _parse_source(self, source: str) -> _ParsedSource:
        if source.startswith("npm:"):
            spec = source[4:].strip()
            name, version = self._parse_npm_spec(spec)
            return _NpmSource(spec=spec, name=name, pinned=bool(version))

        trimmed = source.strip()
        is_win_abs = re.match(r"^[A-Za-z]:[\\/]|^\\\\", trimmed)
        is_local = (
            trimmed.startswith(".")
            or trimmed.startswith("/")
            or trimmed in ("~", )
            or trimmed.startswith("~/")
            or bool(is_win_abs)
        )
        if is_local:
            return _LocalSource(path=source)

        git_parsed = parse_git_url(source)
        if git_parsed:
            return git_parsed

        return _LocalSource(path=source)

    def _parse_npm_spec(self, spec: str) -> tuple[str, str | None]:
        m = re.match(r"^(@?[^@]+(?:/[^@]+)?)(?:@(.+))?$", spec)
        if not m:
            return spec, None
        return m.group(1) or spec, m.group(2)

    def _resolve_path(self, input_path: str) -> str:
        home = os.path.expanduser("~")
        t = input_path.strip()
        if t == "~":
            return home
        if t.startswith("~/"):
            return os.path.join(home, t[2:])
        if t.startswith("~"):
            return os.path.join(home, t[1:])
        return os.path.abspath(os.path.join(self._cwd, t))

    def _resolve_path_from_base(self, input_path: str, base_dir: str) -> str:
        home = os.path.expanduser("~")
        t = input_path.strip()
        if t == "~":
            return home
        if t.startswith("~/"):
            return os.path.join(home, t[2:])
        if t.startswith("~"):
            return os.path.join(home, t[1:])
        return os.path.abspath(os.path.join(base_dir, t))

    def _get_base_dir_for_scope(self, scope: SourceScope) -> str:
        if scope == "project":
            return os.path.join(self._cwd, self._config_dir_name)
        if scope == "user":
            return self._agent_dir
        return self._cwd

    def _get_npm_install_root(self, scope: SourceScope, temporary: bool) -> str:
        if temporary:
            return self._get_temporary_dir("npm")
        if scope == "project":
            return os.path.join(self._cwd, self._config_dir_name, "npm")
        return os.path.expanduser("~/.npm")

    def _get_npm_install_path(self, source: _NpmSource, scope: SourceScope) -> str:
        if scope == "temporary":
            return os.path.join(self._get_temporary_dir("npm"), "node_modules", source.name)
        if scope == "project":
            return os.path.join(
                self._cwd, self._config_dir_name, "npm", "node_modules", source.name
            )
        npm_root = self._get_global_npm_root()
        return os.path.join(npm_root, source.name)

    def _get_global_npm_root(self) -> str:
        try:
            result = subprocess.run(
                ["npm", "root", "-g"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except Exception:
            return os.path.expanduser("~/.npm/lib/node_modules")

    def _get_git_install_path(self, source: GitSource, scope: SourceScope) -> str:
        if scope == "temporary":
            return self._get_temporary_dir(f"git-{source.host}", source.path)
        if scope == "project":
            return os.path.join(
                self._cwd, self._config_dir_name, "git", source.host, source.path
            )
        return os.path.join(self._agent_dir, "git", source.host, source.path)

    def _get_temporary_dir(self, prefix: str, suffix: str | None = None) -> str:
        key = f"{prefix}-{suffix or ''}"
        hash_val = hashlib.sha256(key.encode()).hexdigest()[:8]
        return os.path.join(tempfile.gettempdir(), "pi-extensions", prefix, hash_val, suffix or "")

    def get_installed_path(self, source: str, scope: SourceScope) -> str | None:
        parsed = self._parse_source(source)
        if isinstance(parsed, _NpmSource):
            path = self._get_npm_install_path(parsed, scope)
            return path if os.path.exists(path) else None
        if isinstance(parsed, GitSource):
            path = self._get_git_install_path(parsed, scope)
            return path if os.path.exists(path) else None
        if isinstance(parsed, _LocalSource):
            base = self._get_base_dir_for_scope(scope)
            path = self._resolve_path_from_base(parsed.path, base)
            return path if os.path.exists(path) else None
        return None

    async def resolve(
        self,
        on_missing: Callable[[str], Any] | None = None,
    ) -> ResolvedPaths:
        """Resolve all configured packages to resource paths."""
        accumulator: dict[ResourceType, dict[str, tuple[PathMetadata, bool]]] = {
            "extensions": {}, "skills": {}, "prompts": {}, "themes": {}
        }

        if self._settings_manager:
            global_settings = self._settings_manager.get_global_settings()
            project_settings = self._settings_manager.get_project_settings()

            all_packages: list[tuple[Any, SourceScope]] = []
            for pkg in global_settings.get("packages", []):
                all_packages.append((pkg, "user"))
            for pkg in project_settings.get("packages", []):
                all_packages.append((pkg, "project"))

            for pkg, scope in all_packages:
                source_str = pkg if isinstance(pkg, str) else pkg.get("source", "")
                await self._resolve_source_to_accumulator(
                    source_str, scope, accumulator, on_missing
                )

        return self._to_resolved_paths(accumulator)

    async def resolve_extension_sources(
        self,
        sources: list[str],
        options: dict[str, Any] | None = None,
    ) -> ResolvedPaths:
        opts = options or {}
        scope: SourceScope = "temporary" if opts.get("temporary") else ("project" if opts.get("local") else "user")
        accumulator: dict[ResourceType, dict[str, tuple[PathMetadata, bool]]] = {
            "extensions": {}, "skills": {}, "prompts": {}, "themes": {}
        }
        for source in sources:
            await self._resolve_source_to_accumulator(source, scope, accumulator)
        return self._to_resolved_paths(accumulator)

    async def _resolve_source_to_accumulator(
        self,
        source_str: str,
        scope: SourceScope,
        accumulator: dict,
        on_missing: Callable | None = None,
    ) -> None:
        parsed = self._parse_source(source_str)
        metadata = PathMetadata(source=source_str, scope=scope, origin="package")

        if isinstance(parsed, _LocalSource):
            base = self._get_base_dir_for_scope(scope)
            resolved = self._resolve_path_from_base(parsed.path, base)
            if os.path.exists(resolved):
                if os.path.isfile(resolved):
                    metadata.base_dir = os.path.dirname(resolved)
                    self._add_to_accumulator(accumulator["extensions"], resolved, metadata, True)
                elif os.path.isdir(resolved):
                    metadata.base_dir = resolved
                    self._collect_package_resources(resolved, accumulator, metadata)
            return

        if isinstance(parsed, _NpmSource):
            install_path = self._get_npm_install_path(parsed, scope)
            if not os.path.exists(install_path):
                if on_missing:
                    action = await on_missing(source_str)
                    if action == "skip":
                        return
                    if action == "error":
                        raise RuntimeError(f"Missing source: {source_str}")
                await self._install_npm(parsed, scope, scope == "temporary")
            metadata.base_dir = install_path
            self._collect_package_resources(install_path, accumulator, metadata)
            return

        if isinstance(parsed, GitSource):
            install_path = self._get_git_install_path(parsed, scope)
            if not os.path.exists(install_path):
                if on_missing:
                    action = await on_missing(source_str)
                    if action == "skip":
                        return
                    if action == "error":
                        raise RuntimeError(f"Missing source: {source_str}")
                await self._install_git(parsed, scope)
            metadata.base_dir = install_path
            self._collect_package_resources(install_path, accumulator, metadata)

    def _collect_package_resources(
        self,
        package_root: str,
        accumulator: dict,
        metadata: PathMetadata,
    ) -> bool:
        manifest = self._read_pi_manifest(package_root)
        if manifest:
            for rt in RESOURCE_TYPES:
                entries = getattr(manifest, rt, [])
                self._add_manifest_entries(entries, package_root, rt, accumulator[rt], metadata)
            return True

        has_any = False
        for rt in RESOURCE_TYPES:
            d = os.path.join(package_root, rt)
            if os.path.isdir(d):
                files = _collect_resource_files(d, rt)
                for f in files:
                    self._add_to_accumulator(accumulator[rt], f, metadata, True)
                has_any = True
        return has_any

    def _add_manifest_entries(
        self,
        entries: list[str],
        root: str,
        resource_type: ResourceType,
        target: dict,
        metadata: PathMetadata,
    ) -> None:
        if not entries:
            return
        plain = [e for e in entries if not _is_pattern(e)]
        patterns = [e for e in entries if _is_pattern(e)]
        all_files: list[str] = []
        for e in plain:
            resolved = os.path.abspath(os.path.join(root, e))
            if os.path.isfile(resolved):
                all_files.append(resolved)
            elif os.path.isdir(resolved):
                all_files.extend(_collect_resource_files(resolved, resource_type))

        enabled = _apply_patterns(all_files, patterns, root)
        for f in all_files:
            if f in enabled:
                self._add_to_accumulator(target, f, metadata, True)

    def _read_pi_manifest(self, package_root: str) -> _PiManifest | None:
        pkg_json = os.path.join(package_root, "package.json")
        if not os.path.exists(pkg_json):
            return None
        try:
            with open(pkg_json) as f:
                data = json.load(f)
            pi_data = data.get("pi", {})
            if not isinstance(pi_data, dict):
                return None
            return _PiManifest(
                extensions=pi_data.get("extensions", []),
                skills=pi_data.get("skills", []),
                prompts=pi_data.get("prompts", []),
                themes=pi_data.get("themes", []),
            )
        except Exception:
            return None

    def _add_to_accumulator(
        self,
        target: dict,
        path: str,
        metadata: PathMetadata,
        enabled: bool,
    ) -> None:
        if path and path not in target:
            target[path] = (metadata, enabled)

    def _to_resolved_paths(self, accumulator: dict) -> ResolvedPaths:
        def _build(entries: dict) -> list[ResolvedResource]:
            return [
                ResolvedResource(path=p, enabled=e, metadata=m)
                for p, (m, e) in entries.items()
            ]

        return ResolvedPaths(
            extensions=_build(accumulator["extensions"]),
            skills=_build(accumulator["skills"]),
            prompts=_build(accumulator["prompts"]),
            themes=_build(accumulator["themes"]),
        )

    async def install(self, source: str, options: dict[str, Any] | None = None) -> None:
        parsed = self._parse_source(source)
        scope: SourceScope = "project" if (options or {}).get("local") else "user"
        await self._with_progress("install", source, f"Installing {source}...", lambda: self._do_install(parsed, scope))

    async def _do_install(self, parsed: _ParsedSource, scope: SourceScope) -> None:
        if isinstance(parsed, _NpmSource):
            await self._install_npm(parsed, scope, False)
        elif isinstance(parsed, GitSource):
            await self._install_git(parsed, scope)

    async def remove(self, source: str, options: dict[str, Any] | None = None) -> None:
        parsed = self._parse_source(source)
        scope: SourceScope = "project" if (options or {}).get("local") else "user"
        await self._with_progress("remove", source, f"Removing {source}...", lambda: self._do_remove(parsed, scope))

    async def _do_remove(self, parsed: _ParsedSource, scope: SourceScope) -> None:
        if isinstance(parsed, _NpmSource):
            await self._uninstall_npm(parsed, scope)
        elif isinstance(parsed, GitSource):
            install_path = self._get_git_install_path(parsed, scope)
            if os.path.exists(install_path):
                shutil.rmtree(install_path, ignore_errors=True)

    async def update(self, source: str | None = None) -> None:
        """Update all configured package sources or a specific source."""
        if not self._settings_manager:
            return

        global_settings = self._settings_manager.get_global_settings()
        project_settings = self._settings_manager.get_project_settings()
        target_identity = self._get_package_identity(source) if source else None

        for pkg in global_settings.get("packages", []):
            source_str = pkg if isinstance(pkg, str) else pkg.get("source", "")
            if not source_str:
                continue
            if target_identity and self._get_package_identity(source_str, "user") != target_identity:
                continue
            await self._update_source_for_scope(source_str, "user")

        for pkg in project_settings.get("packages", []):
            source_str = pkg if isinstance(pkg, str) else pkg.get("source", "")
            if not source_str:
                continue
            if target_identity and self._get_package_identity(source_str, "project") != target_identity:
                continue
            await self._update_source_for_scope(source_str, "project")

    async def _update_source_for_scope(self, source: str, scope: SourceScope) -> None:
        parsed = self._parse_source(source)
        if isinstance(parsed, _NpmSource):
            if parsed.pinned:
                return

            async def _op() -> None:
                await self._install_npm(parsed, scope, False)

            await self._with_progress("update", source, f"Updating {source}...", _op)
            return

        if isinstance(parsed, GitSource):
            if parsed.pinned:
                return

            async def _op() -> None:
                install_path = self._get_git_install_path(parsed, scope)
                if os.path.exists(install_path):
                    await self._run_command(["git", "pull", "--ff-only"], cwd=install_path)
                else:
                    await self._install_git(parsed, scope)

            await self._with_progress("update", source, f"Updating {source}...", _op)

    def _get_package_identity(self, source: str, scope: SourceScope = "user") -> str:
        parsed = self._parse_source(source)
        if isinstance(parsed, _NpmSource):
            return f"npm:{parsed.name}"
        if isinstance(parsed, GitSource):
            return f"git:{parsed.host}/{parsed.path}"
        base_dir = self._get_base_dir_for_scope(scope)
        resolved = self._resolve_path_from_base(parsed.path, base_dir)
        return f"local:{resolved}"

    async def _install_npm(self, source: _NpmSource, scope: SourceScope, temporary: bool) -> None:
        if scope == "user" and not temporary:
            await self._run_command(["npm", "install", "-g", source.spec])
        else:
            install_root = self._get_npm_install_root(scope, temporary)
            os.makedirs(install_root, exist_ok=True)
            await self._run_command(["npm", "install", source.spec, "--prefix", install_root])

    async def _uninstall_npm(self, source: _NpmSource, scope: SourceScope) -> None:
        if scope == "user":
            await self._run_command(["npm", "uninstall", "-g", source.name])
        else:
            install_root = self._get_npm_install_root(scope, False)
            if os.path.exists(install_root):
                await self._run_command(["npm", "uninstall", source.name, "--prefix", install_root])

    async def _install_git(self, source: GitSource, scope: SourceScope) -> None:
        target_dir = self._get_git_install_path(source, scope)
        if os.path.exists(target_dir):
            return
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        await self._run_command(["git", "clone", source.repo, target_dir])
        if source.ref:
            await self._run_command(["git", "checkout", source.ref], cwd=target_dir)
        pkg_json = os.path.join(target_dir, "package.json")
        if os.path.exists(pkg_json):
            await self._run_command(["npm", "install"], cwd=target_dir)

    async def _run_command(self, args: list[str], cwd: str | None = None) -> None:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"{' '.join(args)} failed with code {proc.returncode}: {stderr.decode()}"
            )

    def add_source_to_settings(self, source: str, options: dict[str, Any] | None = None) -> bool:
        if not self._settings_manager:
            return False
        scope: SourceScope = "project" if (options or {}).get("local") else "user"
        settings = (
            self._settings_manager.get_project_settings()
            if scope == "project"
            else self._settings_manager.get_global_settings()
        )
        packages = settings.get("packages", [])
        if any(
            (p if isinstance(p, str) else p.get("source", "")) == source
            for p in packages
        ):
            return False
        packages.append(source)
        if scope == "project":
            self._settings_manager.set_project_packages(packages)
        else:
            self._settings_manager.set_packages(packages)
        return True

    def remove_source_from_settings(self, source: str, options: dict[str, Any] | None = None) -> bool:
        if not self._settings_manager:
            return False
        scope: SourceScope = "project" if (options or {}).get("local") else "user"
        settings = (
            self._settings_manager.get_project_settings()
            if scope == "project"
            else self._settings_manager.get_global_settings()
        )
        packages = settings.get("packages", [])
        new_packages = [
            p for p in packages
            if (p if isinstance(p, str) else p.get("source", "")) != source
        ]
        if len(new_packages) == len(packages):
            return False
        if scope == "project":
            self._settings_manager.set_project_packages(new_packages)
        else:
            self._settings_manager.set_packages(new_packages)
        return True
