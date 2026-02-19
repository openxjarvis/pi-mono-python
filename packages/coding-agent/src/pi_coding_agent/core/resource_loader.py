"""
Resource loader for skills, prompts, themes, extensions, and AGENTS.md files.

Mirrors core/resource-loader.ts
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from pi_coding_agent.core.diagnostics import ResourceDiagnostic
from pi_coding_agent.core.prompt_templates import (
    LoadPromptTemplatesOptions,
    PromptTemplate,
    load_prompt_templates,
)
from pi_coding_agent.core.skills import LoadSkillsOptions, Skill, load_skills


@dataclass
class PathMetadata:
    source: str
    scope: str  # "user" | "project" | "temporary"
    origin: str  # "package" | "top-level"
    base_dir: str | None = None


@dataclass
class ResourceExtensionPaths:
    skill_paths: list[dict[str, Any]] = field(default_factory=list)
    prompt_paths: list[dict[str, Any]] = field(default_factory=list)
    theme_paths: list[dict[str, Any]] = field(default_factory=list)


_CONTEXT_CANDIDATES = ["AGENTS.md", "CLAUDE.md"]


def _load_context_file_from_dir(dir_path: str) -> dict[str, str] | None:
    """Load the first AGENTS.md or CLAUDE.md found in dir_path."""
    for filename in _CONTEXT_CANDIDATES:
        fpath = os.path.join(dir_path, filename)
        if os.path.exists(fpath):
            try:
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    return {"path": fpath, "content": f.read()}
            except OSError as e:
                print(f"Warning: Could not read {fpath}: {e}")
    return None


def _load_project_context_files(
    cwd: str | None = None, agent_dir: str | None = None
) -> list[dict[str, str]]:
    """Load AGENTS.md / CLAUDE.md from global and project ancestors."""
    from pi_coding_agent.config import get_agent_dir

    resolved_cwd = cwd or os.getcwd()
    resolved_agent_dir = agent_dir or get_agent_dir()

    context_files: list[dict[str, str]] = []
    seen: set[str] = set()

    global_ctx = _load_context_file_from_dir(resolved_agent_dir)
    if global_ctx:
        context_files.append(global_ctx)
        seen.add(global_ctx["path"])

    ancestor_files: list[dict[str, str]] = []
    current = resolved_cwd
    root = os.path.abspath("/")

    while True:
        ctx = _load_context_file_from_dir(current)
        if ctx and ctx["path"] not in seen:
            ancestor_files.insert(0, ctx)
            seen.add(ctx["path"])

        if os.path.abspath(current) == root:
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    context_files.extend(ancestor_files)
    return context_files


def _resolve_prompt_input(input_path: str | None, description: str) -> str | None:
    if not input_path:
        return None
    if os.path.exists(input_path):
        try:
            with open(input_path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError as e:
            print(f"Warning: Could not read {description} file {input_path}: {e}")
            return input_path
    return input_path


@dataclass
class Theme:
    """A loaded theme."""
    name: str
    path: str
    colors: dict[str, Any] = field(default_factory=dict)


@dataclass
class DefaultResourceLoaderOptions:
    cwd: str | None = None
    agent_dir: str | None = None
    settings_manager: Any = None
    event_bus: Any = None
    additional_extension_paths: list[str] = field(default_factory=list)
    additional_skill_paths: list[str] = field(default_factory=list)
    additional_prompt_template_paths: list[str] = field(default_factory=list)
    additional_theme_paths: list[str] = field(default_factory=list)
    extension_factories: list[Any] = field(default_factory=list)
    no_extensions: bool = False
    no_skills: bool = False
    no_prompt_templates: bool = False
    no_themes: bool = False
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    extensions_override: Callable | None = None
    skills_override: Callable | None = None
    prompts_override: Callable | None = None
    themes_override: Callable | None = None
    agents_files_override: Callable | None = None
    system_prompt_override: Callable | None = None
    append_system_prompt_override: Callable | None = None


class DefaultResourceLoader:
    """Loads and manages agent resources (skills, prompts, themes, extensions, AGENTS files)."""

    def __init__(self, options: DefaultResourceLoaderOptions | None = None) -> None:
        from pi_coding_agent.config import CONFIG_DIR_NAME, get_agent_dir

        opts = options or DefaultResourceLoaderOptions()
        self._cwd = opts.cwd or os.getcwd()
        self._agent_dir = opts.agent_dir or get_agent_dir()
        self._config_dir_name = CONFIG_DIR_NAME
        self._settings_manager = opts.settings_manager
        self._event_bus = opts.event_bus
        self._additional_extension_paths = list(opts.additional_extension_paths)
        self._additional_skill_paths = list(opts.additional_skill_paths)
        self._additional_prompt_paths = list(opts.additional_prompt_template_paths)
        self._additional_theme_paths = list(opts.additional_theme_paths)
        self._extension_factories = list(opts.extension_factories)
        self._no_extensions = opts.no_extensions
        self._no_skills = opts.no_skills
        self._no_prompt_templates = opts.no_prompt_templates
        self._no_themes = opts.no_themes
        self._system_prompt_source = opts.system_prompt
        self._append_system_prompt_source = opts.append_system_prompt
        self._extensions_override = opts.extensions_override
        self._skills_override = opts.skills_override
        self._prompts_override = opts.prompts_override
        self._themes_override = opts.themes_override
        self._agents_files_override = opts.agents_files_override
        self._system_prompt_override = opts.system_prompt_override
        self._append_system_prompt_override = opts.append_system_prompt_override

        self._extensions_result: dict[str, Any] = {"extensions": [], "diagnostics": []}
        self._skills: list[Skill] = []
        self._skill_diagnostics: list[ResourceDiagnostic] = []
        self._prompts: list[PromptTemplate] = []
        self._prompt_diagnostics: list[ResourceDiagnostic] = []
        self._themes: list[Theme] = []
        self._theme_diagnostics: list[ResourceDiagnostic] = []
        self._agents_files: list[dict[str, str]] = []
        self._system_prompt: str | None = None
        self._append_system_prompt: list[str] = []
        self._path_metadata: dict[str, PathMetadata] = {}
        self._last_skill_paths: list[str] = []
        self._last_prompt_paths: list[str] = []
        self._last_theme_paths: list[str] = []
        self._last_extension_paths: list[str] = []

    def get_extensions(self) -> dict[str, Any]:
        """Get loaded extensions result. Mirrors getExtensions() in TypeScript."""
        return dict(self._extensions_result)

    def get_skills(self) -> dict[str, Any]:
        return {"skills": self._skills, "diagnostics": self._skill_diagnostics}

    def get_prompts(self) -> dict[str, Any]:
        return {"prompts": self._prompts, "diagnostics": self._prompt_diagnostics}

    def get_themes(self) -> dict[str, Any]:
        """Get loaded themes. Mirrors getThemes() in TypeScript."""
        return {"themes": self._themes, "diagnostics": self._theme_diagnostics}

    def get_agents_files(self) -> dict[str, Any]:
        return {"agentsFiles": self._agents_files, "agents_files": self._agents_files}

    def get_system_prompt(self) -> str | None:
        return self._system_prompt

    def get_append_system_prompt(self) -> list[str]:
        return self._append_system_prompt

    def get_path_metadata(self) -> dict[str, PathMetadata]:
        return self._path_metadata

    def extend_resources(self, paths: ResourceExtensionPaths) -> None:
        if paths.skill_paths:
            new_paths = [entry["path"] for entry in paths.skill_paths]
            self._last_skill_paths = self._merge_paths(self._last_skill_paths, new_paths)
            self._update_skills_from_paths(self._last_skill_paths)

        if paths.prompt_paths:
            new_paths = [entry["path"] for entry in paths.prompt_paths]
            self._last_prompt_paths = self._merge_paths(self._last_prompt_paths, new_paths)
            self._update_prompts_from_paths(self._last_prompt_paths)

        if paths.theme_paths:
            new_paths = [entry["path"] for entry in paths.theme_paths]
            self._last_theme_paths = self._merge_paths(self._last_theme_paths, new_paths)
            self._update_themes_from_paths(self._last_theme_paths)

    async def reload(self) -> None:
        """Reload all resources from disk."""
        self._path_metadata = {}

        # Load extensions (from package manager + additional paths)
        if not self._no_extensions:
            await self._load_extensions()

        # Load skills
        skill_paths = self._resolve_resource_paths_from_settings("skills") + self._additional_skill_paths
        merged_skill_paths = self._merge_paths([], skill_paths)
        self._last_skill_paths = merged_skill_paths
        self._update_skills_from_paths(merged_skill_paths)

        # Load prompt templates
        prompt_paths = self._resolve_resource_paths_from_settings("prompts") + self._additional_prompt_paths
        merged_prompt_paths = self._merge_paths([], prompt_paths)
        self._last_prompt_paths = merged_prompt_paths
        self._update_prompts_from_paths(merged_prompt_paths)

        # Load themes
        if not self._no_themes:
            theme_paths = self._resolve_resource_paths_from_settings("themes") + self._additional_theme_paths
            merged_theme_paths = self._merge_paths([], theme_paths)
            self._last_theme_paths = merged_theme_paths
            self._update_themes_from_paths(merged_theme_paths)

        # Load AGENTS.md context files
        agents_files_base = {"agentsFiles": _load_project_context_files(self._cwd, self._agent_dir),
                             "agents_files": _load_project_context_files(self._cwd, self._agent_dir)}
        resolved = (
            self._agents_files_override(agents_files_base)
            if self._agents_files_override
            else agents_files_base
        )
        self._agents_files = resolved.get("agentsFiles") or resolved.get("agents_files", [])

        # System prompt
        base_system = _resolve_prompt_input(
            self._system_prompt_source or self._discover_system_prompt_file(),
            "system prompt",
        )
        self._system_prompt = (
            self._system_prompt_override(base_system)
            if self._system_prompt_override
            else base_system
        )

        append_source = self._append_system_prompt_source or self._discover_append_system_prompt_file()
        resolved_append = _resolve_prompt_input(append_source, "append system prompt")
        base_append = [resolved_append] if resolved_append else []
        self._append_system_prompt = (
            self._append_system_prompt_override(base_append)
            if self._append_system_prompt_override
            else base_append
        )

    def _resolve_resource_paths_from_settings(self, resource_type: str) -> list[str]:
        """Get additional resource paths from settings manager."""
        if not self._settings_manager:
            return []
        try:
            getter = getattr(self._settings_manager, f"get_{resource_type}", None)
            if callable(getter):
                val = getter()
                if isinstance(val, list):
                    return [str(p) for p in val if isinstance(p, str)]
        except Exception:
            pass
        return []

    async def _load_extensions(self) -> None:
        """
        Load extensions from settings paths + additional paths.
        Detects conflicts in tool/command/flag names.
        Mirrors extension loading in DefaultResourceLoader.reload() in TypeScript.
        """
        # Get extension paths from settings
        ext_paths = self._resolve_resource_paths_from_settings("extensions") + self._additional_extension_paths

        if not ext_paths and not self._extension_factories:
            base_result: dict[str, Any] = {"extensions": [], "diagnostics": []}
        else:
            try:
                from pi_coding_agent.core.extensions.loader import load_extensions
                event_bus = self._event_bus
                if event_bus is None:
                    from pi_coding_agent.core.event_bus import create_event_bus
                    event_bus = create_event_bus()
                base_result = await load_extensions(
                    ext_paths,
                    self._cwd,
                    event_bus,
                )
                # Detect conflicts
                base_result = self._detect_extension_conflicts(base_result)
            except Exception as e:
                base_result = {"extensions": [], "diagnostics": [{"type": "error", "message": str(e)}]}

        resolved = (
            self._extensions_override(base_result)
            if self._extensions_override
            else base_result
        )
        self._extensions_result = resolved

    def _detect_extension_conflicts(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Detect name collisions in tools, commands, and flags across extensions.
        Mirrors conflict detection in TypeScript.
        """
        seen_tools: dict[str, str] = {}
        seen_commands: dict[str, str] = {}
        seen_flags: dict[str, str] = {}
        diagnostics: list[dict[str, Any]] = list(result.get("diagnostics", []))
        extensions: list[Any] = result.get("extensions", [])

        for ext in extensions:
            ext_path = getattr(ext, "path", "") or ""

            for tool in getattr(ext, "tools", []) or []:
                name = getattr(tool, "name", "") or ""
                if name in seen_tools:
                    diagnostics.append({
                        "type": "collision",
                        "message": f'Tool name "{name}" collision between {seen_tools[name]} and {ext_path}',
                        "path": ext_path,
                    })
                else:
                    seen_tools[name] = ext_path

            for cmd in getattr(ext, "commands", []) or []:
                name = getattr(cmd, "name", "") or ""
                if name in seen_commands:
                    diagnostics.append({
                        "type": "collision",
                        "message": f'Command name "{name}" collision between {seen_commands[name]} and {ext_path}',
                        "path": ext_path,
                    })
                else:
                    seen_commands[name] = ext_path

            for flag in getattr(ext, "flags", []) or []:
                name = getattr(flag, "name", "") or ""
                if name in seen_flags:
                    diagnostics.append({
                        "type": "collision",
                        "message": f'Flag name "{name}" collision between {seen_flags[name]} and {ext_path}',
                        "path": ext_path,
                    })
                else:
                    seen_flags[name] = ext_path

        return {**result, "diagnostics": diagnostics}

    def _update_themes_from_paths(self, theme_paths: list[str]) -> None:
        """Load themes from paths."""
        themes: list[Theme] = []
        diagnostics: list[ResourceDiagnostic] = []

        for path in theme_paths:
            if not os.path.exists(path):
                continue
            try:
                if os.path.isfile(path) and path.endswith(".json"):
                    import json
                    with open(path, encoding="utf-8") as f:
                        colors = json.load(f)
                    name = os.path.splitext(os.path.basename(path))[0]
                    themes.append(Theme(name=name, path=path, colors=colors))
                elif os.path.isdir(path):
                    for fname in os.listdir(path):
                        if fname.endswith(".json"):
                            fpath = os.path.join(path, fname)
                            try:
                                import json
                                with open(fpath, encoding="utf-8") as f:
                                    colors = json.load(f)
                                name = os.path.splitext(fname)[0]
                                themes.append(Theme(name=name, path=fpath, colors=colors))
                            except Exception as e:
                                diagnostics.append(ResourceDiagnostic(
                                    type="error",
                                    message=f"Failed to load theme {fpath}: {e}",
                                    path=fpath,
                                ))
            except Exception as e:
                diagnostics.append(ResourceDiagnostic(
                    type="error",
                    message=f"Failed to load theme from {path}: {e}",
                    path=path,
                ))

        themes_result: dict[str, Any] = {"themes": themes, "diagnostics": diagnostics}
        resolved = self._themes_override(themes_result) if self._themes_override else themes_result
        self._themes = resolved["themes"]
        self._theme_diagnostics = resolved.get("diagnostics", [])

    def _update_skills_from_paths(self, skill_paths: list[str]) -> None:
        if self._no_skills and not skill_paths:
            skills_result = {"skills": [], "diagnostics": []}
        else:
            result = load_skills(
                LoadSkillsOptions(
                    cwd=self._cwd,
                    agent_dir=self._agent_dir,
                    skill_paths=skill_paths,
                    include_defaults=not self._no_skills,
                )
            )
            skills_result = {"skills": result.skills, "diagnostics": result.diagnostics}

        resolved = (
            self._skills_override(skills_result)
            if self._skills_override
            else skills_result
        )
        self._skills = resolved["skills"]
        self._skill_diagnostics = resolved["diagnostics"]

    def _update_prompts_from_paths(self, prompt_paths: list[str]) -> None:
        if self._no_prompt_templates and not prompt_paths:
            prompts_result: dict[str, Any] = {"prompts": [], "diagnostics": []}
        else:
            all_prompts = load_prompt_templates(
                LoadPromptTemplatesOptions(
                    cwd=self._cwd,
                    agent_dir=self._agent_dir,
                    prompt_paths=prompt_paths,
                    include_defaults=not self._no_prompt_templates,
                )
            )
            deduped = self._dedupe_prompts(all_prompts)
            prompts_result = deduped

        resolved = (
            self._prompts_override(prompts_result)
            if self._prompts_override
            else prompts_result
        )
        self._prompts = resolved["prompts"]
        self._prompt_diagnostics = resolved.get("diagnostics", [])

    def _dedupe_prompts(
        self, prompts: list[PromptTemplate]
    ) -> dict[str, Any]:
        seen: dict[str, PromptTemplate] = {}
        diagnostics: list[ResourceDiagnostic] = []
        for prompt in prompts:
            if prompt.name in seen:
                diagnostics.append(
                    ResourceDiagnostic(
                        type="collision",
                        message=f'name "/{prompt.name}" collision',
                        path=prompt.file_path,
                    )
                )
            else:
                seen[prompt.name] = prompt
        return {"prompts": list(seen.values()), "diagnostics": diagnostics}

    def _merge_paths(self, primary: list[str], additional: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for p in primary + additional:
            resolved = self._resolve_resource_path(p)
            if resolved not in seen:
                seen.add(resolved)
                merged.append(resolved)
        return merged

    def _resolve_resource_path(self, p: str) -> str:
        home = os.path.expanduser("~")
        t = p.strip()
        if t == "~":
            expanded = home
        elif t.startswith("~/"):
            expanded = os.path.join(home, t[2:])
        elif t.startswith("~"):
            expanded = os.path.join(home, t[1:])
        else:
            expanded = t
        return os.path.abspath(os.path.join(self._cwd, expanded))

    def _discover_system_prompt_file(self) -> str | None:
        from pi_coding_agent.config import CONFIG_DIR_NAME

        project_path = os.path.join(self._cwd, CONFIG_DIR_NAME, "SYSTEM.md")
        if os.path.exists(project_path):
            return project_path
        global_path = os.path.join(self._agent_dir, "SYSTEM.md")
        if os.path.exists(global_path):
            return global_path
        return None

    def _discover_append_system_prompt_file(self) -> str | None:
        from pi_coding_agent.config import CONFIG_DIR_NAME

        project_path = os.path.join(self._cwd, CONFIG_DIR_NAME, "APPEND_SYSTEM.md")
        if os.path.exists(project_path):
            return project_path
        global_path = os.path.join(self._agent_dir, "APPEND_SYSTEM.md")
        if os.path.exists(global_path):
            return global_path
        return None
