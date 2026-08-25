"""Skill loading — mirrors harness/skills.ts."""
from __future__ import annotations

import re
from typing import Any, Callable, Literal, TypedDict

import yaml

from pi_agent.harness.types import ExecutionEnv, FileInfo, Result, Skill, to_error

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
IGNORE_FILE_NAMES = [".gitignore", ".ignore", ".fdignore"]

SkillDiagnosticCode = Literal["file_info_failed", "list_failed", "read_failed", "parse_failed", "invalid_metadata"]


class SkillDiagnostic(TypedDict):
    type: Literal["warning"]
    code: SkillDiagnosticCode
    message: str
    path: str


class _IgnoreMatcher:
    def __init__(self) -> None:
        self._patterns: list[tuple[bool, str]] = []

    def add(self, patterns: list[str]) -> None:
        for raw in patterns:
            negated = False
            pattern = raw
            if pattern.startswith("!"):
                negated = True
                pattern = pattern[1:]
            self._patterns.append((negated, pattern))

    def ignores(self, path: str) -> bool:
        ignored = False
        normalized = path.replace("\\", "/")
        for negated, pattern in self._patterns:
            if _match_ignore(pattern, normalized):
                ignored = not negated
        return ignored


def _match_ignore(pattern: str, path: str) -> bool:
    directory_only = pattern.endswith("/")
    pattern = pattern.rstrip("/")
    if pattern.startswith("/"):
        pattern = pattern[1:]
    regex = _glob_to_re(pattern)
    if directory_only:
        return bool(regex.match(path.rstrip("/") + "/") or regex.match(path))
    return bool(regex.match(path) or regex.match(path.rstrip("/")))


def _glob_to_re(pattern: str) -> re.Pattern[str]:
    parts: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            parts.append("(?:.*/)?")
            i += 3
            continue
        if pattern[i : i + 2] == "**":
            parts.append(".*")
            i += 2
            continue
        char = pattern[i]
        if char == "*":
            parts.append("[^/]*")
        elif char == "?":
            parts.append("[^/]")
        else:
            parts.append(re.escape(char))
        i += 1
    return re.compile("^" + "".join(parts) + "$")


def format_skill_invocation(skill: Skill, additional_instructions: str | None = None) -> str:
    skill_block = (
        f'<skill name="{skill.name}" location="{skill.file_path}">\n'
        f"References are relative to {_dirname_env_path(skill.file_path)}.\n\n"
        f"{skill.content}\n</skill>"
    )
    return f"{skill_block}\n\n{additional_instructions}" if additional_instructions else skill_block


async def load_skills(env: ExecutionEnv, dirs: str | list[str]) -> dict[str, Any]:
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []
    for directory in dirs if isinstance(dirs, list) else [dirs]:
        root_info_result = await env.file_info(directory)
        if not root_info_result["ok"]:
            if root_info_result["error"].code != "not_found":
                diagnostics.append(
                    {
                        "type": "warning",
                        "code": "file_info_failed",
                        "message": str(root_info_result["error"]),
                        "path": directory,
                    }
                )
            continue
        root_info = root_info_result["value"]
        if await _resolve_kind(env, root_info, diagnostics) != "directory":
            continue
        result = await _load_skills_from_dir(
            env, _info_get(root_info, "path"), True, _IgnoreMatcher(), _info_get(root_info, "path")
        )
        skills.extend(result["skills"])
        diagnostics.extend(result["diagnostics"])
    return {"skills": skills, "diagnostics": diagnostics}


async def load_sourced_skills(
    env: ExecutionEnv,
    inputs: list[dict[str, Any]],
    map_skill: Callable[[Skill, Any], Skill] | None = None,
) -> dict[str, Any]:
    skills: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for item in inputs:
        result = await load_skills(env, item["path"])
        for skill in result["skills"]:
            skills.append({"skill": map_skill(skill, item["source"]) if map_skill else skill, "source": item["source"]})
        for diagnostic in result["diagnostics"]:
            diagnostics.append({**diagnostic, "source": item["source"]})
    return {"skills": skills, "diagnostics": diagnostics}


async def _load_skills_from_dir(
    env: ExecutionEnv,
    directory: str,
    include_root_files: bool,
    ignore_matcher: _IgnoreMatcher,
    root_dir: str,
) -> dict[str, Any]:
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []
    dir_info_result = await env.file_info(directory)
    if not dir_info_result["ok"]:
        if dir_info_result["error"].code != "not_found":
            diagnostics.append(
                {"type": "warning", "code": "file_info_failed", "message": str(dir_info_result["error"]), "path": directory}
            )
        return {"skills": skills, "diagnostics": diagnostics}
    dir_info = dir_info_result["value"]
    if await _resolve_kind(env, dir_info, diagnostics) != "directory":
        return {"skills": skills, "diagnostics": diagnostics}
    await _add_ignore_rules(env, ignore_matcher, directory, root_dir, diagnostics)
    entries_result = await env.list_dir(directory)
    if not entries_result["ok"]:
        diagnostics.append({"type": "warning", "code": "list_failed", "message": str(entries_result["error"]), "path": directory})
        return {"skills": skills, "diagnostics": diagnostics}
    entries = entries_result["value"]
    for entry in entries:
        if entry["name"] != "SKILL.md":
            continue
        kind = await _resolve_kind(env, entry, diagnostics)
        if kind != "file":
            continue
        rel_path = _relative_env_path(root_dir, _info_get(entry, "path"))
        if ignore_matcher.ignores(rel_path):
            continue
        result = await _load_skill_from_file(env, _info_get(entry, "path"), _info_get(dir_info, "name"))
        if result["skill"]:
            skills.append(result["skill"])
        diagnostics.extend(result["diagnostics"])
        return {"skills": skills, "diagnostics": diagnostics}

    for entry in sorted(entries, key=lambda item: item["name"]):
        if entry["name"].startswith(".") or entry["name"] == "node_modules":
            continue
        kind = await _resolve_kind(env, entry, diagnostics)
        if not kind:
            continue
        rel_path = _relative_env_path(root_dir, _info_get(entry, "path"))
        ignore_path = f"{rel_path}/" if kind == "directory" else rel_path
        if ignore_matcher.ignores(ignore_path):
            continue
        if kind == "directory":
            result = await _load_skills_from_dir(env, _info_get(entry, "path"), False, ignore_matcher, root_dir)
            skills.extend(result["skills"])
            diagnostics.extend(result["diagnostics"])
            continue
        if kind != "file" or not include_root_files or not entry["name"].endswith(".md"):
            continue
        result = await _load_skill_from_file(env, _info_get(entry, "path"), _info_get(dir_info, "name"))
        if result["skill"]:
            skills.append(result["skill"])
        diagnostics.extend(result["diagnostics"])
    return {"skills": skills, "diagnostics": diagnostics}


async def _add_ignore_rules(
    env: ExecutionEnv,
    matcher: _IgnoreMatcher,
    directory: str,
    root_dir: str,
    diagnostics: list[SkillDiagnostic],
) -> None:
    relative_dir = _relative_env_path(root_dir, directory)
    prefix = f"{relative_dir}/" if relative_dir else ""
    for filename in IGNORE_FILE_NAMES:
        ignore_path_result = await env.join_path([directory, filename])
        if not ignore_path_result["ok"]:
            diagnostics.append(
                {"type": "warning", "code": "file_info_failed", "message": str(ignore_path_result["error"]), "path": directory}
            )
            continue
        ignore_path = ignore_path_result["value"]
        info = await env.file_info(ignore_path)
        if not info["ok"]:
            if info["error"].code != "not_found":
                diagnostics.append(
                    {"type": "warning", "code": "file_info_failed", "message": str(info["error"]), "path": ignore_path}
                )
            continue
        if info["value"]["kind"] != "file":
            continue
        content = await env.read_text_file(ignore_path)
        if not content["ok"]:
            diagnostics.append({"type": "warning", "code": "read_failed", "message": str(content["error"]), "path": ignore_path})
            continue
        patterns = [prefixed for line in content["value"].splitlines() if (prefixed := _prefix_ignore_pattern(line, prefix))]
        if patterns:
            matcher.add(patterns)


def _prefix_ignore_pattern(line: str, prefix: str) -> str | None:
    trimmed = line.strip()
    if not trimmed:
        return None
    if trimmed.startswith("#") and not trimmed.startswith("\\#"):
        return None
    pattern = line
    negated = False
    if pattern.startswith("!"):
        negated = True
        pattern = pattern[1:]
    elif pattern.startswith("\\!"):
        pattern = pattern[1:]
    if pattern.startswith("/"):
        pattern = pattern[1:]
    prefixed = f"{prefix}{pattern}" if prefix else pattern
    return f"!{prefixed}" if negated else prefixed


async def _load_skill_from_file(env: ExecutionEnv, file_path: str, parent_dir_name: str) -> dict[str, Any]:
    diagnostics: list[SkillDiagnostic] = []
    is_declared = file_path.rstrip("\\/").split("/")[-1].split("\\")[-1] == "SKILL.md"
    raw = await env.read_text_file(file_path)
    if not raw["ok"]:
        diagnostics.append({"type": "warning", "code": "read_failed", "message": str(raw["error"]), "path": file_path})
        return {"skill": None, "diagnostics": diagnostics}
    parsed = _parse_frontmatter(raw["value"])
    if not parsed["ok"]:
        if is_declared:
            diagnostics.append({"type": "warning", "code": "parse_failed", "message": str(parsed["error"]), "path": file_path})
        return {"skill": None, "diagnostics": diagnostics}
    frontmatter = parsed["value"]["frontmatter"]
    body = parsed["value"]["body"]
    description = frontmatter.get("description") if isinstance(frontmatter.get("description"), str) else None
    if not is_declared and (not description or not description.strip()):
        return {"skill": None, "diagnostics": diagnostics}
    for error in _validate_description(description):
        diagnostics.append({"type": "warning", "code": "invalid_metadata", "message": error, "path": file_path})
    frontmatter_name = frontmatter.get("name") if isinstance(frontmatter.get("name"), str) else None
    name = frontmatter_name or parent_dir_name
    for error in _validate_name(name, parent_dir_name):
        diagnostics.append({"type": "warning", "code": "invalid_metadata", "message": error, "path": file_path})
    if not description or not description.strip():
        return {"skill": None, "diagnostics": diagnostics}
    return {
        "skill": Skill(
            name=name,
            description=description,
            content=body,
            file_path=file_path,
            disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
        ),
        "diagnostics": diagnostics,
    }


def _validate_name(name: str, parent_dir_name: str) -> list[str]:
    errors: list[str] = []
    if name != parent_dir_name:
        errors.append(f'name "{name}" does not match parent directory "{parent_dir_name}"')
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: str | None) -> list[str]:
    if not description or not description.strip():
        return ["description is required"]
    if len(description) > MAX_DESCRIPTION_LENGTH:
        return [f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})"]
    return []


def _parse_frontmatter(content: str) -> Result:
    try:
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.startswith("---"):
            return {"ok": True, "value": {"frontmatter": {}, "body": normalized}}
        end_index = normalized.find("\n---", 3)
        if end_index == -1:
            return {"ok": True, "value": {"frontmatter": {}, "body": normalized}}
        yaml_string = normalized[4:end_index]
        body = normalized[end_index + 4 :].strip()
        return {"ok": True, "value": {"frontmatter": yaml.safe_load(yaml_string) or {}, "body": body}}
    except Exception as error:
        return {"ok": False, "error": to_error(error)}


def _info_get(info: FileInfo | dict, key: str) -> Any:
    if isinstance(info, dict):
        return info.get(key)
    return getattr(info, key, None)


async def _resolve_kind(env: ExecutionEnv, info: FileInfo, diagnostics: list[SkillDiagnostic]) -> str | None:
    kind = _info_get(info, "kind")
    path = _info_get(info, "path")
    if kind in ("file", "directory"):
        return kind
    canonical = await env.canonical_path(path)
    if not canonical["ok"]:
        if canonical["error"].code != "not_found":
            diagnostics.append(
                {"type": "warning", "code": "file_info_failed", "message": str(canonical["error"]), "path": path}
            )
        return None
    target = await env.file_info(canonical["value"])
    if not target["ok"]:
        if target["error"].code != "not_found":
            diagnostics.append(
                {"type": "warning", "code": "file_info_failed", "message": str(target["error"]), "path": path}
            )
        return None
    value = target["value"]
    target_kind = _info_get(value, "kind")
    return target_kind if target_kind in ("file", "directory") else None


def _dirname_env_path(path: str) -> str:
    normalized = re.sub(r"[\\/]+$", "", path)
    separator_index = max(normalized.rfind("/"), normalized.rfind("\\"))
    if separator_index == 2 and len(normalized) > 1 and normalized[1] == ":":
        return normalized[:3]
    return "/" if separator_index <= 0 else normalized[:separator_index]


def _relative_env_path(root: str, path: str) -> str:
    normalized_root = root.replace("\\", "/").rstrip("/")
    normalized_path = path.replace("\\", "/").rstrip("/")
    if normalized_path == normalized_root:
        return ""
    if normalized_path.startswith(f"{normalized_root}/"):
        return normalized_path[len(normalized_root) + 1 :]
    return re.sub(r"^/+", "", normalized_path)
