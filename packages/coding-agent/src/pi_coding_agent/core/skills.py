"""
Agent Skills loader.

Discovers, validates, and loads skill files from configured directories.
Each skill is a markdown file with YAML frontmatter describing its metadata.

Mirrors core/skills.ts
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pi_coding_agent.core.diagnostics import ResourceCollision, ResourceDiagnostic
from pi_coding_agent.utils.frontmatter import parse_frontmatter

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
IGNORE_FILE_NAMES = [".gitignore", ".ignore", ".fdignore"]


@dataclass
class Skill:
    name: str
    description: str
    file_path: str
    base_dir: str
    source: str
    disable_model_invocation: bool = False


@dataclass
class LoadSkillsResult:
    skills: list[Skill] = field(default_factory=list)
    diagnostics: list[ResourceDiagnostic] = field(default_factory=list)


def _validate_name(name: str, parent_dir_name: str) -> list[str]:
    errors: list[str] = []
    if name != parent_dir_name:
        errors.append(f'name "{name}" does not match parent directory "{parent_dir_name}"')
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    if not re.match(r"^[a-z0-9-]+$", name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: str | None) -> list[str]:
    errors: list[str] = []
    if not description or not description.strip():
        errors.append("description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(
            f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})"
        )
    return errors


def _load_ignore_patterns(dir_path: str, root_dir: str) -> list[str]:
    """Load ignore patterns from .gitignore / .ignore / .fdignore in dir_path."""
    patterns: list[str] = []
    rel = os.path.relpath(dir_path, root_dir)
    prefix = rel.replace(os.sep, "/") + "/" if rel and rel != "." else ""

    for fname in IGNORE_FILE_NAMES:
        fpath = os.path.join(dir_path, fname)
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\n")
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    patterns.append(prefix + stripped.lstrip("/"))
        except OSError:
            pass

    return patterns


def _is_ignored(rel_path: str, ignore_patterns: list[str]) -> bool:
    """Simple prefix / exact matching for ignore patterns."""
    normalized = rel_path.replace(os.sep, "/")
    for pat in ignore_patterns:
        pat_clean = pat.lstrip("!")
        if normalized == pat_clean or normalized.startswith(pat_clean.rstrip("/") + "/"):
            if pat.startswith("!"):
                return False
            return True
    return False


def _load_skill_from_file(
    file_path: str, source: str
) -> tuple[Skill | None, list[ResourceDiagnostic]]:
    diagnostics: list[ResourceDiagnostic] = []
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            raw_content = f.read()

        frontmatter, _ = parse_frontmatter(raw_content)
        skill_dir = os.path.dirname(file_path)
        parent_dir_name = os.path.basename(skill_dir)

        description = frontmatter.get("description", "")
        for err in _validate_description(description):
            diagnostics.append(ResourceDiagnostic(type="warning", message=err, path=file_path))

        name = frontmatter.get("name") or parent_dir_name
        for err in _validate_name(name, parent_dir_name):
            diagnostics.append(ResourceDiagnostic(type="warning", message=err, path=file_path))

        if not description or not str(description).strip():
            return None, diagnostics

        skill = Skill(
            name=str(name),
            description=str(description),
            file_path=file_path,
            base_dir=skill_dir,
            source=source,
            disable_model_invocation=bool(frontmatter.get("disable-model-invocation", False)),
        )
        return skill, diagnostics
    except Exception as exc:
        diagnostics.append(
            ResourceDiagnostic(type="warning", message=str(exc), path=file_path)
        )
        return None, diagnostics


def _load_skills_from_dir_internal(
    dir_path: str,
    source: str,
    include_root_files: bool,
    root_dir: str | None = None,
    ignore_patterns: list[str] | None = None,
) -> LoadSkillsResult:
    skills: list[Skill] = []
    diagnostics: list[ResourceDiagnostic] = []

    if not os.path.isdir(dir_path):
        return LoadSkillsResult()

    root = root_dir or dir_path
    patterns = list(ignore_patterns or [])
    patterns.extend(_load_ignore_patterns(dir_path, root))

    try:
        entries = sorted(os.scandir(dir_path), key=lambda e: e.name)
    except OSError:
        return LoadSkillsResult()

    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.name in ("node_modules", "__pycache__"):
            continue

        full_path = entry.path
        is_dir = False
        is_file = False

        try:
            if entry.is_symlink():
                stat_r = os.stat(full_path)
                is_dir = os.path.isdir(full_path)
                is_file = os.path.isfile(full_path)
            else:
                is_dir = entry.is_dir()
                is_file = entry.is_file()
        except OSError:
            continue

        rel_path = os.path.relpath(full_path, root).replace(os.sep, "/")
        ignore_key = rel_path + "/" if is_dir else rel_path
        if _is_ignored(ignore_key, patterns):
            continue

        if is_dir:
            sub = _load_skills_from_dir_internal(full_path, source, False, root, patterns)
            skills.extend(sub.skills)
            diagnostics.extend(sub.diagnostics)
            continue

        if not is_file:
            continue

        is_root_md = include_root_files and entry.name.endswith(".md")
        is_skill_md = (not include_root_files) and entry.name == "SKILL.md"
        if not is_root_md and not is_skill_md:
            continue

        skill, diags = _load_skill_from_file(full_path, source)
        if skill:
            skills.append(skill)
        diagnostics.extend(diags)

    return LoadSkillsResult(skills=skills, diagnostics=diagnostics)


def load_skills_from_dir(dir_path: str, source: str) -> LoadSkillsResult:
    """Load skills from a single directory."""
    return _load_skills_from_dir_internal(dir_path, source, include_root_files=True)


def format_skills_for_prompt(skills: list[Skill]) -> str:
    """Format skills list as XML for inclusion in a system prompt."""

    def _escape(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    visible = [s for s in skills if not s.disable_model_invocation]
    if not visible:
        return ""

    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory "
        "(parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
        "",
        "<available_skills>",
    ]
    for skill in visible:
        lines.append("  <skill>")
        lines.append(f"    <name>{_escape(skill.name)}</name>")
        lines.append(f"    <description>{_escape(skill.description)}</description>")
        lines.append(f"    <location>{_escape(skill.file_path)}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def _normalize_path(input_path: str) -> str:
    home = os.path.expanduser("~")
    trimmed = input_path.strip()
    if trimmed == "~":
        return home
    if trimmed.startswith("~/"):
        return os.path.join(home, trimmed[2:])
    if trimmed.startswith("~"):
        return os.path.join(home, trimmed[1:])
    return trimmed


@dataclass
class LoadSkillsOptions:
    cwd: str | None = None
    agent_dir: str | None = None
    skill_paths: list[str] = field(default_factory=list)
    include_defaults: bool = True


def load_skills(options: LoadSkillsOptions | None = None) -> LoadSkillsResult:
    """Load skills from all configured locations."""
    from pi_coding_agent.config import CONFIG_DIR_NAME, get_agent_dir

    opts = options or LoadSkillsOptions()
    cwd = opts.cwd or os.getcwd()
    agent_dir = opts.agent_dir or get_agent_dir()
    skill_paths = opts.skill_paths or []
    include_defaults = opts.include_defaults

    skill_map: dict[str, Skill] = {}
    real_path_set: set[str] = set()
    all_diagnostics: list[ResourceDiagnostic] = []
    collision_diagnostics: list[ResourceDiagnostic] = []

    def add_skills(result: LoadSkillsResult) -> None:
        all_diagnostics.extend(result.diagnostics)
        for skill in result.skills:
            try:
                real_path = os.path.realpath(skill.file_path)
            except OSError:
                real_path = skill.file_path

            if real_path in real_path_set:
                continue

            if skill.name in skill_map:
                collision_diagnostics.append(
                    ResourceDiagnostic(
                        type="collision",
                        message=f'name "{skill.name}" collision',
                        path=skill.file_path,
                        collision=ResourceCollision(
                            resource_type="skill",
                            name=skill.name,
                            winner_path=skill_map[skill.name].file_path,
                            loser_path=skill.file_path,
                        ),
                    )
                )
            else:
                skill_map[skill.name] = skill
                real_path_set.add(real_path)

    if include_defaults:
        add_skills(
            _load_skills_from_dir_internal(
                os.path.join(agent_dir, "skills"), "user", True
            )
        )
        add_skills(
            _load_skills_from_dir_internal(
                os.path.join(cwd, CONFIG_DIR_NAME, "skills"), "project", True
            )
        )

    user_skills_dir = os.path.join(agent_dir, "skills")
    project_skills_dir = os.path.join(cwd, CONFIG_DIR_NAME, "skills")

    def _is_under(target: str, root: str) -> bool:
        norm_root = os.path.abspath(root)
        norm_target = os.path.abspath(target)
        return norm_target == norm_root or norm_target.startswith(norm_root + os.sep)

    def get_source(resolved_path: str) -> str:
        if not include_defaults:
            if _is_under(resolved_path, user_skills_dir):
                return "user"
            if _is_under(resolved_path, project_skills_dir):
                return "project"
        return "path"

    for raw_path in skill_paths:
        resolved_path = (
            raw_path
            if os.path.isabs(_normalize_path(raw_path))
            else os.path.abspath(os.path.join(cwd, _normalize_path(raw_path)))
        )
        if not os.path.exists(resolved_path):
            all_diagnostics.append(
                ResourceDiagnostic(
                    type="warning",
                    message="skill path does not exist",
                    path=resolved_path,
                )
            )
            continue

        try:
            source = get_source(resolved_path)
            if os.path.isdir(resolved_path):
                add_skills(_load_skills_from_dir_internal(resolved_path, source, True))
            elif os.path.isfile(resolved_path) and resolved_path.endswith(".md"):
                skill, diags = _load_skill_from_file(resolved_path, source)
                if skill:
                    add_skills(LoadSkillsResult(skills=[skill], diagnostics=diags))
                else:
                    all_diagnostics.extend(diags)
            else:
                all_diagnostics.append(
                    ResourceDiagnostic(
                        type="warning",
                        message="skill path is not a markdown file",
                        path=resolved_path,
                    )
                )
        except Exception as exc:
            all_diagnostics.append(
                ResourceDiagnostic(
                    type="warning",
                    message=str(exc),
                    path=resolved_path,
                )
            )

    return LoadSkillsResult(
        skills=list(skill_map.values()),
        diagnostics=all_diagnostics + collision_diagnostics,
    )
