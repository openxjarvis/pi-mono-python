"""
Project trust store — mirrors packages/coding-agent/src/core/trust-manager.ts
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from pi_coding_agent.config import CONFIG_DIR_NAME, get_agent_dir
from pi_coding_agent.utils.text import strip_bom

ProjectTrustDecision = bool | None

TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES = (
    "settings.json",
    "extensions",
    "skills",
    "prompts",
    "themes",
    "SYSTEM.md",
    "APPEND_SYSTEM.md",
)


@dataclass
class ProjectTrustStoreEntry:
    path: str
    decision: bool


@dataclass
class ProjectTrustUpdate:
    path: str
    decision: ProjectTrustDecision


@dataclass
class ProjectTrustOption:
    label: str
    trusted: bool
    updates: list[ProjectTrustUpdate] = field(default_factory=list)
    saved_path: str | None = None


def canonicalize_path(path: str) -> str:
    try:
        return os.path.realpath(path)
    except OSError:
        return os.path.abspath(path)


def resolve_path(path: str, base_dir: str | None = None) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(base_dir or os.getcwd(), expanded))


def normalize_cwd(cwd: str) -> str:
    return canonicalize_path(resolve_path(cwd))


def get_project_trust_parent_path(cwd: str) -> str | None:
    trust_path = normalize_cwd(cwd)
    parent_dir = os.path.dirname(trust_path)
    return None if parent_dir == trust_path else parent_dir


def get_project_trust_options(
    cwd: str,
    *,
    include_session_only: bool = False,
) -> list[ProjectTrustOption]:
    trust_path = normalize_cwd(cwd)
    options = [
        ProjectTrustOption(
            label="Trust",
            trusted=True,
            updates=[ProjectTrustUpdate(path=trust_path, decision=True)],
            saved_path=trust_path,
        ),
    ]
    parent_path = get_project_trust_parent_path(cwd)
    if parent_path is not None:
        options.append(
            ProjectTrustOption(
                label=f"Trust parent folder ({parent_path})",
                trusted=True,
                updates=[
                    ProjectTrustUpdate(path=parent_path, decision=True),
                    ProjectTrustUpdate(path=trust_path, decision=None),
                ],
                saved_path=parent_path,
            )
        )
    if include_session_only:
        options.append(ProjectTrustOption(label="Trust (this session only)", trusted=True, updates=[]))
    options.append(
        ProjectTrustOption(
            label="Do not trust",
            trusted=False,
            updates=[ProjectTrustUpdate(path=trust_path, decision=False)],
            saved_path=trust_path,
        )
    )
    if include_session_only:
        options.append(ProjectTrustOption(label="Do not trust (this session only)", trusted=False, updates=[]))
    return options


def _find_nearest_trust_entry(data: dict[str, Any], cwd: str) -> ProjectTrustStoreEntry | None:
    current_dir = normalize_cwd(cwd)
    while True:
        value = data.get(current_dir)
        if value is True or value is False:
            return ProjectTrustStoreEntry(path=current_dir, decision=value)
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            return None
        current_dir = parent_dir


def _read_trust_file(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            parsed = json.loads(strip_bom(f.read()) or "{}")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read trust store {path}: {exc}") from exc
    if not isinstance(parsed, dict) or isinstance(parsed, list):
        raise ValueError(f"Invalid trust store {path}: expected an object")
    data: dict[str, Any] = {}
    for key, value in parsed.items():
        if value not in (True, False, None):
            raise ValueError(
                f"Invalid trust store {path}: value for {json.dumps(key)} must be true, false, or null"
            )
        data[str(key)] = value
    return data


def _write_trust_file(path: str, data: dict[str, Any]) -> None:
    sorted_data = {
        key: data[key]
        for key in sorted(data)
        if data[key] in (True, False, None)
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=2)
        f.write("\n")


def has_trust_requiring_project_resources(cwd: str) -> bool:
    home_dir = canonicalize_path(resolve_path(os.environ.get("HOME") or os.path.expanduser("~")))
    user_agents_skills_dir = os.path.join(home_dir, ".agents", "skills")
    current_dir = canonicalize_path(resolve_path(cwd))

    config_dir = os.path.join(current_dir, CONFIG_DIR_NAME)
    if any(os.path.exists(os.path.join(config_dir, entry)) for entry in TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES):
        return True

    while True:
        agents_skills_dir = os.path.join(current_dir, ".agents", "skills")
        if agents_skills_dir != user_agents_skills_dir and os.path.exists(agents_skills_dir):
            return True
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            return False
        current_dir = parent_dir


class ProjectTrustStore:
    """JSON-backed project trust decisions stored in ``<agentDir>/trust.json``."""

    def __init__(self, agent_dir: str | None = None) -> None:
        resolved = resolve_path(agent_dir or get_agent_dir())
        self.trust_path = os.path.join(resolved, "trust.json")

    def get(self, cwd: str) -> ProjectTrustDecision:
        entry = self.get_entry(cwd)
        return entry.decision if entry else None

    def get_entry(self, cwd: str) -> ProjectTrustStoreEntry | None:
        data = _read_trust_file(self.trust_path)
        return _find_nearest_trust_entry(data, cwd)

    def set(self, cwd: str, decision: ProjectTrustDecision) -> None:
        self.set_many([ProjectTrustUpdate(path=cwd, decision=decision)])

    def set_many(self, decisions: list[ProjectTrustUpdate]) -> None:
        data = _read_trust_file(self.trust_path)
        for update in decisions:
            key = normalize_cwd(update.path)
            if update.decision is None:
                data.pop(key, None)
            else:
                data[key] = update.decision
        _write_trust_file(self.trust_path, data)
