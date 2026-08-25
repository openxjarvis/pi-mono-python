"""
Settings diagnostics — mirrors packages/coding-agent/src/core/settings-diagnostics.ts
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .settings_manager import SettingsManager


@dataclass
class AgentSessionRuntimeDiagnostic:
    type: Literal["info", "warning", "error"]
    message: str


def collect_settings_diagnostics(settings_manager: SettingsManager) -> list[AgentSessionRuntimeDiagnostic]:
    diagnostics: list[AgentSessionRuntimeDiagnostic] = []
    for item in settings_manager.drain_errors():
        if isinstance(item, dict):
            scope = item.get("scope", "settings")
            path = item.get("path")
            error = item.get("error") or item.get("message") or "unknown error"
            if hasattr(error, "message"):
                error = error.message
            message = (
                f"Invalid settings file {path}: {error}"
                if path
                else f"Invalid {scope} settings: {error}"
            )
        else:
            message = str(item)
        diagnostics.append(AgentSessionRuntimeDiagnostic(type="warning", message=message))
    return diagnostics


def deduplicate_diagnostics(
    diagnostics: list[AgentSessionRuntimeDiagnostic],
) -> list[AgentSessionRuntimeDiagnostic]:
    seen: set[str] = set()
    unique: list[AgentSessionRuntimeDiagnostic] = []
    for diagnostic in diagnostics:
        key = f"{diagnostic.type}\0{diagnostic.message}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(diagnostic)
    return unique
