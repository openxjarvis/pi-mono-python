"""
Resource diagnostic types for reporting extension/skill/prompt loading issues.

Mirrors core/diagnostics.ts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ResourceCollision:
    """Describes a naming collision between two resources."""

    resource_type: Literal["extension", "skill", "prompt", "theme"]
    name: str
    winner_path: str
    loser_path: str
    winner_source: str | None = None
    loser_source: str | None = None


@dataclass
class ResourceDiagnostic:
    """A diagnostic message about a resource loading issue."""

    type: Literal["warning", "error", "collision"]
    message: str
    path: str | None = None
    collision: ResourceCollision | None = None
