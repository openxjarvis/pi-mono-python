"""
Split current tools into prefix and transcript-loaded definitions.
Mirrors packages/ai/src/utils/deferred-tools.ts
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pi_ai.types import Context, Tool

ToolNameNormalizer = Callable[[str], str]


def _identity(name: str) -> str:
    return name


def split_deferred_tools(
    context: Context,
    enabled: bool,
    normalize_name: ToolNameNormalizer = _identity,
) -> tuple[list[Tool], dict[str, Tool]]:
    """Split current tools into immediate prefix tools and deferred transcript-loaded ones."""
    unique_tools: dict[str, Tool] = {}
    for tool in context.tools or []:
        unique_tools[normalize_name(tool.name)] = tool
    if not enabled:
        return list(unique_tools.values()), {}

    deferred_names: set[str] = set()
    used_names: set[str] = set()
    for message in context.messages:
        role = getattr(message, "role", None)
        if role == "assistant":
            for block in getattr(message, "content", []) or []:
                if getattr(block, "type", None) == "toolCall":
                    used_names.add(normalize_name(getattr(block, "name", "")))
        elif role == "toolResult":
            for name in getattr(message, "added_tool_names", None) or []:
                normalized = normalize_name(name)
                if normalized not in used_names:
                    deferred_names.add(normalized)

    immediate: list[Tool] = []
    deferred: dict[str, Tool] = {}
    for name, tool in unique_tools.items():
        if name in deferred_names:
            deferred[name] = tool
        else:
            immediate.append(tool)
    return immediate, deferred
