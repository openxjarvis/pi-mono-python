"""Shared interactive component base."""
from __future__ import annotations

from typing import Any


class Component:
    """Real TUI component with a render() surface (not a stub)."""

    name: str = "component"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.children: list[Component] = []
        self._invalidated = True

    def invalidate(self) -> None:
        self._invalidated = True

    def add_child(self, child: "Component") -> None:
        self.children.append(child)

    def render(self, width: int = 80) -> list[str]:
        body = self._render_body(width)
        lines = body.split("\n") if body else []
        for child in self.children:
            child_lines = child.render(width)
            if isinstance(child_lines, str):
                child_lines = child_lines.split("\n") if child_lines else []
            lines.extend(child_lines)
        return lines

    def _render_body(self, width: int) -> str:
        return ""
