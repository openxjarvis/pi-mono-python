"""Thinking selector — mirrors thinking-selector.ts"""
from __future__ import annotations

from typing import Any, Callable

from .component import Component

DEFAULT_LEVELS = ["off", "minimal", "low", "medium", "high"]


class ThinkingSelectorComponent(Component):
    name = "thinking_selector"

    def __init__(
        self,
        levels: list[str] | None = None,
        current: str = "off",
        on_select: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.levels = list(levels or DEFAULT_LEVELS)
        self.current = current
        self.on_select = on_select
        self.on_cancel = on_cancel
        self.selected_index = self.levels.index(current) if current in self.levels else 0

    def select_current(self) -> str:
        level = self.levels[self.selected_index]
        if self.on_select:
            self.on_select(level)
        return level

    def _render_body(self, width: int) -> str:
        lines = ["Thinking level"]
        for index, level in enumerate(self.levels):
            marker = ">" if index == self.selected_index else " "
            lines.append(f"  {marker} {level}")
        return "\n".join(lines)
