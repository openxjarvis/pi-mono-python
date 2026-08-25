"""Interactive component. Mirrors TypeScript skill-invocation-message."""
from __future__ import annotations

from typing import Any, Callable

from pi_tui.components.text import Text
from pi_tui.components.select_list import SelectItem, SelectList


class SkillInvocationMessageComponent(Text):
    def __init__(self, text: str = "", items: list[dict[str, Any]] | None = None, on_select: Callable | None = None, **kwargs: Any) -> None:
        super().__init__(text or "SkillInvocationMessage")
        self.items = items or []
        self.on_select = on_select
        self.kwargs = kwargs

    def set_items(self, items: list[dict[str, Any]]) -> None:
        self.items = items
        self.invalidate()

    def select(self, index: int) -> Any:
        if self.on_select and 0 <= index < len(self.items):
            return self.on_select(self.items[index])
        return None
