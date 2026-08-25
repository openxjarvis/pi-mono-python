"""Theme controller. Mirrors packages/coding-agent/src/modes/interactive/theme/theme-controller.ts"""
from __future__ import annotations

from typing import Any, Callable

from .theme import get_theme, get_theme_by_name


class ThemeController:
    def __init__(self, on_change: Callable[[Any], None] | None = None) -> None:
        self.on_change = on_change
        self.current = get_theme()

    def set_theme(self, name: str) -> Any:
        selected = get_theme_by_name(name)
        self.current = selected
        if self.on_change:
            self.on_change(selected)
        return selected

    def apply_from_settings(self) -> Any:
        return self.current

    def rebind_tui(self) -> None:
        return None


InteractiveThemeController = ThemeController
