"""SettingsList component — mirrors components/settings-list.ts"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

from ..fuzzy import fuzzy_filter
from ..keybindings import get_editor_keybindings
from ..utils import truncate_to_width, visible_width, wrap_text_with_ansi
from .input import Input

if TYPE_CHECKING:
    from ..tui import Component


@dataclass
class SettingItem:
    id: str
    label: str
    current_value: str
    description: str | None = None
    values: list[str] | None = None
    submenu: "Callable[[str, Callable[[str | None], None]], Component] | None" = None


@dataclass
class SettingsListTheme:
    label: Callable[[str, bool], str] = field(default=lambda text, selected: text)
    value: Callable[[str, bool], str] = field(default=lambda text, selected: text)
    description: Callable[[str], str] = field(default=lambda x: x)
    cursor: str = "→ "
    hint: Callable[[str], str] = field(default=lambda x: x)


@dataclass
class SettingsListOptions:
    enable_search: bool = False


class SettingsList:
    """
    A scrollable settings list with optional fuzzy search, value cycling, and submenus.
    Mirrors SettingsList in components/settings-list.ts.
    """

    def __init__(
        self,
        items: list[SettingItem],
        max_visible: int,
        theme: SettingsListTheme,
        on_change: Callable[[str, str], None],
        on_cancel: Callable[[], None],
        options: SettingsListOptions | None = None,
    ) -> None:
        self._items = items
        self._filtered_items = list(items)
        self._theme = theme
        self._selected_index = 0
        self._max_visible = max_visible
        self._on_change = on_change
        self._on_cancel = on_cancel
        opts = options or SettingsListOptions()
        self._search_enabled = opts.enable_search
        self._search_input: Input | None = Input() if self._search_enabled else None

        self._submenu: "Component | None" = None
        self._submenu_item_index: int | None = None

    def update_value(self, id: str, new_value: str) -> None:
        for item in self._items:
            if item.id == id:
                item.current_value = new_value
                break

    def invalidate(self) -> None:
        if self._submenu and hasattr(self._submenu, "invalidate"):
            self._submenu.invalidate()  # type: ignore[union-attr]

    def render(self, width: int) -> list[str]:
        if self._submenu:
            if hasattr(self._submenu, "render"):
                return self._submenu.render(width)  # type: ignore[union-attr]
        return self._render_main_list(width)

    def _render_main_list(self, width: int) -> list[str]:
        lines: list[str] = []

        if self._search_enabled and self._search_input:
            lines.extend(self._search_input.render(width))
            lines.append("")

        if not self._items:
            lines.append(self._theme.hint("  No settings available"))
            if self._search_enabled:
                self._add_hint_line(lines, width)
            return lines

        display_items = self._filtered_items if self._search_enabled else self._items
        if not display_items:
            lines.append(truncate_to_width(self._theme.hint("  No matching settings"), width))
            self._add_hint_line(lines, width)
            return lines

        start_idx = max(
            0,
            min(
                self._selected_index - self._max_visible // 2,
                len(display_items) - self._max_visible,
            ),
        )
        end_idx = min(start_idx + self._max_visible, len(display_items))

        max_label_width = min(30, max(visible_width(item.label) for item in self._items))

        for i in range(start_idx, end_idx):
            item = display_items[i]
            is_selected = i == self._selected_index
            prefix = self._theme.cursor if is_selected else "  "
            prefix_width = visible_width(prefix)

            label_padded = item.label + " " * max(0, max_label_width - visible_width(item.label))
            label_text = self._theme.label(label_padded, is_selected)

            separator = "  "
            used_width = prefix_width + max_label_width + visible_width(separator)
            value_max_width = width - used_width - 2

            value_text = self._theme.value(truncate_to_width(item.current_value, max(1, value_max_width), ""), is_selected)
            lines.append(truncate_to_width(prefix + label_text + separator + value_text, width))

        if start_idx > 0 or end_idx < len(display_items):
            scroll_text = f"  ({self._selected_index + 1}/{len(display_items)})"
            lines.append(self._theme.hint(truncate_to_width(scroll_text, width - 2, "")))

        selected_item = display_items[self._selected_index] if display_items else None
        if selected_item and selected_item.description:
            lines.append("")
            wrapped = wrap_text_with_ansi(selected_item.description, width - 4)
            for ln in wrapped:
                lines.append(self._theme.description(f"  {ln}"))

        self._add_hint_line(lines, width)
        return lines

    def handle_input(self, data: str) -> None:
        if self._submenu and hasattr(self._submenu, "handle_input"):
            self._submenu.handle_input(data)  # type: ignore[union-attr]
            return

        kb = get_editor_keybindings()
        display_items = self._filtered_items if self._search_enabled else self._items

        if kb.matches(data, "selectUp"):
            if display_items:
                self._selected_index = (
                    len(display_items) - 1 if self._selected_index == 0
                    else self._selected_index - 1
                )
        elif kb.matches(data, "selectDown"):
            if display_items:
                self._selected_index = (
                    0 if self._selected_index == len(display_items) - 1
                    else self._selected_index + 1
                )
        elif kb.matches(data, "selectConfirm") or data == " ":
            self._activate_item()
        elif kb.matches(data, "selectCancel"):
            self._on_cancel()
        elif self._search_enabled and self._search_input:
            sanitized = data.replace(" ", "")
            if not sanitized:
                return
            self._search_input.handle_input(sanitized)
            self._apply_filter(self._search_input.get_value())

    def _activate_item(self) -> None:
        display_items = self._filtered_items if self._search_enabled else self._items
        if not display_items:
            return
        item = display_items[self._selected_index]
        if not item:
            return

        if item.submenu:
            self._submenu_item_index = self._selected_index
            def done(selected_value: str | None = None) -> None:
                if selected_value is not None:
                    item.current_value = selected_value
                    self._on_change(item.id, selected_value)
                self._close_submenu()
            self._submenu = item.submenu(item.current_value, done)
        elif item.values:
            current_idx = item.values.index(item.current_value) if item.current_value in item.values else -1
            next_idx = (current_idx + 1) % len(item.values)
            new_value = item.values[next_idx]
            item.current_value = new_value
            self._on_change(item.id, new_value)

    def _close_submenu(self) -> None:
        self._submenu = None
        if self._submenu_item_index is not None:
            self._selected_index = self._submenu_item_index
            self._submenu_item_index = None

    def _apply_filter(self, query: str) -> None:
        self._filtered_items = fuzzy_filter(self._items, query, get_text=lambda item: item.label)
        self._selected_index = 0

    def _add_hint_line(self, lines: list[str], width: int) -> None:
        lines.append("")
        hint_text = (
            "  Type to search · Enter/Space to change · Esc to cancel"
            if self._search_enabled
            else "  Enter/Space to change · Esc to cancel"
        )
        lines.append(truncate_to_width(self._theme.hint(hint_text), width))
