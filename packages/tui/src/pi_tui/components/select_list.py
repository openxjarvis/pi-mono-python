"""SelectList component — mirrors components/select-list.ts"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..keybindings import get_keybindings
from ..utils import truncate_to_width, visible_width


def _normalize_to_single_line(text: str) -> str:
    import re
    return re.sub(r"[\r\n]+", " ", text).strip()


@dataclass
class SelectItem:
    value: str
    label: str
    description: str | None = None


@dataclass
class SelectListTheme:
    selected_prefix: Callable[[str], str] = field(default=lambda x: x)
    selected_text: Callable[[str], str] = field(default=lambda x: x)
    description: Callable[[str], str] = field(default=lambda x: x)
    scroll_info: Callable[[str], str] = field(default=lambda x: x)
    no_match: Callable[[str], str] = field(default=lambda x: x)


@dataclass
class SelectListTruncatePrimaryContext:
    text: str
    max_width: int
    column_width: int
    item: SelectItem
    is_selected: bool


@dataclass
class SelectListLayoutOptions:
    min_primary_column_width: int | None = None
    max_primary_column_width: int | None = None
    truncate_primary: Callable[[SelectListTruncatePrimaryContext], str] | None = None


_DEFAULT_PRIMARY_COLUMN_WIDTH = 32
_PRIMARY_COLUMN_GAP = 2
_MIN_DESCRIPTION_WIDTH = 10


class SelectList:
    """
    Interactive list component with keyboard navigation.
    Mirrors SelectList in components/select-list.ts.
    """

    def __init__(
        self,
        items: list[SelectItem],
        max_visible: int,
        theme: SelectListTheme,
        layout: SelectListLayoutOptions | None = None,
    ) -> None:
        self._items = items
        self._filtered_items = list(items)
        self._selected_index = 0
        self._max_visible = max_visible
        self._theme = theme
        self._layout = layout or SelectListLayoutOptions()

        self.on_select: Callable[[SelectItem], None] | None = None
        self.on_cancel: Callable[[], None] | None = None
        self.on_selection_change: Callable[[SelectItem], None] | None = None

    def set_filter(self, filter_str: str) -> None:
        fl = filter_str.lower()
        self._filtered_items = [
            item for item in self._items
            if item.value.lower().startswith(fl)
        ]
        self._selected_index = 0

    def set_selected_index(self, index: int) -> None:
        self._selected_index = max(0, min(index, len(self._filtered_items) - 1))

    def invalidate(self) -> None:
        pass

    def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()
        if kb.matches(key_data, "tui.select.up"):
            if self._filtered_items:
                self._selected_index = (
                    len(self._filtered_items) - 1
                    if self._selected_index == 0
                    else self._selected_index - 1
                )
                self._notify_selection_change()
        elif kb.matches(key_data, "tui.select.down"):
            if self._filtered_items:
                self._selected_index = (
                    0 if self._selected_index == len(self._filtered_items) - 1
                    else self._selected_index + 1
                )
                self._notify_selection_change()
        elif kb.matches(key_data, "tui.select.pageUp"):
            if self._filtered_items:
                self._selected_index = max(0, self._selected_index - self._max_visible)
                self._notify_selection_change()
        elif kb.matches(key_data, "tui.select.pageDown"):
            if self._filtered_items:
                self._selected_index = min(
                    len(self._filtered_items) - 1,
                    self._selected_index + self._max_visible,
                )
                self._notify_selection_change()
        elif kb.matches(key_data, "tui.select.confirm"):
            if self._filtered_items and self.on_select:
                item = self._filtered_items[self._selected_index]
                self.on_select(item)
        elif kb.matches(key_data, "tui.select.cancel"):
            if self.on_cancel:
                self.on_cancel()

    def _notify_selection_change(self) -> None:
        if self._filtered_items and self.on_selection_change:
            self.on_selection_change(self._filtered_items[self._selected_index])

    def get_selected_item(self) -> SelectItem | None:
        if not self._filtered_items:
            return None
        return self._filtered_items[self._selected_index]

    def _render_item(
        self,
        item: SelectItem,
        is_selected: bool,
        width: int,
        description_single_line: str | None,
        primary_column_width: int,
    ) -> str:
        prefix = "→ " if is_selected else "  "
        prefix_width = visible_width(prefix)

        if description_single_line and width > 40:
            effective = max(1, min(primary_column_width, width - prefix_width - 4))
            max_primary = max(1, effective - _PRIMARY_COLUMN_GAP)
            truncated_value = self._truncate_primary(item, is_selected, max_primary, effective)
            truncated_w = visible_width(truncated_value)
            spacing = " " * max(1, effective - truncated_w)
            description_start = prefix_width + truncated_w + len(spacing)
            remaining = width - description_start - 2
            if remaining > _MIN_DESCRIPTION_WIDTH:
                truncated_desc = truncate_to_width(description_single_line, remaining, "")
                if is_selected:
                    return self._theme.selected_text(f"{prefix}{truncated_value}{spacing}{truncated_desc}")
                return prefix + truncated_value + self._theme.description(spacing + truncated_desc)

        max_width = width - prefix_width - 2
        truncated_value = self._truncate_primary(item, is_selected, max_width, max_width)
        if is_selected:
            return self._theme.selected_text(f"{prefix}{truncated_value}")
        return prefix + truncated_value

    def _get_primary_column_width(self) -> int:
        bounds_min, bounds_max = self._get_primary_column_bounds()
        widest = 0
        for item in self._filtered_items:
            widest = max(widest, visible_width(item.label or item.value) + _PRIMARY_COLUMN_GAP)
        return max(bounds_min, min(bounds_max, widest))

    def _get_primary_column_bounds(self) -> tuple[int, int]:
        raw_min = (
            self._layout.min_primary_column_width
            if self._layout.min_primary_column_width is not None
            else self._layout.max_primary_column_width
            if self._layout.max_primary_column_width is not None
            else _DEFAULT_PRIMARY_COLUMN_WIDTH
        )
        raw_max = (
            self._layout.max_primary_column_width
            if self._layout.max_primary_column_width is not None
            else self._layout.min_primary_column_width
            if self._layout.min_primary_column_width is not None
            else _DEFAULT_PRIMARY_COLUMN_WIDTH
        )
        return max(1, min(raw_min, raw_max)), max(1, max(raw_min, raw_max))

    def _truncate_primary(self, item: SelectItem, is_selected: bool, max_width: int, column_width: int) -> str:
        display_value = item.label or item.value
        if self._layout.truncate_primary:
            truncated = self._layout.truncate_primary(
                SelectListTruncatePrimaryContext(
                    text=display_value,
                    max_width=max_width,
                    column_width=column_width,
                    item=item,
                    is_selected=is_selected,
                )
            )
        else:
            truncated = truncate_to_width(display_value, max_width, "")
        return truncate_to_width(truncated, max_width, "")

    def render(self, width: int) -> list[str]:
        lines: list[str] = []

        if not self._filtered_items:
            lines.append(self._theme.no_match("  No matching commands"))
            return lines

        primary_column_width = self._get_primary_column_width()
        start_idx = max(
            0,
            min(
                self._selected_index - self._max_visible // 2,
                len(self._filtered_items) - self._max_visible,
            ),
        )
        end_idx = min(start_idx + self._max_visible, len(self._filtered_items))

        for i in range(start_idx, end_idx):
            item = self._filtered_items[i]
            is_selected = i == self._selected_index
            desc_single = _normalize_to_single_line(item.description) if item.description else None
            lines.append(self._render_item(item, is_selected, width, desc_single, primary_column_width))

        if start_idx > 0 or end_idx < len(self._filtered_items):
            scroll_text = f"  ({self._selected_index + 1}/{len(self._filtered_items)})"
            lines.append(self._theme.scroll_info(truncate_to_width(scroll_text, width - 2, "")))

        return lines
