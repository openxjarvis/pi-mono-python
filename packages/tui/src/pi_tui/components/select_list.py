"""SelectList component — mirrors components/select-list.ts"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..keybindings import get_editor_keybindings
from ..utils import truncate_to_width


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
    ) -> None:
        self._items = items
        self._filtered_items = list(items)
        self._selected_index = 0
        self._max_visible = max_visible
        self._theme = theme

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
        kb = get_editor_keybindings()
        if kb.matches(key_data, "selectUp"):
            if self._filtered_items:
                self._selected_index = (
                    len(self._filtered_items) - 1
                    if self._selected_index == 0
                    else self._selected_index - 1
                )
                self._notify_selection_change()
        elif kb.matches(key_data, "selectDown"):
            if self._filtered_items:
                self._selected_index = (
                    0 if self._selected_index == len(self._filtered_items) - 1
                    else self._selected_index + 1
                )
                self._notify_selection_change()
        elif kb.matches(key_data, "selectConfirm"):
            if self._filtered_items and self.on_select:
                item = self._filtered_items[self._selected_index]
                self.on_select(item)
        elif kb.matches(key_data, "selectCancel"):
            if self.on_cancel:
                self.on_cancel()

    def _notify_selection_change(self) -> None:
        if self._filtered_items and self.on_selection_change:
            self.on_selection_change(self._filtered_items[self._selected_index])

    def get_selected_item(self) -> SelectItem | None:
        if not self._filtered_items:
            return None
        return self._filtered_items[self._selected_index]

    def render(self, width: int) -> list[str]:
        lines: list[str] = []

        if not self._filtered_items:
            lines.append(self._theme.no_match("  No matching commands"))
            return lines

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
            display_value = item.label or item.value

            if is_selected:
                prefix_width = 2  # "→ "
                if desc_single and width > 40:
                    max_val_w = min(30, width - prefix_width - 4)
                    trunc_val = truncate_to_width(display_value, max_val_w, "")
                    spacing = " " * max(1, 32 - len(trunc_val))
                    desc_start = prefix_width + len(trunc_val) + len(spacing)
                    remain = width - desc_start - 2
                    if remain > 10:
                        trunc_desc = truncate_to_width(desc_single, remain, "")
                        line = self._theme.selected_text(f"→ {trunc_val}{spacing}{trunc_desc}")
                    else:
                        max_w = width - prefix_width - 2
                        line = self._theme.selected_text(f"→ {truncate_to_width(display_value, max_w, '')}")
                else:
                    max_w = width - prefix_width - 2
                    line = self._theme.selected_text(f"→ {truncate_to_width(display_value, max_w, '')}")
            else:
                prefix = "  "
                if desc_single and width > 40:
                    max_val_w = min(30, width - len(prefix) - 4)
                    trunc_val = truncate_to_width(display_value, max_val_w, "")
                    spacing = " " * max(1, 32 - len(trunc_val))
                    desc_start = len(prefix) + len(trunc_val) + len(spacing)
                    remain = width - desc_start - 2
                    if remain > 10:
                        trunc_desc = truncate_to_width(desc_single, remain, "")
                        desc_text = self._theme.description(spacing + trunc_desc)
                        line = prefix + trunc_val + desc_text
                    else:
                        max_w = width - len(prefix) - 2
                        line = prefix + truncate_to_width(display_value, max_w, "")
                else:
                    max_w = width - len(prefix) - 2
                    line = prefix + truncate_to_width(display_value, max_w, "")

            lines.append(line)

        if start_idx > 0 or end_idx < len(self._filtered_items):
            scroll_text = f"  ({self._selected_index + 1}/{len(self._filtered_items)})"
            lines.append(self._theme.scroll_info(truncate_to_width(scroll_text, width - 2, "")))

        return lines
