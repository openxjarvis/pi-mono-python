"""HStack — mirrors components/h-stack.ts"""
from __future__ import annotations

from ..tui import composite_tui_line
from ..utils import visible_width
from .stack import Stack, allocate_stack_sizes, visible_stack_entries


class HStack(Stack):
    layout_type = "hstack"

    def render(self, width: int) -> list[str]:
        safe_width = max(1, width)
        viewport = {"width": safe_width, "height": 10**9}
        entries = visible_stack_entries(self.entries, viewport)
        if not entries:
            return []

        intrinsic = []
        for entry in entries:
            lines = entry.component.render(safe_width) if hasattr(entry.component, "render") else []
            intrinsic.append(max((visible_width(line) for line in lines), default=0))
        widths = allocate_stack_sizes(entries, intrinsic, safe_width, self.gap)
        rendered = [
            [] if widths[index] == 0 else (
                entries[index].component.render(widths[index])
                if hasattr(entries[index].component, "render") else []
            )
            for index in range(len(entries))
        ]
        height = max((len(lines) for lines in rendered), default=0)
        result = [""] * height
        x = 0
        for index, lines in enumerate(rendered):
            child_width = widths[index]
            offset = 0
            if self.align == "center":
                offset = (height - len(lines)) // 2
            elif self.align == "end":
                offset = height - len(lines)
            for row, line in enumerate(lines):
                target = row + offset
                if 0 <= target < len(result):
                    result[target] = composite_tui_line(
                        result[target], line, x, child_width, safe_width
                    )
            x += child_width + self.gap
        return result
