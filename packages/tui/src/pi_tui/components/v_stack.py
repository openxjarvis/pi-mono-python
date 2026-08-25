"""VStack — mirrors components/v-stack.ts"""
from __future__ import annotations

from .stack import Stack, allocate_stack_sizes, visible_stack_entries


class VStack(Stack):
    layout_type = "vstack"

    def render(self, width: int) -> list[str]:
        viewport = {"width": max(1, width), "height": 10**9}
        entries = visible_stack_entries(self.entries, viewport)
        rendered = [
            entry.component.render(viewport["width"]) if hasattr(entry.component, "render") else []
            for entry in entries
        ]
        sizes = allocate_stack_sizes(
            entries,
            [len(lines) for lines in rendered],
            None,
            self.gap,
        )
        lines: list[str] = []
        for index, child_lines in enumerate(rendered):
            if index > 0:
                lines.extend([""] * self.gap)
            sliced = child_lines[: sizes[index]]
            lines.extend(sliced)
            lines.extend([""] * max(0, sizes[index] - len(sliced)))
        return lines
