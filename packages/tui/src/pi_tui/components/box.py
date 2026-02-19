"""Box component — mirrors components/box.ts"""
from __future__ import annotations

from typing import Callable

from ..utils import apply_background_to_line, visible_width


class Box:
    """
    Container that applies padding and optional background to all children.
    Mirrors Box in components/box.ts.
    """

    def __init__(
        self,
        padding_x: int = 1,
        padding_y: int = 1,
        bg_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.children: list[object] = []
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._bg_fn = bg_fn
        self._cache: dict | None = None

    def add_child(self, component: object) -> None:
        self.children.append(component)
        self._cache = None

    def remove_child(self, component: object) -> None:
        try:
            self.children.remove(component)
        except ValueError:
            pass
        self._cache = None

    def clear(self) -> None:
        self.children = []
        self._cache = None

    def set_bg_fn(self, fn: Callable[[str], str] | None) -> None:
        self._bg_fn = fn

    def invalidate(self) -> None:
        self._cache = None
        for child in self.children:
            if hasattr(child, "invalidate"):
                child.invalidate()  # type: ignore

    def handle_input(self, _data: str) -> None:
        pass

    def _apply_bg(self, line: str, width: int) -> str:
        vis = visible_width(line)
        pad_needed = max(0, width - vis)
        padded = line + " " * pad_needed
        if self._bg_fn:
            return apply_background_to_line(padded, width, self._bg_fn)
        return padded

    def render(self, width: int) -> list[str]:
        if not self.children:
            return []

        content_width = max(1, width - self._padding_x * 2)
        left_pad = " " * self._padding_x

        child_lines: list[str] = []
        for child in self.children:
            if hasattr(child, "render"):
                for ln in child.render(content_width):  # type: ignore
                    child_lines.append(left_pad + ln)

        if not child_lines:
            return []

        bg_sample = self._bg_fn("test") if self._bg_fn else None

        # Check cache
        c = self._cache
        if (
            c is not None and
            c["width"] == width and
            c["bg_sample"] == bg_sample and
            len(c["child_lines"]) == len(child_lines) and
            all(c["child_lines"][i] == child_lines[i] for i in range(len(child_lines)))
        ):
            return c["lines"]

        result: list[str] = []
        for _ in range(self._padding_y):
            result.append(self._apply_bg("", width))
        for ln in child_lines:
            result.append(self._apply_bg(ln, width))
        for _ in range(self._padding_y):
            result.append(self._apply_bg("", width))

        self._cache = {
            "child_lines": child_lines,
            "width": width,
            "bg_sample": bg_sample,
            "lines": result,
        }
        return result
