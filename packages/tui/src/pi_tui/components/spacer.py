"""Spacer component — mirrors components/spacer.ts"""
from __future__ import annotations


class Spacer:
    """Renders a fixed number of empty lines. Mirrors Spacer in spacer.ts."""

    def __init__(self, lines: int = 1) -> None:
        self._lines = lines

    def set_lines(self, lines: int) -> None:
        self._lines = lines

    def invalidate(self) -> None:
        pass

    def render(self, _width: int) -> list[str]:
        return [""] * self._lines

    def handle_input(self, _data: str) -> None:
        pass
