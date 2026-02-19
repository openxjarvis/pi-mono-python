"""TruncatedText component — mirrors components/truncated-text.ts"""
from __future__ import annotations

from ..utils import truncate_to_width, visible_width


class TruncatedText:
    """
    Displays a single line of text, truncating to fit viewport width.
    Mirrors TruncatedText in components/truncated-text.ts.
    """

    def __init__(self, text: str = "", padding_x: int = 0, padding_y: int = 0) -> None:
        self._text = text
        self._padding_x = padding_x
        self._padding_y = padding_y

    def set_text(self, text: str) -> None:
        self._text = text

    def invalidate(self) -> None:
        pass

    def handle_input(self, _data: str) -> None:
        pass

    def render(self, width: int) -> list[str]:
        result: list[str] = []
        empty_line = " " * width

        for _ in range(self._padding_y):
            result.append(empty_line)

        avail = max(1, width - self._padding_x * 2)
        # Take only the first line
        single = self._text
        nl = self._text.find("\n")
        if nl != -1:
            single = self._text[:nl]

        display = truncate_to_width(single, avail)
        left_pad = " " * self._padding_x
        right_pad = " " * self._padding_x
        line_with_pad = left_pad + display + right_pad
        vis = visible_width(line_with_pad)
        final = line_with_pad + " " * max(0, width - vis)
        result.append(final)

        for _ in range(self._padding_y):
            result.append(empty_line)

        return result
