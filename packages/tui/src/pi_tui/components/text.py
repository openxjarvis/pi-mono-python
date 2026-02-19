"""Text component — mirrors components/text.ts"""
from __future__ import annotations

from typing import Callable

from ..utils import apply_background_to_line, visible_width, wrap_text_with_ansi


class Text:
    """
    Displays multi-line text with word wrapping, optional padding and background.
    Mirrors Text in components/text.ts.
    """

    def __init__(
        self,
        text: str = "",
        padding_x: int = 1,
        padding_y: int = 1,
        custom_bg_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._text = text
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._custom_bg_fn = custom_bg_fn
        self._cache_text: str | None = None
        self._cache_width: int | None = None
        self._cache_lines: list[str] | None = None

    def set_text(self, text: str) -> None:
        self._text = text
        self._cache_text = None

    def set_custom_bg_fn(self, fn: Callable[[str], str] | None) -> None:
        self._custom_bg_fn = fn
        self._cache_text = None

    def invalidate(self) -> None:
        self._cache_text = None
        self._cache_width = None
        self._cache_lines = None

    def handle_input(self, _data: str) -> None:
        pass

    def render(self, width: int) -> list[str]:
        if (
            self._cache_lines is not None and
            self._cache_text == self._text and
            self._cache_width == width
        ):
            return self._cache_lines

        if not self._text or not self._text.strip():
            result: list[str] = []
            self._cache_text = self._text
            self._cache_width = width
            self._cache_lines = result
            return result

        normalized = self._text.replace("\t", "   ")
        content_width = max(1, width - self._padding_x * 2)
        wrapped = wrap_text_with_ansi(normalized, content_width)

        left_margin = " " * self._padding_x
        right_margin = " " * self._padding_x
        content_lines: list[str] = []

        for ln in wrapped:
            with_margins = left_margin + ln + right_margin
            if self._custom_bg_fn:
                content_lines.append(apply_background_to_line(with_margins, width, self._custom_bg_fn))
            else:
                vis = visible_width(with_margins)
                content_lines.append(with_margins + " " * max(0, width - vis))

        empty_line = " " * width
        empty_lines = []
        for _ in range(self._padding_y):
            if self._custom_bg_fn:
                empty_lines.append(apply_background_to_line(empty_line, width, self._custom_bg_fn))
            else:
                empty_lines.append(empty_line)

        result = empty_lines + content_lines + empty_lines

        self._cache_text = self._text
        self._cache_width = width
        self._cache_lines = result

        return result if result else [""]
