"""Image component — mirrors components/image.ts"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..terminal_image import (
    ImageDimensions,
    ImageRenderOptions,
    get_capabilities,
    get_image_dimensions,
    image_fallback,
    render_image,
)


@dataclass
class ImageTheme:
    fallback_color: Callable[[str], str]


@dataclass
class ImageOptions:
    max_width_cells: int | None = None
    max_height_cells: int | None = None
    filename: str | None = None
    image_id: int | None = None


class Image:
    """
    Renders an inline image using Kitty/iTerm2 protocols.
    Falls back to text if the terminal doesn't support images.
    Mirrors Image in components/image.ts.
    """

    def __init__(
        self,
        base64_data: str,
        mime_type: str,
        theme: ImageTheme,
        options: ImageOptions | None = None,
        dimensions: ImageDimensions | None = None,
    ) -> None:
        self._base64_data = base64_data
        self._mime_type = mime_type
        self._theme = theme
        self._options = options or ImageOptions()
        self._dimensions = (
            dimensions or
            get_image_dimensions(base64_data, mime_type) or
            ImageDimensions(800, 600)
        )
        self._image_id = self._options.image_id
        self._cached_lines: list[str] | None = None
        self._cached_width: int | None = None

    def get_image_id(self) -> int | None:
        return self._image_id

    def invalidate(self) -> None:
        self._cached_lines = None
        self._cached_width = None

    def handle_input(self, _data: str) -> None:
        pass

    def render(self, width: int) -> list[str]:
        if self._cached_lines is not None and self._cached_width == width:
            return self._cached_lines

        max_width = min(width - 2, self._options.max_width_cells or 60)
        caps = get_capabilities()
        lines: list[str]

        if caps.images:
            render_opts = ImageRenderOptions(
                max_width_cells=max_width,
                image_id=self._image_id,
            )
            result = render_image(self._base64_data, self._dimensions, render_opts)
            if result:
                if result.image_id:
                    self._image_id = result.image_id
                lines = [""] * (result.rows - 1)
                move_up = f"\x1b[{result.rows - 1}A" if result.rows > 1 else ""
                lines.append(move_up + result.sequence)
            else:
                fallback = image_fallback(self._mime_type, self._dimensions, self._options.filename)
                lines = [self._theme.fallback_color(fallback)]
        else:
            fallback = image_fallback(self._mime_type, self._dimensions, self._options.filename)
            lines = [self._theme.fallback_color(fallback)]

        self._cached_lines = lines
        self._cached_width = width
        return lines
