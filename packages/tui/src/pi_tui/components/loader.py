"""Loader component — mirrors components/loader.ts"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .text import Text

if TYPE_CHECKING:
    from ..tui import TUI


@dataclass
class LoaderIndicatorOptions:
    """Custom loader frames and animation interval."""
    frames: list[str] | None = None
    interval_ms: int | None = None


_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_DEFAULT_INTERVAL_MS = 80


class Loader(Text):
    """
    Spinner component that updates every 80ms with a braille animation.
    Mirrors Loader in components/loader.ts.
    """

    def __init__(
        self,
        ui: "TUI",
        spinner_color_fn: Callable[[str], str],
        message_color_fn: Callable[[str], str],
        message: str = "Loading...",
        indicator: LoaderIndicatorOptions | None = None,
    ) -> None:
        super().__init__("", padding_x=1, padding_y=0)
        self._frames = list(_FRAMES)
        self._interval_ms = _DEFAULT_INTERVAL_MS
        self._current_frame = 0
        self._timer: threading.Timer | None = None
        self._ui = ui
        self._spinner_color_fn = spinner_color_fn
        self._message_color_fn = message_color_fn
        self._message = message
        self._render_indicator_verbatim = False
        self.set_indicator(indicator)

    def render(self, width: int) -> list[str]:
        return [""] + super().render(width)

    def start(self) -> None:
        self.stop()
        self._update_display()
        self._schedule()

    def stop(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def set_message(self, message: str) -> None:
        self._message = message
        self._update_display()

    def set_indicator(self, indicator: LoaderIndicatorOptions | None = None) -> None:
        self._render_indicator_verbatim = indicator is not None
        self._frames = list(indicator.frames) if indicator and indicator.frames is not None else list(_FRAMES)
        interval = indicator.interval_ms if indicator and indicator.interval_ms else None
        self._interval_ms = interval if interval and interval > 0 else _DEFAULT_INTERVAL_MS
        self._current_frame = 0
        self.start()

    def _schedule(self) -> None:
        if len(self._frames) <= 1:
            return
        self._timer = threading.Timer(self._interval_ms / 1000, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self) -> None:
        if not self._frames:
            return
        self._current_frame = (self._current_frame + 1) % len(self._frames)
        self._update_display()
        self._schedule()

    def _update_display(self) -> None:
        frame = self._frames[self._current_frame] if self._frames else ""
        rendered_frame = frame if self._render_indicator_verbatim else self._spinner_color_fn(frame)
        indicator = f"{rendered_frame} " if frame else ""
        self.set_text(f"{indicator}{self._message_color_fn(self._message)}")
        if self._ui:
            self._ui.request_render()

    def invalidate(self) -> None:
        super().invalidate()
