"""Loader component — mirrors components/loader.ts"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable

from .text import Text

if TYPE_CHECKING:
    from ..tui import TUI

_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


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
    ) -> None:
        super().__init__("", padding_x=1, padding_y=0)
        self._frames = list(_FRAMES)
        self._current_frame = 0
        self._timer: threading.Timer | None = None
        self._ui = ui
        self._spinner_color_fn = spinner_color_fn
        self._message_color_fn = message_color_fn
        self._message = message
        self.start()

    def render(self, width: int) -> list[str]:
        return [""] + super().render(width)

    def start(self) -> None:
        self._update_display()
        self._schedule()

    def stop(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def set_message(self, message: str) -> None:
        self._message = message
        self._update_display()

    def _schedule(self) -> None:
        self._timer = threading.Timer(0.08, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self) -> None:
        self._current_frame = (self._current_frame + 1) % len(self._frames)
        self._update_display()
        self._schedule()

    def _update_display(self) -> None:
        frame = self._frames[self._current_frame]
        self.set_text(f"{self._spinner_color_fn(frame)} {self._message_color_fn(self._message)}")
        if self._ui:
            self._ui.request_render()

    def invalidate(self) -> None:
        super().invalidate()
