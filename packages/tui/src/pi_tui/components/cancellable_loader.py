"""CancellableLoader component — mirrors components/cancellable-loader.ts"""
from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Callable

from ..keybindings import get_editor_keybindings
from .loader import Loader

if TYPE_CHECKING:
    from ..tui import TUI


class CancellableLoader(Loader):
    """
    Loader that can be cancelled with Escape (selectCancel keybinding).
    Provides an asyncio.Event as cancellation signal.
    Mirrors CancellableLoader in components/cancellable-loader.ts.
    """

    def __init__(
        self,
        ui: "TUI",
        spinner_color_fn: Callable[[str], str],
        message_color_fn: Callable[[str], str],
        message: str = "Loading...",
    ) -> None:
        super().__init__(ui, spinner_color_fn, message_color_fn, message)
        self._abort_event = threading.Event()
        self._asyncio_event: asyncio.Event | None = None
        self.on_abort: Callable[[], None] | None = None

    @property
    def aborted(self) -> bool:
        return self._abort_event.is_set()

    @property
    def signal(self) -> threading.Event:
        """Threading event that is set when the user presses Escape."""
        return self._abort_event

    def get_asyncio_signal(self) -> asyncio.Event:
        """Return an asyncio.Event that is set on abort (for async code)."""
        if self._asyncio_event is None:
            self._asyncio_event = asyncio.Event()
        return self._asyncio_event

    def handle_input(self, data: str) -> None:
        kb = get_editor_keybindings()
        if kb.matches(data, "selectCancel"):
            self._abort_event.set()
            if self._asyncio_event is not None:
                try:
                    loop = asyncio.get_event_loop()
                    loop.call_soon_threadsafe(self._asyncio_event.set)
                except RuntimeError:
                    pass
            if self.on_abort:
                self.on_abort()

    def dispose(self) -> None:
        self.stop()
