"""Transient alternate-screen flash stack — mirrors components/alt-screen-flash.ts"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from ..utils import truncate_to_width

_DEFAULT_DURATION_MS = 1000


@dataclass
class _FlashEntry:
    id: int
    message: str
    timer: threading.Timer


class AltScreenFlashContainer:
    """Transient messages composited by the alternate-screen renderer."""

    def __init__(self, request_render: Callable[[], None]) -> None:
        self._entries: list[_FlashEntry] = []
        self._next_id = 0
        self._request_render = request_render

    def flash(self, message: str, duration_ms: int = _DEFAULT_DURATION_MS) -> None:
        entry_id = self._next_id
        self._next_id += 1

        def _expire() -> None:
            index = next((i for i, entry in enumerate(self._entries) if entry.id == entry_id), -1)
            if index == -1:
                return
            self._entries.pop(index)
            self._request_render()

        timer = threading.Timer(max(0, duration_ms) / 1000.0, _expire)
        timer.daemon = True
        self._entries.append(_FlashEntry(id=entry_id, message=message, timer=timer))
        timer.start()
        self._request_render()

    def dispose(self) -> None:
        for entry in self._entries:
            entry.timer.cancel()
        self._entries.clear()

    def invalidate(self) -> None:
        return

    def handle_input(self, _data: str) -> None:
        return

    def render(self, width: int) -> list[str]:
        return [
            f"\x1b[7m{truncate_to_width(f' {entry.message} ', width, '')}\x1b[27m"
            for entry in self._entries
        ]
