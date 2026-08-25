"""Countdown timer — mirrors countdown-timer.ts"""
from __future__ import annotations

import math
from typing import Any, Callable

from .component import Component


class CountdownTimer(Component):
    name = "countdown_timer"

    def __init__(
        self,
        timeout_ms: int,
        on_tick: Callable[[int], None] | None = None,
        on_expire: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.timeout_ms = timeout_ms
        self.remaining_seconds = max(0, math.ceil(timeout_ms / 1000))
        self.on_tick = on_tick
        self.on_expire = on_expire
        self._disposed = False
        if self.on_tick:
            self.on_tick(self.remaining_seconds)

    def tick(self) -> int:
        if self._disposed:
            return self.remaining_seconds
        self.remaining_seconds = max(0, self.remaining_seconds - 1)
        if self.on_tick:
            self.on_tick(self.remaining_seconds)
        if self.remaining_seconds <= 0:
            self.dispose()
            if self.on_expire:
                self.on_expire()
        return self.remaining_seconds

    def dispose(self) -> None:
        self._disposed = True

    def _render_body(self, width: int) -> str:
        return f"{self.remaining_seconds}s"
