"""
Async sleep with cancellation support.

Mirrors utils/sleep.ts (conceptually; TS uses AbortSignal, we use asyncio.Event).
"""

from __future__ import annotations

import asyncio


async def sleep(seconds: float, cancel_event: asyncio.Event | None = None) -> None:
    """Async sleep that respects a cancellation event.

    Args:
        seconds: Number of seconds to sleep.
        cancel_event: Optional asyncio.Event. If set before the sleep completes,
                      raises asyncio.CancelledError.
    """
    if cancel_event is None or not cancel_event.is_set():
        if cancel_event is not None:
            try:
                await asyncio.wait_for(cancel_event.wait(), timeout=seconds)
                raise asyncio.CancelledError("Sleep cancelled")
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.sleep(seconds)
