"""
Abortable sleep.
Mirrors packages/ai/src/utils/sleep.ts
"""
from __future__ import annotations

import asyncio


async def sleep(ms: float, cancel_event: asyncio.Event | None = None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise asyncio.CancelledError("The operation was aborted")
    try:
        await asyncio.sleep(ms / 1000.0)
    finally:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError("The operation was aborted")
