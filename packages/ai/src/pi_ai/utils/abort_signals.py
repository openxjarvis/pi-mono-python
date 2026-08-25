"""
Combine multiple cancel events.
Mirrors packages/ai/src/utils/abort-signals.ts
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class CombinedAbortSignal:
    signal: asyncio.Event | None
    cleanup: callable


def combine_abort_signals(signals: list[asyncio.Event | None]) -> CombinedAbortSignal:
    active = [s for s in signals if s is not None]
    if not active:
        return CombinedAbortSignal(signal=None, cleanup=lambda: None)
    if len(active) == 1:
        return CombinedAbortSignal(signal=active[0], cleanup=lambda: None)

    combined = asyncio.Event()
    tasks: list[asyncio.Task] = []

    async def _watch(event: asyncio.Event) -> None:
        await event.wait()
        combined.set()

    for event in active:
        if event.is_set():
            combined.set()
            break
        tasks.append(asyncio.create_task(_watch(event)))

    def cleanup() -> None:
        for task in tasks:
            task.cancel()

    return CombinedAbortSignal(signal=combined, cleanup=cleanup)
