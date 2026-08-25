"""
Abort helpers — mirrors packages/coding-agent/src/utils/abort.ts
"""
from __future__ import annotations

import asyncio
from typing import TypeVar

T = TypeVar("T")


class AbortError(Exception):
    def __init__(self, message: str = "The operation was aborted") -> None:
        super().__init__(message)
        self.name = "AbortError"


def operation_signal(signal: asyncio.Event | None = None) -> asyncio.Event:
    return signal if signal is not None else asyncio.Event()


async def race_with_abort_signal(operation: asyncio.Future[T] | asyncio.Task[T], signal: asyncio.Event | None) -> T:
    if signal is None:
        return await operation
    if signal.is_set():
        if not operation.done():
            operation.cancel()
        raise AbortError()

    abort_task = asyncio.create_task(signal.wait())
    try:
        done, pending = await asyncio.wait(
            {asyncio.ensure_future(operation), abort_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if abort_task in done:
            if not operation.done():
                operation.cancel()
            raise AbortError()
        return await operation
    finally:
        if not abort_task.done():
            abort_task.cancel()
