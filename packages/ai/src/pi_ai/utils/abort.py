"""
Abort helpers using asyncio.Event (Python counterpart of AbortSignal).
Mirrors packages/ai/src/utils/abort.ts
"""
from __future__ import annotations

import asyncio
from typing import TypeVar

T = TypeVar("T")


def operation_signal(cancel_event: asyncio.Event | None = None) -> asyncio.Event:
    return cancel_event if cancel_event is not None else asyncio.Event()


def _is_aborted(cancel_event: asyncio.Event | None) -> bool:
    return cancel_event is not None and cancel_event.is_set()


async def race_with_abort_signal(operation, cancel_event: asyncio.Event | None):
    if _is_aborted(cancel_event):
        if asyncio.isfuture(operation) or asyncio.iscoroutine(operation):
            asyncio.create_task(_drain(operation))
        raise asyncio.CancelledError("The operation was aborted")

    if cancel_event is None:
        return await operation

    op_task = asyncio.ensure_future(operation)
    abort_task = asyncio.create_task(cancel_event.wait())
    done, pending = await asyncio.wait({op_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    if abort_task in done and cancel_event.is_set():
        if not op_task.done():
            op_task.cancel()
        raise asyncio.CancelledError("The operation was aborted")
    return await op_task


async def _drain(operation) -> None:
    try:
        await operation
    except Exception:
        pass
