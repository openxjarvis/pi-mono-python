"""
Serialize file mutations per path — mirrors packages/coding-agent/src/core/tools/file-mutation-queue.ts
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

_file_mutation_queues: dict[str, asyncio.Future[None] | asyncio.Task[None] | asyncio.Event] = {}
_registration_lock = asyncio.Lock()
_queue_tails: dict[str, asyncio.Task[None]] = {}


def _is_missing_path_error(error: BaseException) -> bool:
    if isinstance(error, FileNotFoundError):
        return True
    if isinstance(error, OSError) and error.errno in {2, 20}:  # ENOENT, ENOTDIR
        return True
    return False


async def get_mutation_queue_key(file_path: str) -> str:
    resolved = os.path.abspath(file_path)
    try:
        return os.path.realpath(resolved)
    except OSError as error:
        if _is_missing_path_error(error):
            return resolved
        raise


async def with_file_mutation_queue(file_path: str, fn: Callable[[], Awaitable[T]]) -> T:
    """Serialize file mutation operations targeting the same file."""
    async with _registration_lock:
        key = await get_mutation_queue_key(file_path)
        previous = _queue_tails.get(key)

        async def _run() -> None:
            if previous is not None:
                try:
                    await previous
                except Exception:
                    pass

        gate = asyncio.create_task(_run())
        done = asyncio.Event()

        async def _hold() -> None:
            await gate
            await done.wait()

        holder = asyncio.create_task(_hold())
        _queue_tails[key] = holder

    await gate
    try:
        return await fn()
    finally:
        done.set()
        if _queue_tails.get(key) is holder:
            _queue_tails.pop(key, None)
