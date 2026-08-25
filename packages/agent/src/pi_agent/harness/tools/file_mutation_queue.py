"""Serialize file mutations. Mirrors packages/agent/src/harness/tools/file-mutation-queue.ts"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class FileMutationQueue:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def run(self, fn: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            return await fn()
