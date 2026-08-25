"""
Session resource cleanup registry.
Mirrors packages/ai/src/session-resources.ts
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

Cleanup = Callable[[], Awaitable[None] | None]
_CLEANUPS: list[Cleanup] = []


def register_session_resource_cleanup(fn: Cleanup) -> None:
    _CLEANUPS.append(fn)


async def cleanup_session_resources() -> None:
    for fn in list(_CLEANUPS):
        result = fn()
        if hasattr(result, "__await__"):
            await result  # type: ignore[misc]
