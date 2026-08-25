"""
Default stream-function registry — mirrors packages/agent/src/stream-fn.ts

Hosts that provide a default model runtime can install its stream function here
without making the agent loop depend on a provider catalog.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import StreamFn

_default_stream_fn: "StreamFn | None" = None


def set_default_stream_fn(stream_fn: "StreamFn | None") -> None:
    """Configure the fallback used when callers omit stream_fn."""
    global _default_stream_fn
    _default_stream_fn = stream_fn


def get_default_stream_fn() -> "StreamFn":
    if _default_stream_fn is None:
        raise RuntimeError(
            "No default stream function configured. Pass stream_fn explicitly or call set_default_stream_fn()."
        )
    return _default_stream_fn
