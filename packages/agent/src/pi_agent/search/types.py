"""Session search interfaces — mirrors packages/agent/src/search/index.ts."""
from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any, Protocol, TypedDict


class SessionSearchOptions(TypedDict, total=False):
    entry_types: list[str]
    limit: int
    abort: Any
    signal: Any


class SessionSearchHit(TypedDict):
    session_id: str
    entry_id: str


class SessionSearch(Protocol):
    def search(self, text: str, options: SessionSearchOptions | None = None) -> AsyncIterable[Any]: ...
