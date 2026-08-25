"""Session testing types. Mirrors packages/agent/src/harness/session/testing/types.ts"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol


class SessionBackendFixture(Protocol):
    repository: Any

    async def aclose(self) -> None: ...


class SimpleSessionBackendFixture:
    def __init__(self, repository: Any, closer: Callable[[], Awaitable[None] | None] | None = None) -> None:
        self.repository = repository
        self._closer = closer

    async def aclose(self) -> None:
        if self._closer:
            result = self._closer()
            if hasattr(result, "__await__"):
                await result

    async def __aenter__(self) -> SimpleSessionBackendFixture:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


SessionBackendFixtureFactory = Callable[[], Awaitable[SessionBackendFixture]]


class SessionBackendConformanceCase:
    def __init__(self, group: str, name: str, run: Callable[[], Awaitable[None]]) -> None:
        self.group = group
        self.name = name
        self.run = run


class SessionFixture(Protocol):
    def create(self) -> Any: ...
