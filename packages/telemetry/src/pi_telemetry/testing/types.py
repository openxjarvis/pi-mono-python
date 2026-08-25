from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from ..index import TelemetryContext
from ..memory import RecordedTelemetrySpan


class TelemetryAdapterFixture(Protocol):
    """A fresh adapter instance and normalized snapshot reader owned by one conformance case."""

    context: TelemetryContext

    async def get_spans(self) -> list[RecordedTelemetrySpan]: ...

    async def __aenter__(self) -> TelemetryAdapterFixture: ...

    async def __aexit__(self, *exc: object) -> None: ...


TelemetryAdapterFixtureFactory = Callable[[], Awaitable[TelemetryAdapterFixture]]


class TelemetryAdapterConformanceCase:
    def __init__(self, group: str, name: str, run: Callable[[], Awaitable[None]]) -> None:
        self.group = group
        self.name = name
        self.run = run
