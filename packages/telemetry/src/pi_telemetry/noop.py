from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from .index import SpanOptions, TelemetryContext, TelemetrySpan

T = TypeVar("T")


async def _start_noop_span(
    _options: SpanOptions,
    callback: Callable[[TelemetrySpan], T | Awaitable[T]],
) -> T:
    result = callback(noop_telemetry_span)
    if hasattr(result, "__await__"):
        return await result  # type: ignore[misc]
    return result  # type: ignore[return-value]


class _NoopTelemetrySpan:
    async def start_span(
        self,
        options: SpanOptions,
        callback: Callable[[TelemetrySpan], T | Awaitable[T]],
    ) -> T:
        return await _start_noop_span(options, callback)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        return None

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        return None

    def set_status(self, status: dict[str, Any]) -> None:
        return None


noop_telemetry_span: TelemetrySpan = _NoopTelemetrySpan()  # type: ignore[assignment]

# Shared telemetry context used when an application does not provide one.
NOOP_TELEMETRY_CONTEXT: TelemetryContext = noop_telemetry_span  # type: ignore[assignment]
