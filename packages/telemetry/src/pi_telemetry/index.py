from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeVar

from .memory import InMemoryTelemetryContext, RecordedTelemetryEvent, RecordedTelemetrySpan
from .noop import NOOP_TELEMETRY_CONTEXT

T = TypeVar("T")

AttributeValue = str | int | float | bool | list[str] | list[int] | list[float] | list[bool]
SpanAttributes = dict[str, AttributeValue | None]


class SpanOptions(dict):
    pass


SpanStatus = dict[str, Any]
TelemetryAttributeType = Literal["string", "number", "boolean", "string[]", "number[]", "boolean[]"]


class TelemetrySpan:
    async def start_span(
        self,
        options: SpanOptions | dict[str, Any],
        callback: Callable[["TelemetrySpan"], T | Awaitable[T]],
    ) -> T:
        raise NotImplementedError

    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None:
        raise NotImplementedError

    def set_attributes(self, attributes: SpanAttributes) -> None:
        raise NotImplementedError

    def set_status(self, status: SpanStatus) -> None:
        raise NotImplementedError


class TelemetryContext:
    async def start_span(
        self,
        options: SpanOptions | dict[str, Any],
        callback: Callable[[TelemetrySpan], T | Awaitable[T]],
    ) -> T:
        raise NotImplementedError


def define_telemetry_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Typed identity helper for serializable telemetry schema data."""
    return schema


def create_typed_span_starter(telemetry_context: TelemetryContext, _schemas: Any = None):
    """Bind an explicit parent context to the combined span vocabulary of one or more schemas."""

    async def start_span(
        name: str,
        attributes: SpanAttributes,
        callback: Callable[..., T | Awaitable[T]],
    ) -> T:
        async def inner(span: TelemetrySpan) -> T:
            result = callback(span, create_typed_span_starter(span, _schemas))
            if hasattr(result, "__await__"):
                return await result  # type: ignore[misc]
            return result  # type: ignore[return-value]

        return await telemetry_context.start_span({"name": name, "attributes": attributes}, inner)

    return start_span


__all__ = [
    "AttributeValue",
    "InMemoryTelemetryContext",
    "NOOP_TELEMETRY_CONTEXT",
    "RecordedTelemetryEvent",
    "RecordedTelemetrySpan",
    "SpanAttributes",
    "SpanOptions",
    "SpanStatus",
    "TelemetryAttributeType",
    "TelemetryContext",
    "TelemetrySpan",
    "create_typed_span_starter",
    "define_telemetry_schema",
]
