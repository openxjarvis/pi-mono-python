from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from .noop import NOOP_TELEMETRY_CONTEXT

AttributeValue = str | int | float | bool | list[str] | list[int] | list[float] | list[bool]
SpanAttributes = dict[str, AttributeValue | None]
SpanOptions = dict[str, Any]
SpanStatus = dict[str, Any]


class TelemetryContext:
    async def start_span(self, options: SpanOptions, callback: Callable[..., Any]) -> Any:
        raise NotImplementedError


class TelemetrySpan(TelemetryContext):
    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None:
        raise NotImplementedError

    def set_attributes(self, attributes: SpanAttributes) -> None:
        raise NotImplementedError

    def set_status(self, status: SpanStatus) -> None:
        raise NotImplementedError

T = TypeVar("T")


class RecordedTelemetryEvent(dict):
    def __init__(self, name: str, attributes: SpanAttributes) -> None:
        super().__init__(name=name, attributes=attributes)
        self.name = name
        self.attributes = attributes


class RecordedTelemetrySpan(dict):
    def __init__(
        self,
        *,
        id: int,
        parent_id: int | None,
        name: str,
        attributes: SpanAttributes,
        events: list[RecordedTelemetryEvent],
        status: SpanStatus,
        settled: bool,
        end_sequence: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "id": id,
            "parentId": parent_id,
            "parent_id": parent_id,
            "name": name,
            "attributes": attributes,
            "events": events,
            "status": status,
            "settled": settled,
        }
        if end_sequence is not None:
            payload["endSequence"] = end_sequence
            payload["end_sequence"] = end_sequence
        super().__init__(payload)
        self.id = id
        self.parent_id = parent_id
        self.name = name
        self.attributes = attributes
        self.events = events
        self.status = status
        self.settled = settled
        self.end_sequence = end_sequence


class _MutableEvent:
    def __init__(self, name: str, attributes: SpanAttributes) -> None:
        self.name = name
        self.attributes = attributes


class _MutableSpan:
    def __init__(
        self,
        *,
        id: int,
        parent_id: int | None,
        name: str,
        attributes: SpanAttributes,
    ) -> None:
        self.id = id
        self.parent_id = parent_id
        self.name = name
        self.attributes = attributes
        self.events: list[_MutableEvent] = []
        self.status: SpanStatus = {"status": "ok"}
        self.explicit_status = False
        self.settled = False
        self.end_sequence: int | None = None


class _InMemoryState:
    def __init__(self) -> None:
        self.spans: list[_MutableSpan] = []
        self.next_span_id = 1
        self.next_end_sequence = 1


def _copy_attribute_value(value: AttributeValue) -> AttributeValue:
    return list(value) if isinstance(value, list) else value


def _copy_attributes(attributes: SpanAttributes | None = None) -> SpanAttributes:
    copy: SpanAttributes = {}
    if not attributes:
        return copy
    for name, value in attributes.items():
        if value is not None:
            copy[name] = _copy_attribute_value(value)
    return copy


def _merge_attributes(current: SpanAttributes, attributes: SpanAttributes) -> SpanAttributes:
    merged = _copy_attributes(current)
    for name, value in attributes.items():
        if value is not None:
            merged[name] = _copy_attribute_value(value)
    return merged


def _copy_status(status: SpanStatus) -> SpanStatus:
    if status.get("status") == "ok":
        return {"status": "ok"}
    error = status.get("error")
    if isinstance(error, dict):
        return {"status": "error", "error": {"name": error.get("name"), "message": error.get("message")}}
    return {"status": "error"}


def _automatic_error_status(error: object) -> SpanStatus:
    try:
        if isinstance(error, Exception):
            return {"status": "error", "error": {"name": type(error).__name__, "message": str(error)}}
    except Exception:
        pass
    return {"status": "error"}


def _settle_span(state: _InMemoryState, span: _MutableSpan, failed: bool, error: object | None = None) -> None:
    if span.settled:
        return
    if failed and not span.explicit_status:
        span.status = _automatic_error_status(error)
    span.settled = True
    span.end_sequence = state.next_end_sequence
    state.next_end_sequence += 1


def _create_span(state: _InMemoryState, parent: _MutableSpan | None, options: SpanOptions) -> _MutableSpan:
    span_id = state.next_span_id
    state.next_span_id += 1
    return _MutableSpan(
        id=span_id,
        parent_id=parent.id if parent is not None else None,
        name=options["name"],
        attributes=_copy_attributes(options.get("attributes")),
    )


class _InMemorySpan:
    def __init__(self, state: _InMemoryState, recorded: _MutableSpan) -> None:
        self._state = state
        self._recorded = recorded

    async def start_span(
        self,
        options: SpanOptions,
        callback: Callable[[TelemetrySpan], T | Awaitable[T]],
    ) -> T:
        return await _start_in_memory_span(self._state, self._recorded, options, callback)

    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None:
        if self._recorded.settled:
            return
        try:
            self._recorded.events.append(_MutableEvent(name, _copy_attributes(attributes)))
        except Exception:
            pass

    def set_attributes(self, attributes: SpanAttributes) -> None:
        if self._recorded.settled:
            return
        try:
            self._recorded.attributes = _merge_attributes(self._recorded.attributes, attributes)
        except Exception:
            pass

    def set_status(self, status: SpanStatus) -> None:
        if self._recorded.settled:
            return
        try:
            self._recorded.status = _copy_status(status)
            self._recorded.explicit_status = True
        except Exception:
            pass


async def _start_in_memory_span(
    state: _InMemoryState,
    parent: _MutableSpan | None,
    options: SpanOptions,
    callback: Callable[[TelemetrySpan], T | Awaitable[T]],
) -> T:
    if parent is not None and parent.settled:
        return await NOOP_TELEMETRY_CONTEXT.start_span(options, callback)

    try:
        recorded = _create_span(state, parent, options)
        state.spans.append(recorded)
    except Exception:
        return await NOOP_TELEMETRY_CONTEXT.start_span(options, callback)

    span = _InMemorySpan(state, recorded)
    try:
        result = callback(span)  # type: ignore[arg-type]
        if hasattr(result, "__await__"):
            value = await result  # type: ignore[misc]
        else:
            value = result  # type: ignore[assignment]
    except Exception as error:
        _settle_span(state, recorded, True, error)
        raise
    _settle_span(state, recorded, False)
    return value  # type: ignore[return-value]


class InMemoryTelemetryContext(TelemetryContext):
    """Backend-neutral reference implementation that records spans in process memory."""

    def __init__(self) -> None:
        self._state = _InMemoryState()

    async def start_span(
        self,
        options: SpanOptions,
        callback: Callable[[TelemetrySpan], T | Awaitable[T]],
    ) -> T:
        return await _start_in_memory_span(self._state, None, options, callback)

    def get_spans(self) -> list[RecordedTelemetrySpan]:
        snapshots: list[RecordedTelemetrySpan] = []
        for span in self._state.spans:
            snapshots.append(
                RecordedTelemetrySpan(
                    id=span.id,
                    parent_id=span.parent_id,
                    name=span.name,
                    attributes=_copy_attributes(span.attributes),
                    events=[
                        RecordedTelemetryEvent(event.name, _copy_attributes(event.attributes))
                        for event in span.events
                    ],
                    status=_copy_status(span.status),
                    settled=span.settled,
                    end_sequence=span.end_sequence,
                )
            )
        return snapshots
