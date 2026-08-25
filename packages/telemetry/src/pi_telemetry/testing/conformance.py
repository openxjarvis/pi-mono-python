from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..index import SpanAttributes, SpanOptions, SpanStatus, TelemetrySpan
from ..memory import RecordedTelemetrySpan
from .types import (
    TelemetryAdapterConformanceCase,
    TelemetryAdapterFixture,
    TelemetryAdapterFixtureFactory,
)

ConformanceTest = Callable[[TelemetryAdapterFixture], Awaitable[None]]


def _create_case(
    factory: TelemetryAdapterFixtureFactory,
    group: str,
    name: str,
    test: ConformanceTest,
) -> TelemetryAdapterConformanceCase:
    async def run() -> None:
        fixture = await factory()
        async with fixture:
            await test(fixture)

    return TelemetryAdapterConformanceCase(group, name, run)


def _find_span(spans: list[RecordedTelemetrySpan], name: str) -> RecordedTelemetrySpan:
    for candidate in spans:
        if candidate.name == name:
            return candidate
    raise AssertionError(f"Expected recorded span {name}")


async def _rejects_with_same_value(operation: Awaitable[object], expected: object) -> None:
    try:
        await operation
    except Exception as error:
        if error is not expected:
            raise AssertionError("Expected operation to reject with the same value") from error
        return
    raise AssertionError("Expected operation to reject")


class _Unreadable:
    def __init__(self, value: object) -> None:
        self._value = value

    def __getattribute__(self, name: str) -> object:
        if name in {"_value", "__class__"}:
            return object.__getattribute__(self, name)
        raise RuntimeError("read")

    def items(self):  # noqa: ANN201
        raise RuntimeError("enumerate")

    def keys(self):  # noqa: ANN201
        raise RuntimeError("enumerate")

    def values(self):  # noqa: ANN201
        raise RuntimeError("enumerate")

    def get(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("read")

    def __iter__(self):  # noqa: ANN201
        raise RuntimeError("enumerate")


def _unreadable(value: object) -> object:
    return _Unreadable(value)


def create_telemetry_adapter_conformance(
    factory: TelemetryAdapterFixtureFactory,
) -> list[TelemetryAdapterConformanceCase]:
    """Creates runner-independent cases for the callback telemetry adapter contract."""

    async def admits_once(fixture: TelemetryAdapterFixture) -> None:
        admitted = False
        calls = 0
        expected = {"value": 42}

        def callback(_span: TelemetrySpan) -> dict[str, int]:
            nonlocal admitted, calls
            admitted = True
            calls += 1
            return expected

        result = fixture.context.start_span({"name": "success"}, callback)
        assert admitted is True
        assert calls == 1
        assert await result == expected
        assert _find_span(await fixture.get_spans(), "success").status == {"status": "ok"}
        assert _find_span(await fixture.get_spans(), "success").settled is True

    async def preserves_rejections(fixture: TelemetryAdapterFixture) -> None:
        sync_error = RuntimeError("sync")

        def throw_sync(_span: TelemetrySpan) -> None:
            raise sync_error

        await _rejects_with_same_value(fixture.context.start_span({"name": "sync-error"}, throw_sync), sync_error)

        async_error = {"kind": "async"}

        async def throw_async(_span: TelemetrySpan) -> None:
            raise ExceptionWrapper(async_error)

        try:
            await fixture.context.start_span({"name": "async-error"}, throw_async)
            raise AssertionError("Expected operation to reject")
        except ExceptionWrapper as error:
            assert error.value is async_error

        spans = await fixture.get_spans()
        for name in ("sync-error", "async-error"):
            assert _find_span(spans, name).status["status"] == "error"

    async def last_explicit_status(fixture: TelemetryAdapterFixture) -> None:
        def set_ok(span: TelemetrySpan) -> None:
            span.set_status({"status": "error", "error": {"name": "Expected", "message": "first"}})
            span.set_status({"status": "ok"})

        await fixture.context.start_span({"name": "last-status"}, set_ok)
        assert _find_span(await fixture.get_spans(), "last-status").status == {"status": "ok"}

    async def merges_attributes(fixture: TelemetryAdapterFixture) -> None:
        def record(span: TelemetrySpan) -> None:
            span.set_attributes({"count": 1, "overwrite": "middle"})
            span.set_attributes({"count": None, "overwrite": "end"})
            span.add_event("first", {"index": 1, "ignored": None})
            span.add_event("second", {"index": 2})

        await fixture.context.start_span(
            {"name": "recording", "attributes": {"start": "value", "overwrite": "start", "ignored": None}},
            record,
        )
        span = _find_span(await fixture.get_spans(), "recording")
        assert span.attributes == {"start": "value", "overwrite": "end", "count": 1}
        assert [event.name for event in span.events] == ["first", "second"]

    async def settled_calls_inert(fixture: TelemetryAdapterFixture) -> None:
        settled: TelemetrySpan | None = None

        def capture(span: TelemetrySpan) -> None:
            nonlocal settled
            settled = span

        await fixture.context.start_span({"name": "settled", "attributes": {"value": "initial"}}, capture)
        assert settled is not None
        settled.set_attributes({"value": "late"})
        settled.add_event("late", {"value": True})
        settled.set_status({"status": "error"})
        child_admitted = False

        def child(_span: TelemetrySpan) -> int:
            nonlocal child_admitted
            child_admitted = True
            return 7

        child_result = settled.start_span({"name": "late-child"}, child)
        assert child_admitted is True
        assert await child_result == 7
        spans = await fixture.get_spans()
        assert len(spans) == 1
        assert spans[0].attributes == {"value": "initial"}
        assert spans[0].events == []
        assert spans[0].status == {"status": "ok"}

    async def parentage(fixture: TelemetryAdapterFixture) -> None:
        release_first: Callable[[], None] | None = None
        first_gate: Awaitable[None] | None = None

        import asyncio

        gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        async def parent(span: TelemetrySpan) -> None:
            async def first(_child: TelemetrySpan) -> None:
                await gate

            first_task = span.start_span({"name": "first-child"}, first)

            def second(_child: TelemetrySpan) -> str:
                return "done"

            assert await span.start_span({"name": "second-child"}, second) == "done"
            gate.set_result(None)
            await first_task

        await fixture.context.start_span({"name": "parent"}, parent)
        spans = await fixture.get_spans()
        parent_span = _find_span(spans, "parent")
        first = _find_span(spans, "first-child")
        second = _find_span(spans, "second-child")
        assert parent_span.parent_id is None
        assert first.parent_id == parent_span.id
        assert second.parent_id == parent_span.id
        assert second.end_sequence is not None and first.end_sequence is not None and parent_span.end_sequence is not None
        assert second.end_sequence < first.end_sequence
        assert first.end_sequence < parent_span.end_sequence

    return [
        _create_case(factory, "callback lifecycle", "admits once synchronously and preserves the result", admits_once),
        _create_case(factory, "callback lifecycle", "preserves synchronous and asynchronous rejection values", preserves_rejections),
        _create_case(factory, "status", "uses last explicit status without automatic overwrite", last_explicit_status),
        _create_case(factory, "recording", "merges attributes and records ordered events", merges_attributes),
        _create_case(factory, "recording", "makes calls after settlement inert", settled_calls_inert),
        _create_case(factory, "parentage", "records nested and concurrent child relationships", parentage),
    ]


class ExceptionWrapper(Exception):
    def __init__(self, value: object) -> None:
        super().__init__(str(value))
        self.value = value
