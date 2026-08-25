"""Harness event bus — mirrors harness/events.ts."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Awaitable, Literal, TypedDict


class RunStartEvent(TypedDict):
    type: Literal["run_start"]
    lane: str
    run_id: str


class RunEndEvent(TypedDict):
    type: Literal["run_end"]
    lane: str
    run_id: str
    outcome: Literal["completed", "aborted", "failed"]
    leaf_id: str


HarnessEvent = RunStartEvent | RunEndEvent
HarnessEventListener = Callable[[HarnessEvent], Awaitable[None] | None]


class WatchHandle:
    def __init__(self, snapshot: Any, start: Callable, unsubscribe: Callable[[], None]) -> None:
        self.snapshot = snapshot
        self.start = start
        self.unsubscribe = unsubscribe


class HarnessEventBus:
    def __init__(self) -> None:
        self._listeners: dict[str, set[HarnessEventListener]] = {}
        self._watch_listeners: set[Callable[[HarnessEvent], None]] = set()

    def on(self, event_type: str, listener: HarnessEventListener) -> Callable[[], None]:
        listeners = self._listeners.get(event_type) or set()
        self._listeners[event_type] = listeners

        def receive(event: HarnessEvent) -> Any:
            if event.get("type") == event_type:
                return listener(event)

        listeners.add(receive)

        def unsubscribe() -> None:
            listeners.discard(receive)
            if not listeners:
                self._listeners.pop(event_type, None)

        return unsubscribe

    def emit(self, event: HarnessEvent) -> None:
        for listener in list(self._listeners.get(event["type"], set())):
            listener(event)
        for listener in list(self._watch_listeners):
            listener(event)

    def watch(self, capture_snapshot: Callable[[], Any]) -> WatchHandle:
        listener: HarnessEventListener | None = None
        buffered: list[HarnessEvent] = []

        def receive(event: HarnessEvent) -> None:
            if listener is not None:
                listener(event)
            else:
                buffered.append(event)

        self._watch_listeners.add(receive)
        snapshot = capture_snapshot()

        def start(next_listener: HarnessEventListener) -> None:
            nonlocal listener, buffered
            while buffered:
                pending = buffered
                buffered = []
                for event in pending:
                    next_listener(event)
            listener = next_listener

        def unsubscribe() -> None:
            nonlocal buffered
            self._watch_listeners.discard(receive)
            buffered = []

        return WatchHandle(snapshot=snapshot, start=start, unsubscribe=unsubscribe)
