from __future__ import annotations

import asyncio
from typing import Any, Callable, Literal

from .transcript import (
    TranscriptState,
    apply_transcript_progress,
    apply_transcript_snapshot,
    create_transcript_state,
    select_transcript,
)

RemoteSessionOperation = Literal["open", "create", "submit", "abort", "setModel", "setThinking", "reconnect"]


class RemoteSessionDisposedError(Exception):
    def __init__(self) -> None:
        super().__init__("Remote session is disposed")
        self.name = "RemoteSessionDisposedError"


async def settle_remote_session_disposal(cleanup: list[Any]) -> None:
    results = await asyncio.gather(*cleanup, return_exceptions=True)
    errors = [result for result in results if isinstance(result, Exception) and not isinstance(result, RemoteSessionDisposedError)]
    if len(errors) == 1:
        raise errors[0]
    if len(errors) > 1:
        raise ExceptionGroup("Failed to dispose remote session", errors)


class RemoteSession:
    def __init__(self, client: Any, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        self._client = client
        self._on_listener_error = options.get("onListenerError") or options.get("on_listener_error")
        self._lifecycle: dict[str, Any] = {"status": "unbound"}
        self._handle: Any | None = None
        self._transcript: TranscriptState | None = None
        self._unsubscribe_snapshot: Callable[[], None] | None = None
        self._unsubscribe_events: Callable[[], None] | None = None
        self._listeners: set[Callable[[dict[str, Any]], None]] = set()
        self._pending_attachment_operations: set[asyncio.Task] = set()
        self._active_operation_states: set[int] = set()
        self._dispose_promise: asyncio.Future | None = None
        self._dispose_event = asyncio.Event()

    @property
    def id(self) -> str | None:
        return getattr(self._handle, "id", None)

    @property
    def state(self) -> dict[str, Any]:
        return {
            "lifecycle": self._lifecycle,
            "snapshot": self._transcript.snapshot if self._transcript else None,
            "transcript": select_transcript(self._transcript) if self._transcript else [],
        }

    @property
    def snapshot(self) -> dict[str, Any] | None:
        return self._transcript.snapshot if self._transcript else None

    @property
    def phase(self) -> str | None:
        return None if self.snapshot is None else self.snapshot.get("phase")

    @property
    def operation(self) -> str | None:
        return self._lifecycle.get("operation") if self._lifecycle.get("status") == "busy" else None

    @property
    def models(self) -> list[dict[str, Any]]:
        snapshot = getattr(self._client, "snapshot", None) or {}
        return list(snapshot.get("models") or [])

    @property
    def sessions(self) -> list[dict[str, Any]]:
        snapshot = getattr(self._client, "snapshot", None) or {}
        return list(snapshot.get("sessions") or [])

    @property
    def connection_state(self) -> str:
        return getattr(self._client, "connection_state", "disconnected")

    @property
    def disposed(self) -> bool:
        return self._lifecycle.get("status") == "disposed"

    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        self._assert_not_disposed()
        self._listeners.add(listener)
        self._call_listener(listener, self.state)
        return lambda: self._listeners.discard(listener)

    def on_connection_state_change(self, listener: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        self._assert_not_disposed()
        return self._client.on_connection_state_change(listener)

    @classmethod
    async def open(cls, client: Any, session_id: str, options: dict[str, Any] | None = None) -> RemoteSession:
        session = cls(client, options)
        try:
            await session.open_session(session_id)
            return session
        except Exception:
            await session.dispose()
            raise

    async def open_session(self, session_id: str) -> None:
        if getattr(self._handle, "id", None) == session_id and self._lifecycle.get("status") == "ready":
            return
        await self._replace("open", lambda: self._client.acquire_session(session_id, {"mode": "exclusive"}))

    @classmethod
    async def create(cls, client: Any, create_options: dict[str, Any], options: dict[str, Any] | None = None) -> RemoteSession:
        session = cls(client, options)
        try:
            await session.create_session(create_options)
            return session
        except Exception:
            await session.dispose()
            raise

    async def create_session(self, options: dict[str, Any]) -> None:
        await self._replace("create", lambda: self._client.create_session(options))

    async def submit(self, text: str) -> None:
        normalized = text.strip()
        if not normalized:
            return
        self._assert_available()
        handle = self._require_handle()
        if self.phase not in {None, "idle", "turn"}:
            raise RuntimeError(f"Session cannot accept input during {self.phase or 'unknown'} phase")
        await self._run_operation(
            "submit",
            lambda: (handle.prompt(normalized) if self.phase == "idle" else handle.steer(normalized)),
        )

    async def abort(self) -> None:
        preempting_submit = self._lifecycle.get("status") == "busy" and self._lifecycle.get("operation") == "submit"
        if preempting_submit:
            self._assert_not_disposed()
        else:
            self._assert_available()
        handle = self._require_handle()
        if self.phase == "idle" and not preempting_submit:
            return
        await self._run_operation("abort", handle.abort, preempting_submit)

    async def set_model(self, model: dict[str, str]) -> None:
        await self._run_idle_operation("setModel", "change model", lambda: self._require_handle().set_model(model))

    async def set_thinking(self, thinking_level: str) -> None:
        await self._run_idle_operation(
            "setThinking", "change thinking level", lambda: self._require_handle().set_thinking(thinking_level)
        )

    async def reconnect(self) -> None:
        self._assert_available()
        session_id = self._require_handle().id

        async def run() -> None:
            await self._client.reconnect()
            handle = await self._client.acquire_session(session_id, {"mode": "exclusive"})
            await self._assert_not_disposed_after_await(handle)
            self._bind(handle)

        await self._run_operation("reconnect", run)

    async def dispose(self) -> None:
        if self._dispose_promise is not None:
            await self._dispose_promise
            return
        handle = self._handle
        self._lifecycle = {"status": "disposed"}
        self._dispose_event.set()
        self._clear_subscriptions()
        self._handle = None
        self._transcript = None
        cleanup = [task for task in list(self._pending_attachment_operations)]
        if handle is not None:
            cleanup.append(asyncio.ensure_future(handle.dispose()))
        self._dispose_promise = asyncio.ensure_future(settle_remote_session_disposal(cleanup))
        self._notify()
        self._listeners.clear()
        await self._dispose_promise

    async def __aenter__(self) -> RemoteSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.dispose()

    async def _replace(self, operation: str, prepare) -> None:
        self._assert_available()
        if self._handle is not None and self.phase not in {None, "idle"}:
            raise RuntimeError(f"Cannot {operation} a session while session is {self.phase or 'unavailable'}")
        await self._run_operation(operation, lambda: self._prepare_replacement(operation, prepare))

    async def _prepare_replacement(self, operation: str, prepare) -> None:
        previous = self._handle
        next_handle = await prepare()
        await self._assert_not_disposed_after_await(next_handle)
        snapshot = getattr(next_handle, "snapshot", None)
        if not snapshot:
            await next_handle.dispose()
            raise RuntimeError(f"Session {next_handle.id} did not provide a snapshot")
        if previous and previous.id != next_handle.id and previous.attached and self.phase not in {None, "idle"}:
            await next_handle.dispose()
            raise RuntimeError(f"Cannot {operation} a session while session is {self.phase or 'unavailable'}")
        if previous and previous.id != next_handle.id and previous.attached:
            try:
                await previous.detach()
            except Exception as error:
                try:
                    await next_handle.dispose()
                except Exception as cleanup_error:
                    raise ExceptionGroup("Failed to replace remote session attachment", [error, cleanup_error]) from error
                raise
        await self._assert_not_disposed_after_await(next_handle)
        self._bind(next_handle, snapshot)

    async def _run_idle_operation(self, operation: str, description: str, run) -> None:
        self._assert_available()
        self._require_handle()
        if self.phase not in {None, "idle"}:
            raise RuntimeError(f"Cannot {description} while session is {self.phase or 'unavailable'}")
        await self._run_operation(operation, run)

    async def _run_operation(self, operation: RemoteSessionOperation, run, preempt: bool = False) -> None:
        if preempt:
            self._assert_not_disposed()
        else:
            self._assert_available()
        previous = self._lifecycle
        busy = {"status": "busy", "operation": operation}
        busy_id = id(busy)
        self._lifecycle = busy
        self._active_operation_states.add(busy_id)
        self._notify()
        try:
            result = run()
            if hasattr(result, "__await__"):
                dispose_wait = asyncio.create_task(self._dispose_event.wait())
                running = asyncio.ensure_future(result)
                done, pending = await asyncio.wait({running, dispose_wait}, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                if dispose_wait in done and self.disposed:
                    raise RemoteSessionDisposedError()
                await running
        finally:
            self._active_operation_states.discard(busy_id)
            if not self.disposed and self._lifecycle is busy:
                if preempt and id(previous) in self._active_operation_states:
                    self._lifecycle = previous
                else:
                    self._lifecycle = {"status": "ready"} if self._handle else {"status": "unbound"}
                self._notify()

    def _bind(self, handle: Any, known_snapshot: dict[str, Any] | None = None) -> None:
        snapshot = known_snapshot or getattr(handle, "snapshot", None)
        if not snapshot:
            raise RuntimeError(f"Session {handle.id} did not provide a snapshot")
        self._clear_subscriptions()
        self._handle = handle
        self._transcript = create_transcript_state(snapshot)
        self._unsubscribe_snapshot = handle.subscribe(self._on_snapshot)
        self._unsubscribe_events = handle.on_event(self._handle_event)

    def _on_snapshot(self, next_snapshot: dict[str, Any]) -> None:
        if self._transcript is None:
            return
        self._transcript = apply_transcript_snapshot(self._transcript, next_snapshot)
        self._notify()

    def _handle_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "session_removed":
            self._clear_subscriptions()
            self._handle = None
            self._transcript = None
            if self._lifecycle.get("status") != "busy":
                self._lifecycle = {"status": "unbound"}
            self._notify()
            return
        if event.get("type") != "session_progress" or self._transcript is None:
            return
        self._transcript = apply_transcript_progress(self._transcript, event["progress"])
        self._notify()

    def _notify(self) -> None:
        state = self.state
        for listener in list(self._listeners):
            self._call_listener(listener, state)

    def _call_listener(self, listener: Callable[[dict[str, Any]], None], state: dict[str, Any]) -> None:
        try:
            listener(state)
        except Exception as error:
            if not self._on_listener_error:
                return
            try:
                self._on_listener_error(error if isinstance(error, Exception) else Exception(str(error)))
            except Exception:
                pass

    def _clear_subscriptions(self) -> None:
        if self._unsubscribe_snapshot:
            self._unsubscribe_snapshot()
        if self._unsubscribe_events:
            self._unsubscribe_events()
        self._unsubscribe_snapshot = None
        self._unsubscribe_events = None

    def _require_handle(self) -> Any:
        if self._handle is None:
            raise RuntimeError("No remote session is attached")
        return self._handle

    def _assert_available(self) -> None:
        self._assert_not_disposed()
        if self._lifecycle.get("status") == "busy":
            raise RuntimeError(f"Remote session is busy with {self._lifecycle.get('operation')}")

    def _assert_not_disposed(self) -> None:
        if self.disposed:
            raise RuntimeError("Remote session is disposed")

    async def _assert_not_disposed_after_await(self, handle: Any) -> None:
        if not self.disposed:
            return
        await handle.dispose()
        raise RemoteSessionDisposedError()
