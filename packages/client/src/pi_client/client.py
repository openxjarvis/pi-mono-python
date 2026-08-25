from __future__ import annotations

from typing import Any, Callable

from pi_protocol import ProtocolValidationError, encode_client_message

from .connection import Connection
from .errors import (
    PiClientDisposedError,
    PiDisconnectedError,
    PiServerError,
    PiSessionDetachedError,
    PiSessionOwnershipError,
    to_error,
)
from .promise import create_promise_resolvers
from .session_handle import SessionHandle, SessionHandleCallbacks
from .state import ClientState


class _LeaseCallbacks:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def is_attached(self) -> bool:
        return self._is_active()

    def get_snapshot(self) -> dict[str, Any] | None:
        return self._get_snapshot()

    def subscribe(self, listener):
        return self._subscribe(listener)

    def on_event(self, listener):
        return self._on_event(listener)

    def detach(self):
        return self._release(False)

    def dispose(self):
        return self._release(True)

    def request(self, command):
        return self._request(command)


class PiClient:
    def __init__(self, options: dict[str, Any]) -> None:
        self._options = options
        factory = options.get("transportFactory") or options.get("transport_factory")
        max_frame = options.get("maxFrameLength", options.get("max_frame_length"))
        on_error = options.get("onListenerError") or options.get("on_listener_error")
        self._state = ClientState(on_error)
        self._connection = Connection(
            transport_factory=factory,
            max_frame_length=max_frame,
            on_handshake=self._state.apply_server_snapshot,
            on_message=self._handle_message,
            on_state_change=self._handle_connection_state_change,
        )
        self._pending_requests: dict[str, dict[str, Any]] = {}
        self._session_lease_counts: dict[str, int] = {}
        self._exclusive_session_leases: dict[str, dict[str, str]] = {}
        self._session_lease_generations: dict[str, int] = {}
        self._session_attachments: dict[str, Any] = {}
        self._session_detachments: dict[str, Any] = {}
        self._session_cleanup_required: set[str] = set()
        self._session_reconciliations: dict[str, Any] = {}
        self._connection_state_listeners: set[Callable[[dict[str, Any]], None]] = set()
        self._request_sequence = 0
        self._disposed = False

    @property
    def disposed(self) -> bool:
        return self._disposed

    @property
    def connection_state(self) -> str:
        return self._connection.state

    @property
    def connected(self) -> bool:
        return self._connection.state == "connected"

    @property
    def snapshot(self) -> dict[str, Any] | None:
        return self._state.snapshot

    @classmethod
    async def connect(cls, options: dict[str, Any]) -> PiClient:
        client = cls(options)
        try:
            await client.connect_client()
            return client
        except Exception:
            await client.dispose()
            raise

    async def connect_client(self) -> dict[str, Any]:
        if self._disposed:
            raise PiClientDisposedError()
        if self._connection.state == "disconnected":
            self._state.reset()
        return await self._connection.connect()

    async def reconnect(self) -> dict[str, Any]:
        return await self.connect_client()

    def disconnect(self, reason: str = "Client disconnected") -> None:
        self._connection.disconnect(reason)

    def subscribe(self, listener: Callable[[dict[str, Any]], None]):
        self._assert_not_disposed()
        return self._state.subscribe(listener)

    def on_event(self, listener: Callable[[dict[str, Any]], None]):
        self._assert_not_disposed()
        return self._state.on_event(listener)

    def on_connection_state_change(self, listener: Callable[[dict[str, Any]], None]):
        self._assert_not_disposed()
        self._connection_state_listeners.add(listener)
        return lambda: self._connection_state_listeners.discard(listener)

    async def list_sessions(self) -> list[dict[str, Any]]:
        return (await self._request({"command": "list"}))["sessions"]

    async def create_session(self, options: dict[str, Any] | None = None) -> SessionHandle:
        result = await self._request({"command": "create", **(options or {})})
        token = self._reserve_session_lease(result["session"]["id"], "exclusive")
        return self._create_session_lease(result["session"]["id"], token)

    async def attach_session(self, session_id: str) -> SessionHandle:
        return await self.acquire_session(session_id, {"mode": "shared"})

    async def acquire_session(self, session_id: str, options: dict[str, Any]) -> SessionHandle:
        self._assert_not_disposed()
        token = self._reserve_session_lease(session_id, options.get("mode", "shared"))
        try:
            detachment = self._session_detachments.get(session_id)
            if detachment is not None:
                try:
                    await detachment
                except Exception:
                    pass
            reconciled = False
            if session_id in self._session_cleanup_required:
                reconciled = await self._reconcile_session_cleanup(session_id)
            if reconciled or not self._state.is_session_attached(session_id):
                attachment = self._session_attachments.get(session_id)
                if attachment is None:
                    attachment = self._attach_session(session_id)
                    self._session_attachments[session_id] = attachment
                try:
                    await attachment
                finally:
                    if self._session_attachments.get(session_id) is attachment:
                        self._session_attachments.pop(session_id, None)
            return self._create_session_lease(session_id, token)
        except Exception:
            self._release_session_lease(session_id, token)
            raise

    async def _attach_session(self, session_id: str) -> None:
        previous = self._state.forget_session_snapshot(session_id)
        try:
            await self._request({"command": "attach", "sessionId": session_id})
        except Exception:
            if previous:
                self._state.restore_session_snapshot(previous)
            raise

    async def _request(self, command: dict[str, Any]) -> dict[str, Any]:
        if self._disposed:
            raise PiClientDisposedError()
        if not self.connected:
            raise PiDisconnectedError()
        self._request_sequence += 1
        request_id = f"request-{self._request_sequence}"
        resolvers = create_promise_resolvers()
        self._pending_requests[request_id] = {"command": command, "resolvers": resolvers}
        try:
            frame = encode_client_message(
                {"type": "request", "id": request_id, "request": command},
                {"maxFrameLength": self._connection.max_frame_length},
            )
        except Exception as error:
            pending = self._take_pending_request(request_id)
            if pending:
                pending["resolvers"].reject(to_error(error))
            return await resolvers.promise
        self._connection.send(frame)
        return await resolvers.promise

    def _create_session_lease(self, session_id: str, token: dict[str, str]) -> SessionHandle:
        generation = self._session_lease_generations.get(session_id, 0)
        self._session_lease_generations[session_id] = generation
        state = {"value": "active", "release": None}

        def refresh_state() -> None:
            if state["value"] in {"active", "releasing"} and self._session_lease_generations.get(session_id) != generation:
                state["value"] = "invalidated"

        def is_active() -> bool:
            refresh_state()
            return state["value"] == "active" and self._state.is_session_attached(session_id)

        def assert_active() -> None:
            self._assert_not_disposed()
            if not self.connected:
                raise PiDisconnectedError()
            if not is_active():
                raise PiSessionDetachedError(session_id)

        async def release(relinquish_on_failure: bool) -> None:
            refresh_state()
            if state["value"] in {"released", "invalidated"}:
                return
            if state["release"] is not None:
                await state["release"]
                return
            assert_active()
            state["value"] = "releasing"

            async def _do_release() -> None:
                count = self._session_lease_counts.get(session_id, 0)
                if count <= 1:
                    detachment = self._request({"command": "detach", "sessionId": session_id})
                    self._session_detachments[session_id] = detachment
                    try:
                        await detachment
                        self._release_session_lease(session_id, token)
                    finally:
                        if self._session_detachments.get(session_id) is detachment:
                            self._session_detachments.pop(session_id, None)
                else:
                    self._release_session_lease(session_id, token)
                state["value"] = "released"

            try:
                state["release"] = _do_release()
                await state["release"]
            except Exception:
                refresh_state()
                if state["value"] == "invalidated":
                    return
                if relinquish_on_failure:
                    self._release_session_lease(session_id, token)
                    self._session_cleanup_required.add(session_id)
                    state["value"] = "released"
                else:
                    state["value"] = "active"
                    state["release"] = None
                raise

        callbacks = _LeaseCallbacks(
            _is_active=is_active,
            _get_snapshot=lambda: self._state.get_session_snapshot(session_id) if is_active() else None,
            _subscribe=lambda listener: (
                assert_active() or self._state.subscribe_session(session_id, lambda snapshot: listener(snapshot) if is_active() else None)
            ),
            _on_event=lambda listener: (
                assert_active()
                or self._state.on_session_event(
                    session_id,
                    lambda event: listener(event) if is_active() or event.get("type") == "session_removed" else None,
                )
            ),
            _release=release,
            _request=lambda command: (assert_active() or True) and self._request(command),
        )
        return SessionHandle(session_id, callbacks)  # type: ignore[arg-type]

    def _handle_message(self, message: dict[str, Any]) -> None:
        if message.get("type") == "event":
            if message["event"].get("type") == "session_removed":
                self._invalidate_session_leases(message["event"]["sessionId"])
            self._state.apply_event(message["event"])
            return
        pending = self._take_pending_request(message["id"])
        if pending is None:
            self._connection.fail(ProtocolValidationError("Response has no matching request"))
            return
        if not message.get("ok"):
            pending["resolvers"].reject(PiServerError(message["error"]))
            return
        if message["result"].get("command") != pending["command"].get("command"):
            error = ProtocolValidationError(
                f"Response command {message['result']['command']} does not match {pending['command']['command']}"
            )
            pending["resolvers"].reject(error)
            self._connection.fail(error)
            return
        self._state.apply_result(message["result"])
        pending["resolvers"].resolve(message["result"])

    def _handle_connection_state_change(self, change: dict[str, Any]) -> None:
        if change.get("state") == "disconnected":
            self._state.clear_attachments()
            self._invalidate_all_session_leases()
            self._reject_pending_requests(change.get("error") or PiDisconnectedError())
        for listener in list(self._connection_state_listeners):
            try:
                listener(change)
            except Exception as error:
                handler = self._options.get("onListenerError") or self._options.get("on_listener_error")
                if handler:
                    try:
                        handler(to_error(error))
                    except Exception:
                        pass

    def _take_pending_request(self, request_id: str) -> dict[str, Any] | None:
        return self._pending_requests.pop(request_id, None)

    def _reject_pending_requests(self, error: Exception) -> None:
        requests = list(self._pending_requests.values())
        self._pending_requests.clear()
        for request in requests:
            request["resolvers"].reject(error)

    async def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        error = PiClientDisposedError()
        self._reject_pending_requests(error)
        self._connection.disconnect(error)
        self._state.dispose()
        self._invalidate_all_session_leases()
        self._connection_state_listeners.clear()

    async def __aenter__(self) -> PiClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.dispose()

    def _assert_not_disposed(self) -> None:
        if self._disposed:
            raise PiClientDisposedError()

    async def _reconcile_session_cleanup(self, session_id: str) -> bool:
        if session_id not in self._session_cleanup_required:
            return False
        reconciliation = self._session_reconciliations.get(session_id)
        if reconciliation is None:

            async def _run() -> None:
                try:
                    await self._request({"command": "detach", "sessionId": session_id})
                    self._session_cleanup_required.discard(session_id)
                finally:
                    self._session_reconciliations.pop(session_id, None)

            reconciliation = _run()
            self._session_reconciliations[session_id] = reconciliation
        await reconciliation
        return True

    def _reserve_session_lease(self, session_id: str, mode: str) -> dict[str, str]:
        count = self._session_lease_counts.get(session_id, 0)
        if mode == "exclusive" and count > 0:
            raise PiSessionOwnershipError(session_id, f"Session {session_id} already has an active lease")
        if mode == "shared" and session_id in self._exclusive_session_leases:
            raise PiSessionOwnershipError(session_id, f"Session {session_id} has an exclusive lease")
        token = {"mode": mode}
        self._session_lease_counts[session_id] = count + 1
        if mode == "exclusive":
            self._exclusive_session_leases[session_id] = token
        return token

    def _release_session_lease(self, session_id: str, token: dict[str, str]) -> None:
        count = self._session_lease_counts.get(session_id, 0)
        if count <= 1:
            self._session_lease_counts.pop(session_id, None)
        else:
            self._session_lease_counts[session_id] = count - 1
        if self._exclusive_session_leases.get(session_id) is token:
            self._exclusive_session_leases.pop(session_id, None)

    def _invalidate_session_leases(self, session_id: str) -> None:
        self._session_lease_counts.pop(session_id, None)
        self._exclusive_session_leases.pop(session_id, None)
        self._session_cleanup_required.discard(session_id)
        self._session_lease_generations[session_id] = self._session_lease_generations.get(session_id, 0) + 1

    def _invalidate_all_session_leases(self) -> None:
        for session_id in list(self._session_lease_counts):
            self._invalidate_session_leases(session_id)
        self._session_cleanup_required.clear()
