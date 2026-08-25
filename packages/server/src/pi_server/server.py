from __future__ import annotations

import asyncio
import uuid
from typing import Any

from pi_protocol import (
    DEFAULT_MAX_FRAME_LENGTH,
    PROTOCOL_VERSION,
    ClientMessageDecoder,
    ProtocolValidationError,
    encode_server_message,
    is_supported_protocol_version,
)

from .connection import ConnectionState, is_terminal_connection
from .errors import INTERNAL_SERVER_ERROR_MESSAGE, NOT_IMPLEMENTED_MESSAGE, InternalServerError, PiServerError
from .sessions import LiveSessionManager
from .snapshots import ServerSnapshotPublisher

DEFAULT_HANDSHAKE_TIMEOUT_MS = 5_000
MAX_UINT32 = 0xFFFF_FFFF


class _Handler:
    def __init__(self, on_data, on_close, on_error) -> None:
        self.on_data = on_data
        self.on_close = on_close
        self.on_error = on_error


class PiServer:
    def __init__(self, service: Any, options: dict[str, Any]) -> None:
        resolved = _resolve_options(options)
        self.listeners = options["listeners"]
        self.id = options.get("serverId") or options.get("server_id") or str(uuid.uuid4())
        self.max_frame_length = resolved["max_frame_length"]
        self.handshake_timeout_ms = resolved["handshake_timeout_ms"]
        self.on_error = options.get("onError") or options.get("on_error")
        self.connections: set[ConnectionState] = set()
        self.closing = False
        self.started = False
        self._close_promise = None
        self._start_promise = None
        self.sessions = LiveSessionManager(
            {
                "service": service,
                "is_closing": lambda: self.closing,
                "send_message": self.send_message,
                "close_connection": self.close_connection,
                "disconnect": self.disconnect,
                "broadcast_server_snapshot": lambda: asyncio.get_event_loop().create_task(self.snapshots.broadcast()),
                "report_error": self.report_error,
            }
        )
        self.snapshots = ServerSnapshotPublisher(
            {
                "server_id": self.id,
                "service": service,
                "connections": self.connections,
                "is_closing": lambda: self.closing,
                "list_sessions": self.sessions.list_metadata,
                "send_message": self.send_message,
                "report_error": self.report_error,
            }
        )

    @property
    def addresses(self) -> list[str]:
        result = []
        for listener in self.listeners:
            address = getattr(listener, "address", None)
            if address is not None:
                result.append(address)
        return result

    async def start(self) -> PiServer:
        if self.started:
            raise RuntimeError("PiServer is already started")
        if self._start_promise is not None:
            raise RuntimeError("PiServer is already starting")
        if self.closing:
            raise RuntimeError("PiServer is closing or closed")
        self._start_promise = self._start_internal()
        try:
            return await self._start_promise
        finally:
            self._start_promise = None

    async def _start_internal(self) -> PiServer:
        started = []
        try:
            for listener in self.listeners:
                await listener.start(self.accept)
                started.append(listener)
            self.started = True
            return self
        except Exception:
            self.closing = True
            for listener in started:
                try:
                    await listener.close()
                except Exception:
                    pass
            await self._close_server_state()
            raise

    def accept(self, connection: Any) -> _Handler:
        if self.closing:
            asyncio.get_event_loop().create_task(self.close_connection(connection))
            return _Handler(lambda _chunk: None, lambda: None, self.report_error)

        loop = asyncio.get_event_loop()
        state_box: dict[str, ConnectionState] = {}

        def on_timeout() -> None:
            state = state_box.get("state")
            if state is not None:
                loop.create_task(self.fail_protocol(state, {"code": "invalid_request", "message": "Handshake timeout"}))

        handle = loop.call_later(self.handshake_timeout_ms / 1000, on_timeout)
        state = ConnectionState(
            id=str(uuid.uuid4()),
            connection=connection,
            decoder=ClientMessageDecoder({"maxFrameLength": self.max_frame_length}),
            handshake_timeout=handle,
        )
        state_box["state"] = state
        self.connections.add(state)
        return _Handler(
            lambda chunk: self.receive(state, chunk),
            lambda: self.transport_closed(state),
            lambda error: self._on_connection_error(state, connection, error),
        )

    def _on_connection_error(self, state: ConnectionState, connection: Any, error: Exception) -> None:
        self.report_error(error)
        asyncio.get_event_loop().create_task(self._close_then_disconnect(connection, state))

    async def _close_then_disconnect(self, connection: Any, state: ConnectionState) -> None:
        await self.close_connection(connection)
        await self.disconnect(state)

    async def close(self) -> None:
        if self._close_promise is not None:
            await self._close_promise
            return
        self.closing = True
        self._close_promise = self._close_internal()
        await self._close_promise

    async def _close_internal(self) -> None:
        if self._start_promise is not None:
            try:
                await self._start_promise
            except Exception:
                pass
        try:
            for listener in self.listeners:
                await listener.close()
        finally:
            await self._close_server_state()
            self.started = False

    def receive(self, state: ConnectionState, chunk: bytes) -> None:
        if is_terminal_connection(state):
            return
        try:
            messages = state.decoder.push(chunk)
        except Exception as error:
            asyncio.get_event_loop().create_task(self.fail_protocol(state, self.to_protocol_error(error)))
            return
        for message in messages:
            if is_terminal_connection(state):
                return
            self.dispatch_message(state, message)

    def dispatch_message(self, state: ConnectionState, message: dict[str, Any]) -> None:
        if state.stage == "awaitingHello":
            if message.get("type") != "hello":
                asyncio.get_event_loop().create_task(
                    self.fail_protocol(state, {"code": "invalid_request", "message": "The first client message must be hello"})
                )
                return
            state.stage = "handshaking"
            state.handshake = self.finish_handshake(state, message)
            return
        if message.get("type") == "hello":
            asyncio.get_event_loop().create_task(
                self.fail_protocol(state, {"code": "invalid_request", "message": "hello may only be sent as the first message"})
            )
            return
        if state.stage == "ready":
            asyncio.get_event_loop().create_task(self.handle_request(state, message))
            return
        if state.stage != "handshaking":
            return
        handshake = state.handshake
        if handshake is None:
            return

        async def after_handshake() -> None:
            await handshake
            if state.stage == "ready" and not state.disconnected:
                await self.handle_request(state, message)

        asyncio.get_event_loop().create_task(after_handshake())

    async def finish_handshake(self, state: ConnectionState, hello: dict[str, Any]) -> None:
        if not is_supported_protocol_version(hello.get("version")):
            await self.fail_protocol(
                state,
                {
                    "code": "version",
                    "message": f"Unsupported protocol version {hello.get('version')}; expected {PROTOCOL_VERSION}",
                },
            )
            return
        snapshot = await self.snapshots.get()
        if self.closing or state.disconnected or state.stage != "handshaking" or state.connection.closed:
            return
        sent = await self.send_message(
            state,
            {"type": "hello", "version": PROTOCOL_VERSION, "connectionId": state.id, "snapshot": snapshot},
        )
        if sent and not state.disconnected and state.stage == "handshaking":
            state.handshake_complete = True
            state.stage = "ready"
            state.handshake_timeout.cancel()
            if snapshot.get("revision") != self.snapshots.current_revision:
                current = await self.snapshots.get()
                await self.send_message(state, {"type": "event", "event": {"type": "server_snapshot", "snapshot": current}})

    async def handle_request(self, state: ConnectionState, envelope: dict[str, Any]) -> None:
        try:
            result = await self.sessions.execute_command(state, envelope["request"])
            await self.send_message(state, {"type": "response", "id": envelope["id"], "ok": True, "result": result})
        except Exception as error:
            await self.send_message(
                state, {"type": "response", "id": envelope["id"], "ok": False, "error": self.to_protocol_error(error)}
            )

    def transport_closed(self, connection: ConnectionState) -> None:
        if not connection.disconnected and connection.stage != "closing":
            try:
                connection.decoder.end()
            except Exception as error:
                self.report_error(error)
        asyncio.get_event_loop().create_task(self.disconnect(connection))

    async def disconnect(self, connection: ConnectionState) -> None:
        if connection.disconnected:
            return
        handshake_complete = connection.handshake_complete
        connection.disconnected = True
        connection.stage = "closed"
        connection.handshake_timeout.cancel()
        self.connections.discard(connection)
        await self.sessions.disconnect(connection)
        if not self.closing and handshake_complete:
            asyncio.get_event_loop().create_task(self.snapshots.broadcast())

    async def send_message(self, connection: ConnectionState, message: dict[str, Any]) -> bool:
        if connection.disconnected or connection.connection.closed:
            return False
        try:
            frame = encode_server_message(message, {"maxFrameLength": self.max_frame_length})
        except Exception as error:
            self.report_error(error)
            await self.close_connection(connection.connection)
            await self.disconnect(connection)
            return False
        try:
            await connection.connection.send(frame)
            return True
        except Exception as error:
            self.report_error(error)
            await self.close_connection(connection.connection)
            await self.disconnect(connection)
            return False

    async def fail_protocol(self, connection: ConnectionState, error: dict[str, Any]) -> None:
        if connection.disconnected or connection.stage in {"closing", "closed"}:
            return
        connection.stage = "closing"
        connection.handshake_timeout.cancel()
        final_frame = None
        try:
            final_frame = encode_server_message({"type": "hello_error", "error": error}, {"maxFrameLength": self.max_frame_length})
        except Exception as encode_error:
            self.report_error(encode_error)
        await self.close_connection(connection.connection, final_frame)
        await self.disconnect(connection)

    async def _close_server_state(self) -> None:
        connections = list(self.connections)
        for connection in connections:
            connection.stage = "closing"
            connection.handshake_timeout.cancel()
        for connection in connections:
            await self.close_connection(connection.connection)
        for connection in connections:
            await self.disconnect(connection)
        await self.sessions.close()
        self.connections.clear()

    async def close_connection(self, connection: Any, final_chunk: bytes | None = None) -> None:
        try:
            result = connection.close(final_chunk)
            if hasattr(result, "__await__"):
                await result
        except Exception as error:
            self.report_error(error)

    def to_protocol_error(self, error: object) -> dict[str, Any]:
        if isinstance(error, InternalServerError):
            self.report_error(error.cause)
            return {"code": "internal_error", "message": INTERNAL_SERVER_ERROR_MESSAGE}
        if isinstance(error, PiServerError):
            if error.code == "not_implemented":
                return {"code": "not_implemented", "message": NOT_IMPLEMENTED_MESSAGE}
            payload = {"code": error.code, "message": str(error)}
            if error.details is not None:
                payload["details"] = error.details
            return payload
        if isinstance(error, ProtocolValidationError):
            return {"code": "invalid_request", "message": str(error)}
        self.report_error(error)
        return {"code": "internal_error", "message": INTERNAL_SERVER_ERROR_MESSAGE}

    def report_error(self, error: object) -> None:
        try:
            if self.on_error:
                self.on_error(error if isinstance(error, Exception) else Exception(str(error)))
        except Exception:
            pass


def _resolve_options(options: dict[str, Any]) -> dict[str, int]:
    if not isinstance(options.get("listeners"), (list, tuple)):
        raise TypeError("PiServer listeners must be an array")
    if options.get("serverId") == "" or options.get("server_id") == "":
        raise TypeError("PiServer serverId must not be empty")
    max_frame_length = options.get("maxFrameLength", options.get("max_frame_length", DEFAULT_MAX_FRAME_LENGTH))
    if not isinstance(max_frame_length, int) or max_frame_length <= 0 or max_frame_length > MAX_UINT32:
        raise TypeError(f"PiServer maxFrameLength must be an integer between 1 and {MAX_UINT32}")
    handshake_timeout_ms = options.get("handshakeTimeoutMs", options.get("handshake_timeout_ms", DEFAULT_HANDSHAKE_TIMEOUT_MS))
    if not isinstance(handshake_timeout_ms, int) or handshake_timeout_ms <= 0:
        raise TypeError("PiServer handshakeTimeoutMs must be a positive integer")
    return {"max_frame_length": max_frame_length, "handshake_timeout_ms": handshake_timeout_ms}
