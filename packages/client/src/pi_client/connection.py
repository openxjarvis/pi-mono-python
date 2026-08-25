from __future__ import annotations

from typing import Any, Callable

from pi_protocol import (
    DEFAULT_MAX_FRAME_LENGTH,
    PROTOCOL_VERSION,
    ProtocolValidationError,
    ServerMessageDecoder,
    encode_client_message,
)

from .errors import PiDisconnectedError, PiServerError, to_disconnected_error, to_error
from .promise import PromiseResolvers, create_promise_resolvers
from .transport import ByteTransport, SimpleHandlers

MAX_UINT32 = 0xFFFF_FFFF


class Connection:
    def __init__(
        self,
        *,
        transport_factory: Callable,
        max_frame_length: int | None = None,
        on_handshake: Callable[[dict[str, Any]], None],
        on_message: Callable[[dict[str, Any]], None],
        on_state_change: Callable[[dict[str, Any]], None],
    ) -> None:
        self._transport_factory = transport_factory
        self._max_frame_length = max_frame_length if max_frame_length is not None else DEFAULT_MAX_FRAME_LENGTH
        if not isinstance(self._max_frame_length, int) or self._max_frame_length <= 0 or self._max_frame_length > MAX_UINT32:
            raise TypeError(f"PiClient maxFrameLength must be an integer between 1 and {MAX_UINT32}")
        self._on_handshake = on_handshake
        self._on_message = on_message
        self._on_state_change = on_state_change
        self._lifecycle: dict[str, Any] = {"state": "disconnected"}
        self._sequence = 0

    @property
    def state(self) -> str:
        return self._lifecycle["state"]

    @property
    def max_frame_length(self) -> int:
        return self._max_frame_length

    async def connect(self) -> dict[str, Any]:
        if self._lifecycle["state"] != "disconnected":
            raise PiDisconnectedError(f"PiClient is already {self._lifecycle['state']}")
        self._sequence += 1
        identifier = self._sequence
        handshake = create_promise_resolvers()
        self._lifecycle = {
            "state": "connecting",
            "id": identifier,
            "decoder": ServerMessageDecoder({"maxFrameLength": self._max_frame_length}),
            "handshake": handshake,
        }
        self._on_state_change({"state": "connecting"})
        handlers = SimpleHandlers(
            on_data=lambda chunk: self._handle_data(identifier, chunk),
            on_close=lambda: self._handle_close() if self._is_current(identifier) else None,
            on_error=lambda error: self._fail_and_close(to_disconnected_error(error)) if self._is_current(identifier) else None,
        )
        await self._open_transport(identifier, handlers)
        return await handshake.promise

    def disconnect(self, reason: str | Exception = "Client disconnected") -> None:
        if self._lifecycle["state"] == "disconnected":
            return
        self._fail_and_close(PiDisconnectedError(reason) if isinstance(reason, str) else reason)

    def fail(self, error: Exception) -> None:
        self._fail_and_close(error)

    def send(self, frame: bytes) -> None:
        lifecycle = self._lifecycle
        if lifecycle["state"] != "connected":
            raise PiDisconnectedError()
        transport: ByteTransport = lifecycle["transport"]

        async def _send() -> None:
            try:
                await transport.send(frame)
            except Exception as error:
                current = self._lifecycle
                if current.get("state") != "disconnected" and current.get("transport") is transport:
                    self._fail_and_close(to_disconnected_error(error))

        try:
            import asyncio

            asyncio.get_event_loop().create_task(_send())
        except Exception as error:
            self._fail_and_close(to_disconnected_error(error))

    async def _open_transport(self, identifier: int, handlers: SimpleHandlers) -> None:
        try:
            transport = self._transport_factory(handlers)
            if hasattr(transport, "__await__"):
                transport = await transport
        except Exception as error:
            if self._is_current(identifier):
                self._fail(to_disconnected_error(error))
            return
        lifecycle = self._lifecycle
        if lifecycle.get("state") != "connecting" or lifecycle.get("id") != identifier:
            transport.close()
            return
        self._lifecycle = {**lifecycle, "transport": transport}
        try:
            await transport.send(
                encode_client_message({"type": "hello", "version": PROTOCOL_VERSION}, {"maxFrameLength": self._max_frame_length})
            )
        except Exception as error:
            if self._is_current(identifier):
                self._fail_and_close(to_disconnected_error(error))

    def _handle_data(self, identifier: int, chunk: bytes) -> None:
        lifecycle = self._lifecycle
        if lifecycle.get("state") == "disconnected" or lifecycle.get("id") != identifier:
            return
        if lifecycle.get("state") == "connecting" and not lifecycle.get("transport"):
            self._fail_and_close(ProtocolValidationError("Received server data before the client hello was sent"))
            return
        try:
            messages = lifecycle["decoder"].push(chunk)
        except Exception as error:
            self._fail_and_close(to_error(error))
            return
        for message in messages:
            if self._lifecycle.get("state") == "disconnected":
                return
            self._handle_message(message)

    def _handle_message(self, message: dict[str, Any]) -> None:
        lifecycle = self._lifecycle
        if lifecycle.get("state") == "connecting":
            if message.get("type") == "hello_error":
                self._fail_and_close(PiServerError(message["error"]))
                return
            if message.get("type") != "hello":
                self._fail_and_close(ProtocolValidationError("Expected server hello as first message"))
                return
            if not lifecycle.get("transport"):
                self._fail_and_close(ProtocolValidationError("Received server hello before the client hello was sent"))
                return
            connected = {
                "state": "connected",
                "id": lifecycle["id"],
                "decoder": lifecycle["decoder"],
                "transport": lifecycle["transport"],
                "handshake": lifecycle["handshake"],
            }
            self._lifecycle = connected
            try:
                self._on_handshake(message["snapshot"])
            except Exception as error:
                if self._lifecycle is connected:
                    self._fail_and_close(to_error(error))
                return
            if self._lifecycle is not connected:
                return
            self._on_state_change({"state": "connected"})
            if self._lifecycle is not connected:
                return
            self._lifecycle = {**connected, "handshake": None}
            lifecycle["handshake"].resolve(message["snapshot"])
            return
        if lifecycle.get("state") != "connected":
            return
        if message.get("type") in {"hello", "hello_error"}:
            self._fail_and_close(ProtocolValidationError("Unexpected handshake message"))
            return
        self._on_message(message)

    def _handle_close(self) -> None:
        lifecycle = self._lifecycle
        if lifecycle.get("state") == "disconnected":
            return
        error: Exception = PiDisconnectedError("Byte transport closed")
        try:
            lifecycle["decoder"].end()
        except Exception as decoder_error:
            error = to_error(decoder_error)
        self._fail(error)

    def _fail_and_close(self, error: Exception) -> None:
        lifecycle = self._lifecycle
        transport = None if lifecycle.get("state") == "disconnected" else lifecycle.get("transport")
        self._fail(error)
        if transport is not None:
            transport.close()

    def _fail(self, error: Exception) -> None:
        lifecycle = self._lifecycle
        if lifecycle.get("state") == "disconnected":
            return
        self._lifecycle = {"state": "disconnected"}
        handshake: PromiseResolvers | None = lifecycle.get("handshake")
        if handshake is not None:
            handshake.reject(error)
        self._on_state_change({"state": "disconnected", "error": error})

    def _is_current(self, identifier: int) -> bool:
        return self._lifecycle.get("state") != "disconnected" and self._lifecycle.get("id") == identifier
