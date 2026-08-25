from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol

from pi_protocol import ClientMessageDecoder

MaybePromise = Any
ConnectionStage = Literal["awaitingHello", "handshaking", "ready", "closing", "closed"]


class ByteConnection(Protocol):
    closed: bool

    async def send(self, chunk: bytes) -> None: ...
    def close(self, final_chunk: bytes | None = None) -> MaybePromise: ...


class ByteConnectionHandler(Protocol):
    def on_data(self, chunk: bytes) -> None: ...
    def on_close(self) -> None: ...
    def on_error(self, error: Exception) -> None: ...


ByteConnectionAcceptor = Callable[[ByteConnection], ByteConnectionHandler]


class ConnectionState:
    def __init__(
        self,
        *,
        id: str,
        connection: ByteConnection,
        decoder: ClientMessageDecoder,
        handshake_timeout: Any,
    ) -> None:
        self.id = id
        self.connection = connection
        self.decoder = decoder
        self.session_ids: set[str] = set()
        self.stage: ConnectionStage = "awaitingHello"
        self.disconnected = False
        self.handshake_complete = False
        self.handshake: Awaitable[None] | None = None
        self.handshake_timeout = handshake_timeout


def is_terminal_connection(state: ConnectionState) -> bool:
    return state.disconnected or state.stage in {"closing", "closed"}
