from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypedDict

from .transport import ByteTransportFactory

ConnectionState = Literal["disconnected", "connecting", "connected"]


class ConnectionStateChange(TypedDict, total=False):
    state: ConnectionState
    error: Exception


Unsubscribe = Callable[[], None]
ListenerErrorHandler = Callable[[Exception], None]


class PiClientOptions(TypedDict, total=False):
    transportFactory: ByteTransportFactory
    transport_factory: ByteTransportFactory
    maxFrameLength: int
    max_frame_length: int
    onListenerError: ListenerErrorHandler
    on_listener_error: ListenerErrorHandler


class CreateSessionOptions(TypedDict, total=False):
    cwd: str
    name: str
    model: dict[str, str]
    thinkingLevel: str
    thinking_level: str
