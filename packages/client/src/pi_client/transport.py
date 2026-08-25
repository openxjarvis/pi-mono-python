from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol


class ByteTransport(Protocol):
    async def send(self, chunk: bytes) -> None: ...
    def close(self) -> None: ...


class ByteTransportHandlers(Protocol):
    def on_data(self, chunk: bytes) -> None: ...
    def on_close(self) -> None: ...
    def on_error(self, error: Exception) -> None: ...


class SimpleHandlers:
    def __init__(
        self,
        on_data: Callable[[bytes], None],
        on_close: Callable[[], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self.on_data = on_data
        self.on_close = on_close
        self.on_error = on_error


ByteTransportFactory = Callable[[SimpleHandlers], ByteTransport | Awaitable[ByteTransport]]
