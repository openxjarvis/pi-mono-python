from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from .connection import ByteConnectionAcceptor


class PiServerListener(Protocol):
    address: str | None

    async def start(self, accept: ByteConnectionAcceptor) -> None: ...
    async def close(self) -> None: ...
