from __future__ import annotations

import asyncio
import os
import socket
import sys

from pi_protocol import DEFAULT_MAX_FRAME_LENGTH

from .transport import ByteTransport, SimpleHandlers

MAX_UNIX_SOCKET_PATH_BYTES = 107 if sys.platform == "linux" else 103


class UnixByteTransport:
    def __init__(self, writer: asyncio.StreamWriter, max_pending_bytes: int, mark_local_close) -> None:
        self._writer = writer
        self._max_pending_bytes = max_pending_bytes
        self._mark_local_close = mark_local_close
        self._closed = False
        self._pending_bytes = 0
        self._write_lock = asyncio.Lock()

    async def send(self, chunk: bytes) -> None:
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("Unix transport chunks must be Uint8Array")
        data = bytes(chunk)
        if self._closed:
            raise RuntimeError("Unix transport is closed")
        if self._pending_bytes + len(data) > self._max_pending_bytes:
            raise RuntimeError("Unix transport exceeded its pending byte limit")
        self._pending_bytes += len(data)
        try:
            async with self._write_lock:
                if self._closed or self._writer.is_closing():
                    raise RuntimeError("Unix transport is closed")
                self._writer.write(data)
                await self._writer.drain()
        finally:
            self._pending_bytes -= len(data)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._mark_local_close()
        self._writer.close()


def create_unix_transport_factory(options: dict[str, object]):
    path = str(options["path"])
    if path == "":
        raise TypeError("Unix transport path must not be empty")
    if len(path.encode("utf-8")) > MAX_UNIX_SOCKET_PATH_BYTES:
        raise TypeError(f"Unix transport path is too long; maximum is {MAX_UNIX_SOCKET_PATH_BYTES} UTF-8 bytes")
    max_pending_bytes = int(options.get("maxPendingBytes") or options.get("max_pending_bytes") or DEFAULT_MAX_FRAME_LENGTH * 4)
    if max_pending_bytes <= 0:
        raise TypeError("Unix transport maxPendingBytes must be a positive safe integer")
    if os.name == "nt":
        raise RuntimeError("Unix transport is not supported on Windows")

    async def factory(handlers: SimpleHandlers) -> ByteTransport:
        try:
            reader, writer = await asyncio.open_unix_connection(path)
        except Exception as error:
            raise error

        closed = False

        def mark_local_close() -> None:
            nonlocal closed
            closed = True

        transport = UnixByteTransport(writer, max_pending_bytes, mark_local_close)

        async def pump() -> None:
            try:
                while True:
                    chunk = await reader.read(64 * 1024)
                    if not chunk:
                        if not closed:
                            handlers.on_close()
                        break
                    if not closed:
                        handlers.on_data(chunk)
            except Exception as error:
                if not closed:
                    handlers.on_error(error if isinstance(error, Exception) else Exception(str(error)))

        asyncio.get_event_loop().create_task(pump())
        return transport

    return factory


# Keep the unused import for API parity with Node net sockets on platforms that need it.
_ = socket
