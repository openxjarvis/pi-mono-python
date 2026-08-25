from __future__ import annotations

import asyncio
import hashlib
import os
import stat
import sys
import uuid
from pathlib import Path

from pi_protocol import DEFAULT_MAX_FRAME_LENGTH

from ...connection import ByteConnectionAcceptor
from ...listener import PiServerListener

DEFAULT_SOCKET_MODE = 0o600
DEFAULT_GRACEFUL_CLOSE_TIMEOUT_MS = 5_000
MAX_UNIX_SOCKET_PATH_BYTES = 107 if sys.platform == "linux" else 103


def validate_unix_socket_path(path: str, description: str = "Unix socket path") -> None:
    if not path:
        raise TypeError(f"{description} must not be empty")
    if len(path.encode("utf-8")) > MAX_UNIX_SOCKET_PATH_BYTES:
        raise TypeError(f"{description} is too long; maximum is {MAX_UNIX_SOCKET_PATH_BYTES} UTF-8 bytes")


class UnixByteConnection:
    def __init__(self, writer: asyncio.StreamWriter, graceful_close_timeout_ms: int, max_pending_bytes: int) -> None:
        self._writer = writer
        self._graceful_close_timeout_ms = graceful_close_timeout_ms
        self._max_pending_bytes = max_pending_bytes
        self._pending_bytes = 0
        self.closed = False
        self._closing = False
        self._lock = asyncio.Lock()

    async def send(self, chunk: bytes) -> None:
        data = bytes(chunk)
        if self.closed or self._closing:
            raise RuntimeError("Unix connection is closed")
        if self._pending_bytes + len(data) > self._max_pending_bytes:
            raise RuntimeError("Unix connection exceeded its pending byte limit")
        self._pending_bytes += len(data)
        try:
            async with self._lock:
                if self.closed or self._closing or self._writer.is_closing():
                    raise RuntimeError("Unix connection is closed")
                self._writer.write(data)
                await self._writer.drain()
        finally:
            self._pending_bytes -= len(data)

    async def close(self, final_chunk: bytes | None = None) -> None:
        if self.closed or self._writer.is_closing():
            self.mark_closed()
            return
        if self._closing:
            return
        self._closing = True
        try:
            if final_chunk:
                self._writer.write(bytes(final_chunk))
                await self._writer.drain()
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:
            self._writer.close()
        self.mark_closed()

    def mark_closed(self) -> None:
        self.closed = True
        self._closing = True


class UnixListener:
    def __init__(self, options: dict[str, object]) -> None:
        path = str(options["path"])
        validate_unix_socket_path(path, "PiServer Unix socket path")
        self._path = path
        self._mode = int(options.get("mode", DEFAULT_SOCKET_MODE))
        max_frame = int(options.get("maxFrameLength") or options.get("max_frame_length") or DEFAULT_MAX_FRAME_LENGTH)
        self._max_pending_bytes = int(options.get("maxPendingBytes") or options.get("max_pending_bytes") or max_frame * 4)
        self._graceful_close_timeout_ms = int(
            options.get("gracefulCloseTimeoutMs") or options.get("graceful_close_timeout_ms") or DEFAULT_GRACEFUL_CLOSE_TIMEOUT_MS
        )
        self._on_error = options.get("onError") or options.get("on_error")
        self.address: str | None = None
        self._server = None
        self._accept: ByteConnectionAcceptor | None = None
        self._connections: set[UnixByteConnection] = set()
        self._closing = False

    async def start(self, accept: ByteConnectionAcceptor) -> None:
        if self._server is not None:
            raise RuntimeError("Unix listener is already started")
        if self._closing:
            raise RuntimeError("Unix listener is closing or closed")
        self._accept = accept
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(self._path):
            os.unlink(self._path)

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            if self._closing or self._accept is None:
                writer.close()
                return
            connection = UnixByteConnection(writer, self._graceful_close_timeout_ms, self._max_pending_bytes)
            self._connections.add(connection)
            handler = self._accept(connection)

            async def pump() -> None:
                try:
                    while True:
                        chunk = await reader.read(64 * 1024)
                        if not chunk:
                            break
                        handler.on_data(chunk)
                except Exception as error:
                    handler.on_error(error if isinstance(error, Exception) else Exception(str(error)))
                finally:
                    connection.mark_closed()
                    self._connections.discard(connection)
                    handler.on_close()

            asyncio.get_event_loop().create_task(pump())

        self._server = await asyncio.start_unix_server(handle, path=self._path)
        try:
            os.chmod(self._path, self._mode)
        except OSError:
            pass
        self.address = self._path

    async def close(self) -> None:
        self._closing = True
        self.address = None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for connection in list(self._connections):
            await connection.close()
        if os.path.exists(self._path):
            try:
                os.unlink(self._path)
            except OSError:
                pass
        self._server = None


def create_unix_listener(options: dict[str, object]) -> PiServerListener:
    return UnixListener(options)
