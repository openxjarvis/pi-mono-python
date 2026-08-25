"""
Unix-socket experimental server — mirrors packages/coding-agent/src/cli/experimental/commands/server.ts

Listens with ``socket.AF_UNIX`` and speaks JSON-line RPC.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import unquote, urlparse

CommandHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]


@dataclass(frozen=True)
class UnixTransportAddress:
    transport: str
    path: str


def parse_unix_listen_address(value: str, option: str = "--listen") -> UnixTransportAddress:
    try:
        url = urlparse(value)
    except ValueError as exc:
        raise ValueError(f'Invalid {option} address "{value}"') from exc
    if url.scheme != "unix":
        raise ValueError(f'Unsupported {option} transport "{url.scheme}:"')
    if url.hostname or url.port or url.username or url.password:
        raise ValueError("Unix transport address must not include an authority")
    if (
        not value.startswith("unix:///")
        or value.startswith("unix:////")
        or "?" in value
        or "#" in value
    ):
        raise ValueError(f'Invalid {option} address "{value}"')
    path = unquote(url.path)
    if "\0" in path:
        raise ValueError(f'Invalid {option} address "{value}"')
    if not os.path.isabs(path):
        raise ValueError("Unix transport address requires an absolute path")
    return UnixTransportAddress(transport="unix", path=path)


class UnixSocketServer:
    """Accept JSON-line commands over an AF_UNIX socket."""

    def __init__(
        self,
        path: str,
        handler: CommandHandler | None = None,
    ) -> None:
        self.path = path
        self.handler = handler
        self._server: asyncio.AbstractServer | None = None
        self._sock: socket.socket | None = None

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    command = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    writer.write(json.dumps({"success": False, "error": f"parse error: {exc}"}).encode() + b"\n")
                    await writer.drain()
                    continue
                if self.handler is None:
                    response: dict[str, Any] = {"success": True, "data": command}
                else:
                    result = self.handler(command)
                    response = await result if isinstance(result, Awaitable) else result
                writer.write(json.dumps(response).encode("utf-8") + b"\n")
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def start(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if os.path.exists(self.path):
            os.unlink(self.path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.path)
        sock.listen(16)
        sock.setblocking(False)
        self._sock = sock
        self._server = await asyncio.start_unix_server(self._handle_client, sock=sock)

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except OSError:
                pass


async def run_unix_socket_server(
    path: str,
    handler: CommandHandler | None = None,
) -> UnixSocketServer:
    server = UnixSocketServer(path, handler)
    await server.start()
    return server
