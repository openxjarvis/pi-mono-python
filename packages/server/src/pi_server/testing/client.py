from __future__ import annotations

import asyncio
from typing import Any, Callable

from pi_protocol import PROTOCOL_VERSION, ServerMessageDecoder, encode_client_message

from .service import Deferred


class ProtocolTestClient:
    def __init__(self, channel: dict[str, Any]) -> None:
        self.messages: list[dict[str, Any]] = []
        self._channel = channel
        self._decoder = ServerMessageDecoder()
        self._waiters: list[dict[str, Any]] = []
        self._closed_deferred = Deferred()
        self._request_sequence = 0
        self.closed = False

    async def hello(self, version: int = PROTOCOL_VERSION) -> dict[str, Any]:
        response = self.next(lambda message: message.get("type") in {"hello", "hello_error"})
        await self.send_message({"type": "hello", "version": version})
        return await response

    async def request(self, command: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        self._request_sequence += 1
        identifier = request_id or f"request-{self._request_sequence}"
        response = self.next(lambda message: message.get("type") == "response" and message.get("id") == identifier)
        await self.send_message({"type": "request", "id": identifier, "request": command})
        return await response

    async def send_message(self, message: dict[str, Any]) -> None:
        await self._channel["send"](encode_client_message(message))

    async def send_bytes(self, chunk: bytes) -> None:
        await self._channel["send"](chunk)

    def next(self, predicate: Callable[[dict[str, Any]], bool]):
        return self.next_from(0, predicate)

    def next_from(self, index: int, predicate: Callable[[dict[str, Any]], bool]):
        existing = next((message for message in self.messages[index:] if predicate(message)), None)
        if existing is not None:
            future = asyncio.get_event_loop().create_future()
            future.set_result(existing)
            return future
        if self.closed:
            future = asyncio.get_event_loop().create_future()
            future.set_exception(RuntimeError("Wire client is closed"))
            return future
        future = asyncio.get_event_loop().create_future()
        self._waiters.append({"predicate": predicate, "future": future})
        return future

    def wait_for_close(self):
        return asyncio.sleep(0) if self.closed else self._closed_deferred.promise

    async def close(self) -> None:
        await self._channel["close"]()

    def receive(self, chunk: bytes) -> None:
        try:
            for message in self._decoder.push(chunk):
                self.messages.append(message)
                for waiter in list(self._waiters):
                    if waiter["predicate"](message):
                        self._waiters.remove(waiter)
                        if not waiter["future"].done():
                            waiter["future"].set_result(message)
        except Exception as error:
            self.fail(error if isinstance(error, Exception) else Exception(str(error)))

    def mark_closed(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._closed_deferred.resolve(None)
        self.fail(RuntimeError("Wire connection closed"))

    def fail(self, error: Exception) -> None:
        for waiter in self._waiters:
            if not waiter["future"].done():
                waiter["future"].set_exception(error)
        self._waiters.clear()


async def connect_unix_test_client(path: str) -> ProtocolTestClient:
    reader, writer = await asyncio.open_unix_connection(path)

    async def send(chunk: bytes) -> None:
        writer.write(bytes(chunk))
        await writer.drain()

    async def send_fragmented(chunk: bytes, split_at: int) -> None:
        await send(chunk[:split_at])
        await send(chunk[split_at:])

    async def close() -> None:
        writer.close()
        await writer.wait_closed()

    client = ProtocolTestClient({"send": send, "sendFragmented": send_fragmented, "close": close})

    async def pump() -> None:
        try:
            while True:
                chunk = await reader.read(64 * 1024)
                if not chunk:
                    break
                client.receive(chunk)
        except Exception as error:
            client.fail(error if isinstance(error, Exception) else Exception(str(error)))
        finally:
            client.mark_closed()

    asyncio.get_event_loop().create_task(pump())
    return client
