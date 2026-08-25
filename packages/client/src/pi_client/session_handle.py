from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol

from .types import Unsubscribe

SessionLeaseMode = Literal["shared", "exclusive"]


class AcquireSessionOptions(dict):
    pass


class SessionHandleCallbacks(Protocol):
    def is_attached(self) -> bool: ...
    def get_snapshot(self) -> dict[str, Any] | None: ...
    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> Unsubscribe: ...
    def on_event(self, listener: Callable[[dict[str, Any]], None]) -> Unsubscribe: ...
    def detach(self) -> Awaitable[None]: ...
    def dispose(self) -> Awaitable[None]: ...
    def request(self, command: dict[str, Any]) -> Awaitable[dict[str, Any]]: ...


class SessionHandle:
    def __init__(self, session_id: str, callbacks: SessionHandleCallbacks) -> None:
        self.id = session_id
        self._callbacks = callbacks

    @property
    def attached(self) -> bool:
        return self._callbacks.is_attached()

    @property
    def active(self) -> bool:
        return self.attached

    @property
    def snapshot(self) -> dict[str, Any] | None:
        return self._callbacks.get_snapshot()

    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> Unsubscribe:
        return self._callbacks.subscribe(listener)

    def on_event(self, listener: Callable[[dict[str, Any]], None]) -> Unsubscribe:
        return self._callbacks.on_event(listener)

    async def detach(self) -> None:
        await self._callbacks.detach()

    async def dispose(self) -> None:
        await self._callbacks.dispose()

    async def __aenter__(self) -> SessionHandle:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.dispose()

    async def prompt(self, text: str) -> dict[str, Any]:
        return (await self._request({"command": "prompt", "sessionId": self.id, "text": text}))["session"]

    async def steer(self, text: str) -> dict[str, Any]:
        return (await self._request({"command": "steer", "sessionId": self.id, "text": text}))["session"]

    async def abort(self) -> dict[str, Any]:
        return (await self._request({"command": "abort", "sessionId": self.id}))["session"]

    async def set_model(self, model: dict[str, str]) -> dict[str, Any]:
        return (await self._request({"command": "set_model", "sessionId": self.id, "model": model}))["session"]

    async def set_thinking(self, thinking_level: str) -> dict[str, Any]:
        return (
            await self._request({"command": "set_thinking", "sessionId": self.id, "thinkingLevel": thinking_level})
        )["session"]

    def _request(self, command: dict[str, Any]):
        return self._callbacks.request(command)


PiSessionHandle = SessionHandle
SessionLease = SessionHandle
