from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from .errors import PiServerError
from .listener import PiServerListener


class PiServerOptions(dict):
    pass


MaybePromise = Any
PromptInput = dict[str, Any]
SteerInput = dict[str, Any]
CreateSessionOptions = dict[str, Any]
PiSessionRuntimeEvent = dict[str, Any]


class PiSessionRuntime(Protocol):
    def snapshot(self) -> dict[str, Any] | Awaitable[dict[str, Any]]: ...
    def get_phase(self) -> str: ...
    async def prompt(self, input: PromptInput) -> None: ...
    async def steer(self, input: SteerInput) -> None: ...
    async def abort(self) -> None: ...
    async def set_model(self, model: dict[str, str]) -> None: ...
    async def set_thinking(self, thinking_level: str) -> None: ...
    def subscribe(self, listener: Callable[[PiSessionRuntimeEvent], None]) -> Callable[[], None]: ...
    async def dispose(self) -> None: ...


class PiServerService(Protocol):
    async def list_sessions(self) -> list[dict[str, Any]]: ...
    async def list_models(self) -> list[dict[str, Any]]: ...
    async def create_session(self, options: CreateSessionOptions) -> PiSessionRuntime: ...
    async def open_session(self, session_id: str) -> PiSessionRuntime: ...


SessionRuntime = PiSessionRuntime
SessionRuntimeEvent = PiSessionRuntimeEvent

# Re-export so type checkers see the listener dependency.
_ = (PiServerListener, PiServerError)
