from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any, Callable

from ..errors import PiServerError

TEST_MODEL = {
    "provider": "test",
    "id": "small",
    "name": "Test Small",
    "api": "test-api",
    "reasoning": True,
    "input": ["text", "image"],
    "contextWindow": 16_000,
    "maxTokens": 2_000,
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    "supportedThinkingLevels": ["off", "medium", "high"],
    "authenticated": True,
}


class Deferred:
    def __init__(self) -> None:
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()

    @property
    def promise(self):
        return self.future

    def resolve(self, value=None) -> None:
        if not self.future.done():
            self.future.set_result(value)


class TestSessionRuntime:
    def __init__(self, stored: dict[str, Any], on_dispose: Callable[[], None]) -> None:
        self.disposed = Deferred()
        self.dispose_count = 0
        self.steers: list[dict[str, Any]] = []
        self._stored = stored
        self._on_dispose = on_dispose
        self._listeners: set[Callable[[dict[str, Any]], None]] = set()
        self._pending_prompt: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._stored["snapshot"])

    def get_phase(self) -> str:
        return self._stored["snapshot"]["phase"]

    async def prompt(self, input: dict[str, Any]) -> None:
        if self.get_phase() != "idle":
            raise PiServerError("busy", "A prompt is already running")
        done = Deferred()
        self._pending_prompt = {"input": input, "done": done}
        self._update(
            {
                "phase": "turn",
                "transcript": [
                    *self._stored["snapshot"]["transcript"],
                    {
                        "id": f"user-{self._stored['snapshot']['revision'] + 1}",
                        "role": "user",
                        "content": [{"type": "text", "text": input["text"]}],
                        "timestamp": self._stored["snapshot"]["revision"] + 1,
                    },
                ],
            }
        )
        outcome = await done.promise
        assistant = (
            {
                "id": f"assistant-{self._stored['snapshot']['revision'] + 1}",
                "role": "assistant",
                "content": [{"type": "text", "text": f"reply:{input['text']}"}],
                "status": "complete",
                "model": self._stored["snapshot"]["model"],
                "stopReason": "stop",
                "timestamp": self._stored["snapshot"]["revision"] + 1,
            }
            if outcome == "complete"
            else {
                "id": f"assistant-{self._stored['snapshot']['revision'] + 1}",
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
                "status": "aborted",
                "model": self._stored["snapshot"]["model"],
                "stopReason": "aborted",
                "timestamp": self._stored["snapshot"]["revision"] + 1,
            }
        )
        self._update({"phase": "idle", "transcript": [*self._stored["snapshot"]["transcript"], assistant]})
        self._pending_prompt = None

    async def steer(self, input: dict[str, Any]) -> None:
        if self.get_phase() == "idle":
            raise PiServerError("busy", "There is no active prompt to steer")
        self.steers.append(input)
        self._update(
            {
                "queuedSteerCount": self._stored["snapshot"]["queuedSteerCount"] + 1,
                "queuedSteer": [
                    *self._stored["snapshot"]["queuedSteer"],
                    {
                        "id": f"steer-{self._stored['snapshot']['revision'] + 1}",
                        "role": "user",
                        "content": [{"type": "text", "text": input["text"]}],
                        "timestamp": self._stored["snapshot"]["revision"] + 1,
                    },
                ],
            }
        )

    async def abort(self) -> None:
        if self._pending_prompt is None:
            raise PiServerError("busy", "There is no active prompt to abort")
        self._pending_prompt["done"].resolve("aborted")

    async def set_model(self, model: dict[str, str]) -> None:
        if self.get_phase() != "idle":
            raise PiServerError("busy", "Session is busy")
        self._update({"model": model})

    async def set_thinking(self, thinking_level: str) -> None:
        if self.get_phase() != "idle":
            raise PiServerError("busy", "Session is busy")
        self._update({"thinkingLevel": thinking_level})

    def subscribe(self, listener: Callable[[dict[str, Any]], None]):
        self._listeners.add(listener)
        return lambda: self._listeners.discard(listener)

    async def dispose(self) -> None:
        self.dispose_count += 1
        self._on_dispose()
        self.disposed.resolve(None)

    def set_phase(self, phase: str) -> None:
        self._stored["snapshot"] = {**self._stored["snapshot"], "phase": phase}

    def finish_prompt(self) -> None:
        if self._pending_prompt is None:
            raise RuntimeError("No prompt is pending")
        self._pending_prompt["done"].resolve("complete")

    def emit_progress(self, progress: dict[str, Any]) -> None:
        for listener in list(self._listeners):
            listener({"type": "progress", "progress": progress})

    def emit_error(self, error: PiServerError) -> None:
        for listener in list(self._listeners):
            listener({"type": "error", "error": error})

    def emit_snapshot(self) -> None:
        for listener in list(self._listeners):
            listener({"type": "snapshot"})

    def _update(self, updates: dict[str, Any]) -> None:
        snapshot = self._stored["snapshot"]
        self._stored["snapshot"] = {
            **snapshot,
            **updates,
            "revision": snapshot["revision"] + 1,
            "updatedAt": snapshot["updatedAt"] + 1,
        }
        self.emit_snapshot()


class TestServerService:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.runtimes: dict[str, list[TestSessionRuntime]] = {}
        self.locked: set[str] = set()
        self.last_created_id: str | None = None
        self._next_list_delay = None

    async def list_sessions(self) -> list[dict[str, Any]]:
        delay = self._next_list_delay
        if delay is not None:
            self._next_list_delay = None
            delay["entered"].resolve(None)
            await delay["release"].promise
        return [
            {
                "id": stored["snapshot"]["id"],
                "createdAt": stored["snapshot"]["createdAt"],
                "updatedAt": stored["snapshot"]["updatedAt"],
                "sessionName": stored["snapshot"].get("name"),
                "cwd": stored["snapshot"].get("cwd"),
            }
            for stored in self.sessions.values()
        ]

    async def list_models(self) -> list[dict[str, Any]]:
        return [TEST_MODEL]

    async def create_session(self, options: dict[str, Any]) -> TestSessionRuntime:
        self.last_created_id = options["id"]
        if options["id"] in self.sessions:
            raise PiServerError("session_locked", "Session already exists")
        self.seed(options["id"], options.get("name"), options.get("cwd"), options.get("model"), options.get("thinkingLevel"))
        return self._acquire(options["id"])

    async def open_session(self, session_id: str) -> TestSessionRuntime:
        if session_id not in self.sessions:
            raise PiServerError("not_found", f"Unknown session: {session_id}")
        if session_id in self.locked:
            raise PiServerError("session_locked", f"Session is locked: {session_id}")
        return self._acquire(session_id)

    def seed(
        self,
        session_id: str = "session-1",
        name: str | None = None,
        cwd: str = "/tmp/pi-server-conformance",
        model: dict[str, str] | None = None,
        thinking_level: str = "off",
    ) -> None:
        self.sessions[session_id] = {
            "snapshot": {
                "id": session_id,
                "name": name or f"Session {session_id}",
                "cwd": cwd,
                "createdAt": 1,
                "updatedAt": 1,
                "phase": "idle",
                "model": model or {"provider": TEST_MODEL["provider"], "id": TEST_MODEL["id"]},
                "thinkingLevel": thinking_level,
                "attached": False,
                "locked": False,
                "revision": 0,
                "transcript": [],
                "queuedSteer": [],
                "queuedSteerCount": 0,
            }
        }

    def delay_next_list(self) -> dict[str, Deferred]:
        delay = {"entered": Deferred(), "release": Deferred()}
        self._next_list_delay = delay
        return delay

    def latest_runtime(self, session_id: str) -> TestSessionRuntime:
        runtimes = self.runtimes.get(session_id)
        if not runtimes:
            raise RuntimeError(f"No runtime for {session_id}")
        return runtimes[-1]

    def _acquire(self, session_id: str) -> TestSessionRuntime:
        stored = self.sessions[session_id]
        self.locked.add(session_id)
        runtime = TestSessionRuntime(stored, lambda: self.locked.discard(session_id))
        self.runtimes.setdefault(session_id, []).append(runtime)
        return runtime
