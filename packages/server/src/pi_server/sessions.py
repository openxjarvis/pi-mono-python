from __future__ import annotations

import uuid
from typing import Any, Callable

from .errors import PiServerError


def _to_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": snapshot["id"],
        "createdAt": snapshot["createdAt"],
        "updatedAt": snapshot.get("updatedAt"),
        "sessionName": snapshot.get("name"),
        "cwd": snapshot.get("cwd"),
    }


class LiveSession:
    def __init__(self, session_id: str, runtime: Any) -> None:
        self.id = session_id
        self.runtime = runtime
        self.connections: set[Any] = set()
        self.unsubscribe: Callable[[], None] = lambda: None
        self.operation_count = 0
        self.ready = False
        self.terminal = False
        self.disposing = None


class LiveSessionManager:
    def __init__(self, options: dict[str, Any]) -> None:
        self._options = options
        self._live_sessions: dict[str, LiveSession] = {}
        self._opening_sessions: dict[str, Any] = {}

    async def execute_command(self, connection: Any, command: dict[str, Any]) -> dict[str, Any]:
        name = command["command"]
        if name == "list":
            return {"command": "list", "sessions": await self.list_metadata()}
        if name == "create":
            session_id = str(uuid.uuid4())
            options = {
                "id": session_id,
                "cwd": command.get("cwd"),
                "name": command.get("name"),
                "model": command.get("model"),
                "thinkingLevel": command.get("thinkingLevel"),
            }
            live = await self._acquire(session_id, lambda: self._options["service"].create_session(options))
            await self._attach(connection, live)
            session = self._for_connection(await self._broadcast_snapshot(live), connection)
            self._options["broadcast_server_snapshot"]()
            return {"command": "create", "session": session}
        if name == "attach":
            live = await self._acquire(command["sessionId"], lambda: self._options["service"].open_session(command["sessionId"]))
            await self._attach(connection, live)
            session = self._for_connection(await self._broadcast_snapshot(live), connection)
            self._options["broadcast_server_snapshot"]()
            return {"command": "attach", "session": session}
        if name == "detach":
            live = self._live_sessions.get(command["sessionId"])
            if command["sessionId"] in connection.session_ids:
                connection.session_ids.discard(command["sessionId"])
                if live is not None:
                    live.connections.discard(connection)
                    if live.connections and not live.terminal and live.disposing is None:
                        await self._broadcast_snapshot(live)
                    await self._maybe_dispose(live)
                self._options["broadcast_server_snapshot"]()
            return {"command": "detach", "sessionId": command["sessionId"]}
        live = self._require_attached(connection, command["sessionId"])
        if name == "prompt":
            session = await self._run_operation(connection, live, lambda: live.runtime.prompt({"text": command["text"]}))
            return {"command": "prompt", "session": session}
        if name == "steer":
            session = await self._run_operation(connection, live, lambda: live.runtime.steer({"text": command["text"]}))
            return {"command": "steer", "session": session}
        if name == "abort":
            session = await self._run_operation(connection, live, live.runtime.abort)
            return {"command": "abort", "session": session}
        if name == "set_model":
            session = await self._run_operation(connection, live, lambda: live.runtime.set_model(command["model"]))
            return {"command": "set_model", "session": session}
        if name == "set_thinking":
            session = await self._run_operation(
                connection, live, lambda: live.runtime.set_thinking(command["thinkingLevel"])
            )
            return {"command": "set_thinking", "session": session}
        raise PiServerError("invalid_request", f"Unknown command: {name}")

    async def disconnect(self, connection: Any) -> None:
        sessions = [self._live_sessions[session_id] for session_id in list(connection.session_ids) if session_id in self._live_sessions]
        connection.session_ids.clear()
        for live in sessions:
            live.connections.discard(connection)
        for live in sessions:
            try:
                await self._maybe_dispose(live)
            except Exception as error:
                self._options["report_error"](error)

    async def list_metadata(self) -> list[dict[str, Any]]:
        stored = await self._options["service"].list_sessions()
        live_snapshots = []
        for live in list(self._live_sessions.values()):
            if live.disposing is None:
                live_snapshots.append((live.id, await self._normalized_snapshot(live)))
        live_by_id = dict(live_snapshots)
        metadata = []
        for item in stored:
            snapshot = live_by_id.pop(item["id"], None)
            metadata.append({**item, **_to_metadata(snapshot)} if snapshot else item)
        for snapshot in live_by_id.values():
            metadata.append(_to_metadata(snapshot))
        return metadata

    async def close(self) -> None:
        opening = list(self._opening_sessions.values())
        for pending in opening:
            try:
                await pending
            except Exception as error:
                self._options["report_error"](error)
        sessions = list(self._live_sessions.values())
        self._live_sessions.clear()
        for live in sessions:
            if live.disposing is not None:
                await live.disposing
                continue
            live.unsubscribe()
            await live.runtime.dispose()

    async def _run_operation(self, connection: Any, live: LiveSession, operation) -> dict[str, Any]:
        live.operation_count += 1
        try:
            await operation()
            return self._for_connection(await self._broadcast_snapshot(live), connection)
        finally:
            live.operation_count -= 1
            self._schedule_maybe_dispose(live)

    async def _acquire(self, session_id: str, acquire_runtime) -> LiveSession:
        while True:
            existing = self._live_sessions.get(session_id)
            if existing is not None:
                if existing.terminal:
                    raise PiServerError("session_locked", f"Session runtime is terminating: {session_id}")
                if existing.disposing is not None:
                    await existing.disposing
                    continue
                return existing
            opening = self._opening_sessions.get(session_id)
            if opening is not None:
                return await opening
            pending = self._create(session_id, acquire_runtime)
            self._opening_sessions[session_id] = pending
            try:
                return await pending
            finally:
                if self._opening_sessions.get(session_id) is pending:
                    self._opening_sessions.pop(session_id, None)

    async def _create(self, session_id: str, acquire_runtime) -> LiveSession:
        runtime = await acquire_runtime()
        if self._options["is_closing"]():
            await runtime.dispose()
            raise RuntimeError("PiServer closed while acquiring a session runtime")
        live: LiveSession | None = None
        try:
            snapshot = runtime.snapshot()
            if hasattr(snapshot, "__await__"):
                snapshot = await snapshot
            if snapshot["id"] != session_id:
                raise PiServerError(
                    "invalid_request",
                    f"Service returned session {snapshot['id']} for server-assigned session {session_id}",
                )
            live = LiveSession(session_id, runtime)
            live.unsubscribe = runtime.subscribe(lambda event, current=live: self._handle_runtime_event(current, event))
            self._live_sessions[session_id] = live
            live.ready = True
            return live
        except Exception:
            if live is not None:
                live.unsubscribe()
            try:
                await runtime.dispose()
            except Exception as dispose_error:
                self._options["report_error"](dispose_error)
            raise

    def _handle_runtime_event(self, live: LiveSession, event: dict[str, Any]) -> None:
        import asyncio

        if event.get("type") == "error":
            asyncio.get_event_loop().create_task(self._terminate(live, event["error"]))
            return
        if event.get("type") == "progress":
            envelope = {
                "type": "event",
                "event": {"type": "session_progress", "sessionId": live.id, "progress": event["progress"]},
            }
            for connection in list(live.connections):
                asyncio.get_event_loop().create_task(self._options["send_message"](connection, envelope))
        else:
            asyncio.get_event_loop().create_task(self._broadcast_snapshot(live))
        self._schedule_maybe_dispose(live)

    async def _terminate(self, live: LiveSession, error: PiServerError) -> None:
        if live.terminal:
            return
        live.terminal = True
        self._options["report_error"](error)
        live.unsubscribe()
        connections = list(live.connections)
        for connection in connections:
            await self._options["close_connection"](connection.connection)
        for connection in connections:
            await self._options["disconnect"](connection)
        await self._maybe_dispose(live)

    async def _normalized_snapshot(self, live: LiveSession) -> dict[str, Any]:
        snapshot = live.runtime.snapshot()
        if hasattr(snapshot, "__await__"):
            snapshot = await snapshot
        if snapshot["id"] != live.id:
            raise PiServerError("invalid_request", f"Runtime session ID changed from {live.id} to {snapshot['id']}")
        return {**snapshot, "phase": live.runtime.get_phase(), "attached": len(live.connections) > 0, "locked": True}

    def _for_connection(self, snapshot: dict[str, Any], connection: Any) -> dict[str, Any]:
        return {**snapshot, "attached": snapshot["id"] in connection.session_ids}

    async def _broadcast_snapshot(self, live: LiveSession) -> dict[str, Any]:
        snapshot = await self._normalized_snapshot(live)
        envelope = {"type": "event", "event": {"type": "session_snapshot", "snapshot": snapshot}}
        for connection in list(live.connections):
            await self._options["send_message"](connection, envelope)
        return snapshot

    async def _attach(self, connection: Any, live: LiveSession) -> None:
        if connection.disconnected or connection.stage != "ready" or connection.connection.closed:
            await self._maybe_dispose(live)
            raise PiServerError("invalid_request", "Connection closed while attaching to a session")
        connection.session_ids.add(live.id)
        live.connections.add(connection)

    def _require_attached(self, connection: Any, session_id: str) -> LiveSession:
        if session_id not in connection.session_ids:
            raise PiServerError("invalid_request", f"Connection is not attached to session {session_id}")
        live = self._live_sessions.get(session_id)
        if live is None or live.terminal or live.disposing is not None:
            raise PiServerError("not_found", f"Session is not live: {session_id}")
        return live

    def _schedule_maybe_dispose(self, live: LiveSession) -> None:
        import asyncio

        asyncio.get_event_loop().create_task(self._maybe_dispose(live))

    async def _maybe_dispose(self, live: LiveSession) -> Any:
        phase = live.runtime.get_phase()
        if (
            self._options["is_closing"]()
            or not live.ready
            or live.disposing is not None
            or live.connections
            or live.operation_count > 0
            or (not live.terminal and phase != "idle")
        ):
            return live.disposing
        live.unsubscribe()

        async def _dispose() -> None:
            try:
                await live.runtime.dispose()
            finally:
                if self._live_sessions.get(live.id) is live:
                    self._live_sessions.pop(live.id, None)

        live.disposing = _dispose()
        await live.disposing
        if not self._options["is_closing"]():
            self._options["broadcast_server_snapshot"]()
        return None
