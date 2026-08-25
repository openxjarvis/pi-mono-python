from __future__ import annotations

from typing import Any, Callable

from pi_protocol import PROTOCOL_VERSION


class ServerSnapshotPublisher:
    def __init__(self, options: dict[str, Any]) -> None:
        self._options = options
        self._revision = 0
        self._broadcast_queue: Any = None

    @property
    def current_revision(self) -> int:
        return self._revision

    async def get(self, models: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "serverId": self._options["server_id"],
            "protocolVersion": PROTOCOL_VERSION,
            "revision": self._revision,
            "sessions": await self._options["list_sessions"](),
            "models": models if models is not None else await self._options["service"].list_models(),
        }

    async def broadcast(self) -> None:
        await self._perform_broadcast()

    async def _perform_broadcast(self) -> None:
        ready = [
            connection
            for connection in list(self._options["connections"])
            if connection.stage == "ready" and not connection.disconnected
        ]
        if not ready or self._options["is_closing"]():
            return
        self._revision += 1
        revision = self._revision
        models = await self._options["service"].list_models()
        current = await self.get(models)
        snapshot = {**current, "revision": revision}
        envelope = {"type": "event", "event": {"type": "server_snapshot", "snapshot": snapshot}}
        for connection in ready:
            await self._options["send_message"](connection, envelope)
