from __future__ import annotations

from typing import Any

from ...server import PiServer
from .listener import create_unix_listener


def create_unix_server(service: Any, options: dict[str, Any]) -> PiServer:
    listener = create_unix_listener(options)
    return PiServer(
        service,
        {
            "listeners": [listener],
            "maxFrameLength": options.get("maxFrameLength", options.get("max_frame_length")),
            "handshakeTimeoutMs": options.get("handshakeTimeoutMs", options.get("handshake_timeout_ms")),
            "serverId": options.get("serverId", options.get("server_id")),
            "onError": options.get("onError", options.get("on_error")),
        },
    )
