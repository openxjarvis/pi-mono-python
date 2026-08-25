from __future__ import annotations

from typing import Any

from ..server import PiServer
from .service import TestServerService


def create_test_server(options: dict[str, Any]) -> dict[str, Any]:
    service = options.get("service") or TestServerService()
    return {
        "server": PiServer(
            service,
            {
                "listeners": options["listeners"],
                "maxFrameLength": options.get("maxFrameLength", options.get("max_frame_length")),
                "handshakeTimeoutMs": options.get("handshakeTimeoutMs", options.get("handshake_timeout_ms")),
                "serverId": options.get("serverId", options.get("server_id")),
                "onError": options.get("onError", options.get("on_error")),
            },
        ),
        "service": service,
    }
