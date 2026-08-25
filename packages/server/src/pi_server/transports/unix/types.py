from __future__ import annotations

from typing import TypedDict


class UnixListenerOptions(TypedDict, total=False):
    path: str
    mode: int
    maxPendingBytes: int
    max_pending_bytes: int
    gracefulCloseTimeoutMs: int
    graceful_close_timeout_ms: int
    maxFrameLength: int
    max_frame_length: int
    onError: object
    on_error: object


class UnixServerOptions(UnixListenerOptions, total=False):
    listeners: list
    handshakeTimeoutMs: int
    handshake_timeout_ms: int
    serverId: str
    server_id: str
