"""
Shared HTTP client pool — mirrors packages/coding-agent/src/core/http-dispatcher.ts

Uses httpx instead of undici.
"""
from __future__ import annotations

from typing import Any

import httpx

DEFAULT_HTTP_IDLE_TIMEOUT_MS = 300_000
DEFAULT_AUTO_SELECT_FAMILY_ATTEMPT_TIMEOUT_MS = 2_000

HTTP_IDLE_TIMEOUT_CHOICES = (
    {"label": "30 sec", "timeoutMs": 30_000},
    {"label": "1 min", "timeoutMs": 60_000},
    {"label": "2 min", "timeoutMs": 120_000},
    {"label": "5 min", "timeoutMs": 300_000},
    {"label": "disabled", "timeoutMs": 0},
)

_client: httpx.Client | None = None
_async_client: httpx.AsyncClient | None = None
_timeout_ms: int = DEFAULT_HTTP_IDLE_TIMEOUT_MS


def parse_http_idle_timeout_ms(value: Any) -> int | None:
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed.lower() == "disabled":
            return 0
        if not trimmed:
            return None
        try:
            return parse_http_idle_timeout_ms(float(trimmed))
        except ValueError:
            return None
    if not isinstance(value, (int, float)) or value < 0:
        return None
    return int(value)


def format_http_idle_timeout_ms(timeout_ms: int) -> str:
    for item in HTTP_IDLE_TIMEOUT_CHOICES:
        if item["timeoutMs"] == timeout_ms:
            return str(item["label"])
    return f"{timeout_ms / 1000} sec"


def apply_http_proxy_settings(http_proxy: str | None) -> None:
    proxy = (http_proxy or "").strip()
    if not proxy:
        return
    import os

    os.environ.setdefault("HTTP_PROXY", proxy)
    os.environ.setdefault("HTTPS_PROXY", proxy)


def _timeout_for(timeout_ms: int) -> httpx.Timeout | None:
    if timeout_ms <= 0:
        return None
    seconds = timeout_ms / 1000.0
    return httpx.Timeout(seconds, connect=min(seconds, 30.0))


def configure_http_dispatcher(timeout_ms: int = DEFAULT_HTTP_IDLE_TIMEOUT_MS) -> None:
    global _client, _async_client, _timeout_ms
    normalized = parse_http_idle_timeout_ms(timeout_ms)
    if normalized is None:
        raise ValueError(f"Invalid HTTP idle timeout: {timeout_ms}")
    _timeout_ms = normalized
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
    timeout = _timeout_for(normalized)
    if _client is not None:
        _client.close()
    if _async_client is not None:
        try:
            _async_client.close()
        except Exception:
            pass
    _client = httpx.Client(timeout=timeout, limits=limits, follow_redirects=True)
    _async_client = httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True)


def get_http_client() -> httpx.Client:
    global _client
    if _client is None:
        configure_http_dispatcher(_timeout_ms)
    assert _client is not None
    return _client


def get_async_http_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        configure_http_dispatcher(_timeout_ms)
    assert _async_client is not None
    return _async_client


def close_http_dispatcher() -> None:
    global _client, _async_client
    if _client is not None:
        _client.close()
        _client = None
    if _async_client is not None:
        try:
            _async_client.close()
        except Exception:
            pass
        _async_client = None
