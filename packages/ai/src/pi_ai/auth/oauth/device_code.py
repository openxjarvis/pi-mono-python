"""
OAuth device-code polling (RFC 8628).
Mirrors packages/ai/src/auth/oauth/device-code.ts
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx


async def poll_device_code_token(
    token_url: str,
    *,
    client_id: str,
    device_code: str,
    interval_seconds: float = 5,
    expires_in_seconds: float = 900,
    extra: dict[str, Any] | None = None,
    cancel_event: asyncio.Event | None = None,
) -> dict[str, Any]:
    deadline = time.time() + expires_in_seconds
    interval = max(1.0, interval_seconds)
    payload = {
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        "device_code": device_code,
        "client_id": client_id,
        **(extra or {}),
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError("The operation was aborted")
            if time.time() >= deadline:
                raise TimeoutError("Device-code login timed out")
            response = await client.post(token_url, data=payload)
            data = response.json() if response.content else {}
            if response.is_success and data.get("access_token"):
                return data
            error = data.get("error")
            if error == "authorization_pending":
                await asyncio.sleep(interval)
                continue
            if error == "slow_down":
                interval += 5
                await asyncio.sleep(interval)
                continue
            raise RuntimeError(data.get("error_description") or error or f"HTTP {response.status_code}")
