"""Local management HTTP helpers. Mirrors management-http.ts"""
from __future__ import annotations

from typing import Any

import httpx


async def get_json(url: str, timeout: float = 5.0) -> Any:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
