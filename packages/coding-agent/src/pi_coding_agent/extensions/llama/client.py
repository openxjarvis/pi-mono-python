"""llama.cpp HTTP client. Mirrors packages/coding-agent/src/extensions/llama/client.ts"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urljoin

import httpx

LlamaModelStatusValue = Literal["unloaded", "loading", "loaded", "downloading", "sleeping", "unknown"]


def normalize_llama_server_url(url: str) -> str:
    return url.rstrip("/")


def llama_inference_url(server_url: str) -> str:
    return f"{normalize_llama_server_url(server_url)}/v1"


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f" {n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


@dataclass
class LlamaModelStatus:
    value: LlamaModelStatusValue = "unknown"
    failed: bool = False
    args: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LlamaProgress:
    message: str = ""
    ratio: float | None = None
    detail: str | None = None


@dataclass
class LlamaModelInfo:
    id: str
    status: LlamaModelStatus
    extra: dict[str, Any] | None = None
    aliases: list[str] = field(default_factory=list)
    architecture: dict[str, Any] | None = None
    source: str | None = None
    meta: dict[str, Any] | None = None


class LlamaClient:
    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self.base_url = normalize_llama_server_url(base_url)
        self.api_key = api_key

    @property
    def server_url(self) -> str:
        return self.base_url

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _model_from_payload(self, item: dict[str, Any]) -> LlamaModelInfo:
        raw_status = item.get("status")
        if isinstance(raw_status, dict):
            status = LlamaModelStatus(
                value=str(raw_status.get("value") or "unknown"),  # type: ignore[arg-type]
                failed=bool(raw_status.get("failed")),
                args=list(raw_status.get("args") or []),
                extra=raw_status,
            )
        else:
            status = LlamaModelStatus(value=str(raw_status or item.get("state") or "unknown"))  # type: ignore[arg-type]
        return LlamaModelInfo(
            id=item.get("id") or item.get("name", ""),
            status=status,
            extra=item,
            aliases=list(item.get("aliases") or []),
            architecture=item.get("architecture") if isinstance(item.get("architecture"), dict) else None,
            source=item.get("source"),
            meta=item.get("meta") if isinstance(item.get("meta"), dict) else None,
        )

    async def list_models(self) -> list[LlamaModelInfo]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(urljoin(self.base_url + "/", "v1/models"), headers=self._headers())
            response.raise_for_status()
            data = response.json()
        models = data.get("data") or data.get("models") or []
        return [self._model_from_payload(item) for item in models if isinstance(item, dict)]

    async def list(self) -> list[LlamaModelInfo]:
        return await self.list_models()

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(urljoin(self.base_url + "/", "health"), headers=self._headers())
            return response.status_code < 500
        except Exception:
            return False
