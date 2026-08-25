"""
Hugging Face helpers — mirrors packages/coding-agent/src/extensions/llama/huggingface.ts
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DEFAULT_HUGGING_FACE_URL = "https://huggingface.co"


@dataclass
class HuggingFaceModel:
    id: str
    downloads: int = 0


@dataclass
class HuggingFaceQuantization:
    name: str
    size: int | None = None


@dataclass
class HuggingFaceModelDetails:
    id: str
    gated: bool | Literal["auto", "manual"] = False
    quantizations: list[HuggingFaceQuantization] = field(default_factory=list)


async def find_hugging_face_token(env: dict[str, str] | None = None) -> str | None:
    source = env or os.environ
    from_env = (source.get("HF_TOKEN") or "").strip()
    if from_env:
        return from_env
    paths = [
        source.get("HF_TOKEN_PATH"),
        os.path.join(source["HF_HOME"], "token") if source.get("HF_HOME") else None,
        os.path.join(source["XDG_CACHE_HOME"], "huggingface", "token") if source.get("XDG_CACHE_HOME") else None,
        str(Path.home() / ".cache" / "huggingface" / "token"),
    ]
    seen: set[str] = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        try:
            token = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if token:
            return token
    return None


class HuggingFaceClient:
    def __init__(self, token: str | None = None, base_url: str = DEFAULT_HUGGING_FACE_URL) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")

    async def _request(self, path: str) -> Any:
        from pi_coding_agent.core.http_dispatcher import get_async_http_client

        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        client = get_async_http_client()
        response = await client.get(f"{self.base_url}{path}", headers=headers)
        payload = None
        try:
            payload = response.json()
        except Exception:
            payload = None
        if response.is_error:
            message = payload.get("error") if isinstance(payload, dict) else None
            raise RuntimeError(message or f"Hugging Face returned HTTP {response.status_code}")
        return payload

    async def search(self, query: str, limit: int = 20) -> list[HuggingFaceModel]:
        payload = await self._request(f"/api/models?search={query}&limit={limit}")
        entries = payload if isinstance(payload, list) else []
        models: list[HuggingFaceModel] = []
        for item in entries:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                models.append(HuggingFaceModel(id=item["id"], downloads=int(item.get("downloads") or 0)))
        return models

    async def details(self, repository: str) -> HuggingFaceModelDetails:
        payload = await self._request(f"/api/models/{repository}")
        if not isinstance(payload, dict):
            return HuggingFaceModelDetails(id=repository)
        gated = payload.get("gated", False)
        files = payload.get("siblings") or []
        quantizations: list[HuggingFaceQuantization] = []
        if isinstance(files, list):
            for item in files:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("rfilename") or "")
                if name.endswith(".gguf"):
                    quantizations.append(HuggingFaceQuantization(name=name, size=item.get("size")))
        return HuggingFaceModelDetails(id=str(payload.get("id") or repository), gated=gated, quantizations=quantizations)


find_huggingface_token = find_hugging_face_token
