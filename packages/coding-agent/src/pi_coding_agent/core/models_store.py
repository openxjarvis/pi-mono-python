"""
Models catalog store — mirrors packages/coding-agent/src/core/models-store.ts
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any

from pi_coding_agent.config import get_agent_dir
from pi_coding_agent.utils.text import strip_bom


class InMemoryCodingAgentModelsStore:
    def __init__(self) -> None:
        self._entries: dict[str, Any] = {}

    async def read(self, provider_id: str, options: Any | None = None) -> dict[str, Any] | None:
        entry = self._entries.get(provider_id)
        return deepcopy(entry) if entry is not None else None

    async def write(self, provider_id: str, entry: dict[str, Any], options: Any | None = None) -> None:
        self._entries[provider_id] = deepcopy(entry)

    async def delete(self, provider_id: str, options: Any | None = None) -> None:
        self._entries.pop(provider_id, None)


class FileModelsStore:
    """Locked JSON-backed storage for dynamically refreshed provider catalogs."""

    def __init__(self, path: str | None = None) -> None:
        self.path = os.path.abspath(path or os.path.join(get_agent_dir(), "models-store.json"))
        self._data: dict[str, Any] = {}
        self._loaded = False

    def _load(self) -> dict[str, Any]:
        if not os.path.exists(self.path):
            self._data = {}
            self._loaded = True
            return self._data
        try:
            with open(self.path, encoding="utf-8") as f:
                parsed = json.loads(strip_bom(f.read()) or "{}")
            self._data = parsed if isinstance(parsed, dict) else {}
        except (OSError, json.JSONDecodeError):
            self._data = {}
        self._loaded = True
        return self._data

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
            f.write("\n")

    async def read(self, provider_id: str, options: Any | None = None) -> dict[str, Any] | None:
        if not self._loaded:
            self._load()
        entry = self._data.get(provider_id)
        return deepcopy(entry) if isinstance(entry, dict) else None

    async def write(self, provider_id: str, entry: dict[str, Any], options: Any | None = None) -> None:
        if not self._loaded:
            self._load()
        self._data[provider_id] = deepcopy(entry)
        self._save()

    async def delete(self, provider_id: str, options: Any | None = None) -> None:
        if not self._loaded:
            self._load()
        if provider_id in self._data:
            del self._data[provider_id]
            self._save()
