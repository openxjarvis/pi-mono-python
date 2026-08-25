"""
Runtime credential overlay — mirrors packages/coding-agent/src/core/runtime-credentials.ts
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from .auth_storage import AuthStorage


class RuntimeCredentials:
    """Async credential store overlay for non-persistent runtime API keys."""

    def __init__(self, store: AuthStorage | None = None) -> None:
        self.store = store or AuthStorage()
        self._overrides: dict[str, str] = {}

    def set_runtime_api_key(self, provider_id: str, api_key: str) -> None:
        self._overrides[provider_id] = api_key

    def remove_runtime_api_key(self, provider_id: str) -> None:
        self._overrides.pop(provider_id, None)

    def has_runtime_api_key(self, provider_id: str) -> bool:
        return provider_id in self._overrides

    async def read(self, provider_id: str, options: Any | None = None) -> dict[str, Any] | None:
        override = self._overrides.get(provider_id)
        if override:
            return {"type": "api_key", "key": override}
        key = self.store.resolve_api_key(provider_id) if hasattr(self.store, "resolve_api_key") else self.store.get_api_key(provider_id)
        return {"type": "api_key", "key": key} if key else None

    async def list(self, options: Any | None = None) -> list[dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {}
        stored = getattr(self.store, "_data", {})
        if not getattr(self.store, "_loaded", True) and hasattr(self.store, "_ensure_loaded"):
            self.store._ensure_loaded()
            stored = getattr(self.store, "_data", {})
        for provider in (stored.get("api_keys") or {}):
            entries[provider] = {"providerId": provider, "type": "api_key"}
        for provider in self._overrides:
            entries[provider] = {"providerId": provider, "type": "api_key"}
        return list(entries.values())

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[dict[str, Any] | None], Awaitable[dict[str, Any] | None] | dict[str, Any] | None],
        options: Any | None = None,
    ) -> dict[str, Any] | None:
        current = await self.read(provider_id, options)
        updated = fn(current)
        if isinstance(updated, Awaitable):
            updated = await updated
        return updated

    async def delete(self, provider_id: str, options: Any | None = None) -> None:
        if hasattr(self.store, "delete_api_key"):
            self.store.delete_api_key(provider_id)
        self._overrides.pop(provider_id, None)
