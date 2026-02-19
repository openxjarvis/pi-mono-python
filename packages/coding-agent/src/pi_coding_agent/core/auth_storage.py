"""
API key and OAuth credential storage — mirrors packages/coding-agent/src/core/auth-storage.ts
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class AuthStorage:
    """
    Stores API keys and OAuth credentials securely on disk.
    Mirrors AuthStorage in TypeScript.

    Storage: ~/.pi/agent/auth.json
    """

    AUTH_DIR = os.path.join(os.path.expanduser("~"), ".pi", "agent")
    AUTH_FILE = os.path.join(AUTH_DIR, "auth.json")

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()

    def _load(self) -> None:
        if os.path.exists(self.AUTH_FILE):
            try:
                with open(self.AUTH_FILE, encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}
        self._loaded = True

    def _save(self) -> None:
        os.makedirs(self.AUTH_DIR, exist_ok=True)
        # Write with restricted permissions
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        mode = 0o600
        fd = os.open(self.AUTH_FILE, flags, mode)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception:
            os.close(fd)
            raise

    def get_api_key(self, provider: str) -> str | None:
        """Get the stored API key for a provider."""
        self._ensure_loaded()
        return self._data.get("api_keys", {}).get(provider)

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Store an API key for a provider."""
        self._ensure_loaded()
        if "api_keys" not in self._data:
            self._data["api_keys"] = {}
        self._data["api_keys"][provider] = api_key
        self._save()

    def delete_api_key(self, provider: str) -> None:
        """Delete the stored API key for a provider."""
        self._ensure_loaded()
        if "api_keys" in self._data and provider in self._data["api_keys"]:
            del self._data["api_keys"][provider]
            self._save()

    def get_oauth_token(self, provider: str) -> dict[str, Any] | None:
        """Get the stored OAuth token for a provider."""
        self._ensure_loaded()
        return self._data.get("oauth_tokens", {}).get(provider)

    def set_oauth_token(self, provider: str, token: dict[str, Any]) -> None:
        """Store an OAuth token for a provider."""
        self._ensure_loaded()
        if "oauth_tokens" not in self._data:
            self._data["oauth_tokens"] = {}
        self._data["oauth_tokens"][provider] = token
        self._save()

    def delete_oauth_token(self, provider: str) -> None:
        """Delete the stored OAuth token for a provider."""
        self._ensure_loaded()
        if "oauth_tokens" in self._data and provider in self._data["oauth_tokens"]:
            del self._data["oauth_tokens"][provider]
            self._save()

    def resolve_api_key(self, provider: str) -> str | None:
        """
        Resolve API key for a provider.
        Priority: stored key → environment variable.
        """
        stored = self.get_api_key(provider)
        if stored:
            return stored
        from pi_ai.env_api_keys import get_env_api_key
        return get_env_api_key(provider)

    def list_stored_providers(self) -> list[str]:
        """List all providers with stored credentials."""
        self._ensure_loaded()
        api_providers = list(self._data.get("api_keys", {}).keys())
        oauth_providers = list(self._data.get("oauth_tokens", {}).keys())
        return sorted(set(api_providers + oauth_providers))
