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
        self._runtime_overrides: dict[str, str] = {}  # Runtime API key overrides (in-memory only)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()

    def _load(self) -> None:
        if os.path.exists(self.AUTH_FILE):
            try:
                from pi_coding_agent.utils.text import load_json_file
                self._data = load_json_file(self.AUTH_FILE)
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

    def set_runtime_api_key(self, provider: str, api_key: str) -> None:
        """
        Set runtime API key override (in-memory only, not persisted to disk).
        Mirrors TypeScript AuthStorage.setRuntimeApiKey().
        
        Runtime overrides have the highest priority in resolve_api_key().
        Used by openclaw to inject API keys from auth-profiles.json.
        """
        self._runtime_overrides[provider] = api_key

    def resolve_api_key(self, provider: str) -> str | None:
        """
        Resolve API key for a provider.
        Priority (aligned with TypeScript):
        1. Runtime override (CLI --api-key, openclaw injection) - highest
        2. OAuth token from auth.json (auto-refresh)
        3. Stored API key from auth.json
        4. Environment variable
        """
        # 1. Runtime override takes highest priority
        if provider in self._runtime_overrides:
            return self._runtime_overrides[provider]
        
        # 2. Try OAuth token
        oauth = self.get_oauth_token(provider)
        if oauth:
            access_token = oauth.get("access_token")
            if access_token:
                # Check expiry
                expires_at = oauth.get("expires_at", 0)
                import time
                if expires_at and expires_at > time.time():
                    return access_token
                # Token expired, try refresh
                refreshed = self._refresh_oauth_token(provider)
                if refreshed:
                    return refreshed

        # 3. Stored key from auth.json
        stored = self.get_api_key(provider)
        if stored:
            return stored
        
        # 4. Environment variable fallback
        from pi_ai.env_api_keys import get_env_api_key
        return get_env_api_key(provider)

    def is_using_oauth(self, provider: str) -> bool:
        """Check if provider uses OAuth authentication."""
        self._ensure_loaded()
        return provider in self._data.get("oauth_tokens", {})

    async def login(
        self,
        provider: str,
        client_id: str | None = None,
        client_secret: str | None = None,
        auth_url: str | None = None,
        token_url: str | None = None,
        scopes: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """
        Perform OAuth login for a provider.
        Returns the token dict on success, None on failure.
        Mirrors login() in TypeScript AuthStorage.
        """
        import webbrowser
        import urllib.parse
        import secrets
        import asyncio

        state = secrets.token_urlsafe(32)
        redirect_port = 19747
        redirect_uri = f"http://localhost:{redirect_port}/oauth/callback"

        if not auth_url or not token_url:
            return None

        params = {
            "client_id": client_id or "",
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "state": state,
            "scope": " ".join(scopes or []),
        }
        full_url = f"{auth_url}?{urllib.parse.urlencode(params)}"

        auth_code: str | None = None
        received_state: str | None = None

        from http.server import HTTPServer, BaseHTTPRequestHandler

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self_handler):
                nonlocal auth_code, received_state
                parsed = urllib.parse.urlparse(self_handler.path)
                qs = urllib.parse.parse_qs(parsed.query)
                auth_code = qs.get("code", [None])[0]
                received_state = qs.get("state", [None])[0]
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "text/html")
                self_handler.end_headers()
                self_handler.wfile.write(b"<html><body><h1>Login successful. You can close this tab.</h1></body></html>")

            def log_message(self_handler, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", redirect_port), CallbackHandler)
        server.timeout = 120

        webbrowser.open(full_url)

        # Wait for callback in a thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, server.handle_request)
        server.server_close()

        if not auth_code or received_state != state:
            return None

        # Exchange code for token
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(token_url, data={
                "grant_type": "authorization_code",
                "code": auth_code,
                "redirect_uri": redirect_uri,
                "client_id": client_id or "",
                "client_secret": client_secret or "",
            })
            if resp.status_code != 200:
                return None
            token_data = resp.json()

        import time
        expires_in = token_data.get("expires_in", 3600)
        token_data["expires_at"] = time.time() + expires_in

        self.set_oauth_token(provider, token_data)
        return token_data

    def logout(self, provider: str) -> None:
        """Logout from a provider by removing stored credentials."""
        self.delete_api_key(provider)
        self.delete_oauth_token(provider)

    def _refresh_oauth_token(self, provider: str) -> str | None:
        """Attempt to refresh an expired OAuth token synchronously."""
        oauth = self.get_oauth_token(provider)
        if not oauth:
            return None
        refresh_token = oauth.get("refresh_token")
        token_url = oauth.get("token_url")
        client_id = oauth.get("client_id")
        if not refresh_token or not token_url:
            return None

        try:
            import httpx
            with httpx.Client() as client:
                resp = client.post(token_url, data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id or "",
                })
                if resp.status_code != 200:
                    return None
                token_data = resp.json()

            import time
            expires_in = token_data.get("expires_in", 3600)
            token_data["expires_at"] = time.time() + expires_in
            token_data["refresh_token"] = refresh_token
            token_data["token_url"] = token_url
            token_data["client_id"] = client_id

            self.set_oauth_token(provider, token_data)
            return token_data.get("access_token")
        except Exception:
            return None

    def get_oauth_providers(self) -> list[str]:
        """Get list of providers with OAuth credentials."""
        self._ensure_loaded()
        return list(self._data.get("oauth_tokens", {}).keys())

    def list_stored_providers(self) -> list[str]:
        """List all providers with stored credentials."""
        self._ensure_loaded()
        api_providers = list(self._data.get("api_keys", {}).keys())
        oauth_providers = list(self._data.get("oauth_tokens", {}).keys())
        return sorted(set(api_providers + oauth_providers))
