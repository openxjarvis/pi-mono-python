"""
OpenRouter OAuth PKCE — exchanges an authorization code for a user API key.
Mirrors packages/ai/src/auth/oauth/openrouter.ts
"""
from __future__ import annotations

import asyncio
import time

import httpx

from pi_ai.auth.types import AuthEvent, AuthInteraction, AuthPrompt, ModelAuth, OAuthAuth, OAuthCredential
from pi_ai.utils.oauth.pkce import generate_pkce
from pi_ai.utils.provider_env import get_provider_env_value

AUTHORIZE_URL = "https://openrouter.ai/auth"
TOKEN_URL = "https://openrouter.ai/api/v1/auth/keys"


def _parse_code(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    if "code=" in text:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(text if "://" in text else f"http://local/?{text}")
        codes = parse_qs(parsed.query).get("code")
        if codes:
            return codes[0]
    return text


async def _login(interaction: AuthInteraction) -> OAuthCredential:
    verifier, challenge = generate_pkce()
    host = get_provider_env_value("PI_OAUTH_CALLBACK_HOST") or "127.0.0.1"
    callback = f"http://{host}/oauth/callback"
    url = f"{AUTHORIZE_URL}?callback_url={callback}&code_challenge={challenge}&code_challenge_method=S256"
    interaction.notify(AuthEvent(type="auth_url", url=url, instructions="Open the URL and paste the redirect or code."))
    raw = await interaction.prompt(AuthPrompt(type="manual_code", message="Paste OpenRouter redirect URL or code"))
    code = _parse_code(raw)
    if not code:
        raise RuntimeError("Missing OpenRouter authorization code")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            TOKEN_URL,
            json={"code": code, "code_verifier": verifier, "code_challenge_method": "S256"},
        )
        response.raise_for_status()
        data = response.json()
    key = data.get("key") or data.get("api_key") or data.get("access_token")
    if not key:
        raise RuntimeError("OpenRouter token exchange returned no key")
    return OAuthCredential(refresh=str(key), access=str(key), expires=int(time.time() * 1000) + 365 * 24 * 3600 * 1000)


async def _refresh(credential: OAuthCredential, cancel: asyncio.Event) -> OAuthCredential:
    return credential


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access or credential.refresh)


openrouter_oauth = OAuthAuth(
    name="OpenRouter (OAuth)",
    login=_login,
    refresh=_refresh,
    to_auth=_to_auth,
)
