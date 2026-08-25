"""
Kimi Code subscription OAuth (RFC 8628).
Mirrors packages/ai/src/auth/oauth/kimi-coding.ts
"""
from __future__ import annotations

import asyncio
import time

import httpx

from pi_ai.auth.oauth.device_code import poll_device_code_token
from pi_ai.auth.types import AuthEvent, AuthInteraction, ModelAuth, OAuthAuth, OAuthCredential
from pi_ai.utils.provider_env import get_provider_env_value

CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
DEFAULT_OAUTH_HOST = "https://auth.kimi.com"


def _oauth_host() -> str:
    override = get_provider_env_value("KIMI_CODE_OAUTH_HOST") or get_provider_env_value("KIMI_OAUTH_HOST")
    return (override or DEFAULT_OAUTH_HOST).rstrip("/")


def _credential_from_token(data: dict) -> OAuthCredential:
    expires_in = int(data.get("expires_in") or 3600)
    return OAuthCredential(
        refresh=str(data.get("refresh_token") or ""),
        access=str(data.get("access_token") or ""),
        expires=int(time.time() * 1000) + expires_in * 1000,
    )


async def _login(interaction: AuthInteraction) -> OAuthCredential:
    host = _oauth_host()
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{host}/oauth/device/code",
            json={"client_id": CLIENT_ID},
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
    interaction.notify(
        AuthEvent(
            type="device_code",
            user_code=body.get("user_code"),
            verification_uri=body.get("verification_uri"),
            interval_seconds=body.get("interval"),
            expires_in_seconds=body.get("expires_in"),
        )
    )
    token = await poll_device_code_token(
        f"{host}/oauth/token",
        client_id=CLIENT_ID,
        device_code=body["device_code"],
        interval_seconds=float(body.get("interval") or 5),
        expires_in_seconds=float(body.get("expires_in") or 900),
        cancel_event=interaction.cancel_event,
    )
    return _credential_from_token(token)


async def _refresh(credential: OAuthCredential, cancel: asyncio.Event) -> OAuthCredential:
    host = _oauth_host()
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{host}/oauth/token",
            data={"grant_type": "refresh_token", "refresh_token": credential.refresh, "client_id": CLIENT_ID},
        )
        response.raise_for_status()
        return _credential_from_token(response.json())


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


kimi_coding_oauth = OAuthAuth(
    name="Kimi Coding (OAuth)",
    login=_login,
    refresh=_refresh,
    to_auth=_to_auth,
    is_subscription=True,
)
