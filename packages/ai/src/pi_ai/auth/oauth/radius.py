"""
Radius gateway OAuth (device-code + PKCE).
Mirrors packages/ai/src/auth/oauth/radius.ts
"""
from __future__ import annotations

import asyncio
import time

import httpx

from pi_ai.auth.oauth.device_code import poll_device_code_token
from pi_ai.auth.types import AuthEvent, AuthInteraction, AuthPrompt, ModelAuth, OAuthAuth, OAuthCredential
from pi_ai.utils.provider_env import get_provider_env_value

OAUTH_CLIENT_ID = "pi-gateway"
OAUTH_SCOPE = "gateway offline_access"


def _gateway() -> str:
    return (get_provider_env_value("RADIUS_GATEWAY_URL") or "https://api.pi.dev").rstrip("/")


async def _login(interaction: AuthInteraction) -> OAuthCredential:
    gateway = _gateway()
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            discovery = await client.get(f"{gateway}/v1/oauth", headers={"accept": "application/json"})
            discovery.raise_for_status()
        except Exception:
            discovery = None
        response = await client.post(
            f"{gateway}/v1/oauth/device/code",
            json={"client_id": OAUTH_CLIENT_ID, "scope": OAUTH_SCOPE},
        )
        if response.is_success:
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
                f"{gateway}/v1/oauth/token",
                client_id=OAUTH_CLIENT_ID,
                device_code=body["device_code"],
                interval_seconds=float(body.get("interval") or 5),
                expires_in_seconds=float(body.get("expires_in") or 900),
                cancel_event=interaction.cancel_event,
            )
            return OAuthCredential(
                refresh=str(token.get("refresh_token") or ""),
                access=str(token.get("access_token") or ""),
                expires=int(time.time() * 1000) + int(token.get("expires_in") or 3600) * 1000,
            )
    raw = await interaction.prompt(AuthPrompt(type="manual_code", message="Paste Radius authorization code"))
    return OAuthCredential(refresh=raw, access=raw, expires=int(time.time() * 1000) + 3600 * 1000)


async def _refresh(credential: OAuthCredential, cancel: asyncio.Event) -> OAuthCredential:
    gateway = _gateway()
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{gateway}/v1/oauth/token",
            data={"grant_type": "refresh_token", "refresh_token": credential.refresh, "client_id": OAUTH_CLIENT_ID},
        )
        response.raise_for_status()
        token = response.json()
    return OAuthCredential(
        refresh=str(token.get("refresh_token") or credential.refresh),
        access=str(token.get("access_token") or ""),
        expires=int(time.time() * 1000) + int(token.get("expires_in") or 3600) * 1000,
    )


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


radius_oauth = OAuthAuth(
    name="Radius (OAuth)",
    login=_login,
    refresh=_refresh,
    to_auth=_to_auth,
)
