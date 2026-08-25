"""
xAI OAuth device-code flow.
Mirrors packages/ai/src/auth/oauth/xai.ts
"""
from __future__ import annotations

import asyncio
import time

import httpx

from pi_ai.auth.oauth.device_code import poll_device_code_token
from pi_ai.auth.types import (
    AuthEvent,
    AuthInteraction,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
)

XAI_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"
DEFAULT_TOKEN_LIFETIME_SECONDS = 3600


def _credential_from_token(data: dict) -> OAuthCredential:
    expires_in = int(data.get("expires_in") or DEFAULT_TOKEN_LIFETIME_SECONDS)
    return OAuthCredential(
        refresh=str(data.get("refresh_token") or ""),
        access=str(data.get("access_token") or ""),
        expires=int(time.time() * 1000) + expires_in * 1000,
    )


async def _login(interaction: AuthInteraction) -> OAuthCredential:
    cancel = interaction.cancel_event
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            XAI_DEVICE_CODE_URL,
            data={"client_id": XAI_CLIENT_ID, "scope": XAI_SCOPE},
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
        XAI_TOKEN_URL,
        client_id=XAI_CLIENT_ID,
        device_code=body["device_code"],
        interval_seconds=float(body.get("interval") or 5),
        expires_in_seconds=float(body.get("expires_in") or 900),
        cancel_event=cancel,
    )
    return _credential_from_token(token)


async def _refresh(credential: OAuthCredential, cancel: asyncio.Event) -> OAuthCredential:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            XAI_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": credential.refresh,
                "client_id": XAI_CLIENT_ID,
            },
        )
        response.raise_for_status()
        return _credential_from_token(response.json())


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


xai_oauth = OAuthAuth(
    name="xAI (OAuth)",
    login=_login,
    refresh=_refresh,
    to_auth=_to_auth,
    login_label="Sign in with SuperGrok or X Premium",
)
