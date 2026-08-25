"""
Standard api-key and lazy OAuth helpers.
Mirrors packages/ai/src/auth/helpers.ts
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from .types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthInteraction,
    AuthPrompt,
    AuthResult,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
)


def env_api_key_auth(name: str, env_vars: list[str]) -> ApiKeyAuth:
    async def login(interaction: AuthInteraction) -> ApiKeyCredential:
        if interaction.cancel_event is not None and interaction.cancel_event.is_set():
            raise asyncio.CancelledError("The operation was aborted")
        key = await interaction.prompt(AuthPrompt(type="secret", message=f"Enter {name}"))
        return ApiKeyCredential(key=key)

    async def resolve(*, ctx, credential=None, signal=None, **_kwargs) -> AuthResult | None:
        if signal is not None and getattr(signal, "is_set", lambda: False)():
            raise asyncio.CancelledError("The operation was aborted")
        if credential is not None and getattr(credential, "key", None):
            return AuthResult(
                auth=ModelAuth(api_key=credential.key),
                env=getattr(credential, "env", None),
                source="stored credential",
            )
        for env_var in env_vars:
            value = await ctx.env(env_var)
            if value:
                return AuthResult(auth=ModelAuth(api_key=value), source=env_var)
        return None

    return ApiKeyAuth(name=name, login=login, resolve=resolve)


def lazy_oauth(
    *,
    name: str,
    load: Callable[[], Awaitable[OAuthAuth]],
    is_subscription: bool | None = None,
    login_label: str | None = None,
) -> OAuthAuth:
    cached: OAuthAuth | None = None

    async def loaded() -> OAuthAuth:
        nonlocal cached
        if cached is None:
            cached = await load()
        return cached

    async def login(interaction: AuthInteraction) -> OAuthCredential:
        return await (await loaded()).login(interaction)

    async def refresh(credential: OAuthCredential, signal: asyncio.Event) -> OAuthCredential:
        return await (await loaded()).refresh(credential, signal)

    async def to_auth(credential: OAuthCredential) -> ModelAuth:
        return await (await loaded()).to_auth(credential)

    return OAuthAuth(
        name=name,
        login=login,
        refresh=refresh,
        to_auth=to_auth,
        is_subscription=is_subscription,
        login_label=login_label,
    )
