"""
Auth resolution shared by Models collections.
Mirrors packages/ai/src/auth/resolve.ts
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from pi_ai.utils.abort import operation_signal, race_with_abort_signal
from pi_ai.utils.diagnostics import format_thrown_value

from .types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthContext,
    AuthResult,
    Credential,
    CredentialStore,
    OAuthAuth,
    OAuthCredential,
    ProviderAuth,
)

ModelsErrorCode = str
DEFAULT_OAUTH_MINIMUM_VALIDITY_MS = 5 * 60 * 1000
DEFAULT_OAUTH_REFRESH_TIMEOUT_MS = 15_000


@dataclass
class AuthResolutionOverrides:
    api_key: str | None = None
    env: dict[str, str] | None = None
    min_oauth_validity_ms: int | None = None
    cancel_event: asyncio.Event | None = None


class ModelsError(Exception):
    def __init__(self, code: ModelsErrorCode, message: str, cause: object | None = None) -> None:
        super().__init__(_with_cause_detail(message, cause))
        self.code = code
        self.cause = cause


def _with_cause_detail(message: str, cause: object | None) -> str:
    if cause is None:
        return message
    detail = format_thrown_value(cause).strip()
    if not detail or detail in message:
        return message
    return f"{message}: {detail}"


class _OverlayAuthContext:
    def __init__(self, base: AuthContext, env: dict[str, str]) -> None:
        self._base = base
        self._env = env

    async def env(self, name: str) -> str | None:
        return self._env.get(name) or await self._base.env(name)

    async def file_exists(self, path: str) -> bool:
        return await self._base.file_exists(path)


async def resolve_provider_auth(
    provider: object,
    credentials: CredentialStore,
    auth_context: AuthContext,
    overrides: AuthResolutionOverrides | None = None,
) -> AuthResult | None:
    cancel = operation_signal(overrides.cancel_event if overrides else None)
    return await race_with_abort_signal(
        _resolve_with_signal(provider, credentials, auth_context, overrides, cancel),
        cancel,
    )


async def _resolve_with_signal(
    provider: object,
    credentials: CredentialStore,
    auth_context: AuthContext,
    overrides: AuthResolutionOverrides | None,
    cancel: asyncio.Event,
) -> AuthResult | None:
    provider_id = getattr(provider, "id", None) or getattr(provider, "provider_id", "")
    auth: ProviderAuth = getattr(provider, "auth")
    request_ctx = _OverlayAuthContext(auth_context, overrides.env) if overrides and overrides.env else auth_context

    if overrides and overrides.api_key is not None and auth.api_key:
        return await _resolve_api_key(
            request_ctx,
            auth.api_key,
            provider_id,
            ApiKeyCredential(key=overrides.api_key, env=overrides.env),
            cancel,
        )

    stored = await _read_credential(credentials, provider_id, cancel)
    if stored:
        if stored.type == "oauth" and auth.oauth:
            return await _resolve_stored_oauth(
                credentials,
                provider_id,
                auth.oauth,
                stored,  # type: ignore[arg-type]
                cancel,
                overrides.min_oauth_validity_ms if overrides else None,
            )
        if stored.type == "api_key" and auth.api_key:
            credential = stored
            if overrides and overrides.env:
                credential = ApiKeyCredential(
                    key=getattr(stored, "key", None),
                    env={**(getattr(stored, "env", None) or {}), **overrides.env},
                )
            return await _resolve_api_key(request_ctx, auth.api_key, provider_id, credential, cancel)
        return None

    return await _resolve_api_key(request_ctx, auth.api_key, provider_id, None, cancel) if auth.api_key else None


async def _resolve_stored_oauth(
    credentials: CredentialStore,
    provider_id: str,
    oauth: OAuthAuth,
    stored: OAuthCredential,
    cancel: asyncio.Event,
    min_oauth_validity_ms: int | None,
) -> AuthResult | None:
    minimum = max(DEFAULT_OAUTH_MINIMUM_VALIDITY_MS, min_oauth_validity_ms or 0)

    def expires_soon(credential: OAuthCredential) -> bool:
        return time.time() * 1000 + minimum >= credential.expires

    credential = stored
    if expires_soon(credential):
        async def _refresh(current: Credential | None) -> Credential | None:
            if current is None or current.type != "oauth":
                return None
            if not expires_soon(current):  # type: ignore[arg-type]
                return None
            try:
                return await oauth.refresh(current, cancel)  # type: ignore[arg-type]
            except Exception as error:
                raise ModelsError("oauth", f"OAuth refresh failed for {provider_id}", error) from error

        try:
            post = await credentials.modify(provider_id, _refresh)
        except ModelsError:
            raise
        except Exception as error:
            raise ModelsError("auth", f"Credential store modify failed for {provider_id}", error) from error
        if post is None or post.type != "oauth":
            return None
        credential = post  # type: ignore[assignment]
        if min_oauth_validity_ms is not None and expires_soon(credential):
            raise ModelsError("oauth", f"OAuth refresh returned a token that expires too soon for {provider_id}")

    try:
        return AuthResult(auth=await oauth.to_auth(credential), source="OAuth")
    except Exception as error:
        raise ModelsError("oauth", f"OAuth auth derivation failed for {provider_id}", error) from error


async def _resolve_api_key(
    auth_context: AuthContext,
    api_key: ApiKeyAuth,
    provider_id: str,
    credential: ApiKeyCredential | None,
    cancel: asyncio.Event,
) -> AuthResult | None:
    if api_key.resolve is None:
        return None
    try:
        return await api_key.resolve(ctx=auth_context, credential=credential, signal=cancel)
    except Exception as error:
        raise ModelsError("auth", f"API key auth failed for provider {provider_id}", error) from error


async def _read_credential(
    credentials: CredentialStore,
    provider_id: str,
    cancel: asyncio.Event,
) -> Credential | None:
    try:
        from .types import AuthOperationOptions

        return await credentials.read(provider_id, AuthOperationOptions(cancel_event=cancel))
    except Exception as error:
        raise ModelsError("auth", f"Credential store read failed for {provider_id}", error) from error
