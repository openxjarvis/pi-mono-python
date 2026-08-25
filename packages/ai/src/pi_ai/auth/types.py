"""
Auth types — mirrors packages/ai/src/auth/types.ts
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Protocol


@dataclass
class ModelAuth:
    api_key: str | None = None
    headers: dict[str, str] | None = None
    base_url: str | None = None


@dataclass
class ApiKeyCredential:
    type: Literal["api_key"] = "api_key"
    key: str | None = None
    env: dict[str, str] | None = None


@dataclass
class OAuthCredentials:
    refresh: str
    access: str
    expires: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OAuthCredential:
    type: Literal["oauth"] = "oauth"
    refresh: str = ""
    access: str = ""
    expires: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


Credential = ApiKeyCredential | OAuthCredential


@dataclass
class CredentialInfo:
    provider_id: str
    type: Literal["api_key", "oauth"]


@dataclass
class AuthOperationOptions:
    cancel_event: asyncio.Event | None = None


class CredentialStore(Protocol):
    async def read(self, provider_id: str, options: AuthOperationOptions | None = None) -> Credential | None: ...

    async def list(self, options: AuthOperationOptions | None = None) -> list[CredentialInfo]: ...

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None: ...

    async def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> None: ...


class AuthContext(Protocol):
    async def env(self, name: str) -> str | None: ...

    async def file_exists(self, path: str) -> bool: ...


@dataclass
class AuthResult:
    auth: ModelAuth
    env: dict[str, str] | None = None
    source: str | None = None


@dataclass
class AuthCheck:
    type: Literal["api_key", "oauth"]
    source: str | None = None


AuthType = Literal["api_key", "oauth"]


@dataclass
class AuthPrompt:
    type: Literal["text", "secret", "select", "manual_code"]
    message: str
    placeholder: str | None = None
    options: list[dict[str, str]] | None = None
    cancel_event: asyncio.Event | None = None


@dataclass
class AuthInfoLink:
    url: str
    label: str | None = None


@dataclass
class AuthEvent:
    type: Literal["info", "auth_url", "device_code", "progress"]
    message: str | None = None
    url: str | None = None
    instructions: str | None = None
    user_code: str | None = None
    verification_uri: str | None = None
    interval_seconds: float | None = None
    expires_in_seconds: float | None = None
    links: list[AuthInfoLink] | None = None


class AuthInteraction(Protocol):
    cancel_event: asyncio.Event | None

    async def prompt(self, prompt: AuthPrompt) -> str: ...

    def notify(self, event: AuthEvent) -> None: ...


@dataclass
class ApiKeyAuth:
    name: str
    login: Callable[[AuthInteraction], Awaitable[ApiKeyCredential]] | None = None
    check: Callable[..., Awaitable[AuthCheck | None]] | None = None
    resolve: Callable[..., Awaitable[AuthResult | None]] | None = None


@dataclass
class OAuthAuth:
    name: str
    login: Callable[[AuthInteraction], Awaitable[OAuthCredential]]
    refresh: Callable[[OAuthCredential, asyncio.Event], Awaitable[OAuthCredential]]
    to_auth: Callable[[OAuthCredential], Awaitable[ModelAuth]]
    is_subscription: bool | None = None
    login_label: str | None = None


@dataclass
class ProviderAuth:
    api_key: ApiKeyAuth | None = None
    oauth: OAuthAuth | None = None
