"""
Models collection runtime.
Mirrors the createModels() surface in packages/ai/src/models.ts
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pi_ai.auth import InMemoryCredentialStore, default_provider_auth_context, resolve_provider_auth
from pi_ai.auth.types import AuthResult, CredentialStore
from pi_ai.models_store import InMemoryModelsStore, ModelsStore
from pi_ai.provider_factory import Provider
from pi_ai.providers.catalog import builtin_providers
from pi_ai.types import Model


@dataclass
class Models:
    providers: list[Provider] = field(default_factory=list)
    credentials: CredentialStore = field(default_factory=InMemoryCredentialStore)
    store: ModelsStore = field(default_factory=InMemoryModelsStore)

    def get_models(self, provider_id: str | None = None) -> list[Model]:
        models: list[Model] = []
        for provider in self.providers:
            if provider_id and provider.id != provider_id:
                continue
            models.extend(provider.get_models())
        return models

    def get_provider(self, provider_id: str) -> Provider | None:
        return next((p for p in self.providers if p.id == provider_id), None)

    async def get_auth(self, provider_id: str) -> AuthResult | None:
        provider = self.get_provider(provider_id)
        if provider is None:
            return None
        return await resolve_provider_auth(provider, self.credentials, default_provider_auth_context())

    async def refresh(self, *, force: bool = False) -> None:
        del force
        return None


def create_models(
    providers: list[Provider] | None = None,
    credentials: CredentialStore | None = None,
    store: ModelsStore | None = None,
) -> Models:
    return Models(
        providers=providers if providers is not None else builtin_providers(),
        credentials=credentials or InMemoryCredentialStore(),
        store=store or InMemoryModelsStore(),
    )


def create_provider_from_id(provider_id: str) -> Provider:
    from pi_ai.providers.catalog import create_builtin_provider

    return create_builtin_provider(provider_id)


def has_api(api: str) -> bool:
    from pi_ai.api_registry import get_api_provider

    return get_api_provider(api) is not None
