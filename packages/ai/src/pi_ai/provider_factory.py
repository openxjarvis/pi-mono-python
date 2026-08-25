"""
Provider factory — mirrors createProvider() in packages/ai/src/models.ts
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pi_ai.auth.helpers import env_api_key_auth
from pi_ai.auth.types import ProviderAuth
from pi_ai.models_generated import MODELS
from pi_ai.types import Model


@dataclass
class Provider:
    id: str
    name: str
    auth: ProviderAuth
    base_url: str | None = None
    headers: dict[str, str] | None = None
    models: list[Model] = field(default_factory=list)
    api: str | None = None
    get_models_fn: Callable[[], list[Model]] | None = None

    def get_models(self) -> list[Model]:
        if self.get_models_fn:
            return list(self.get_models_fn())
        return list(self.models)

    def filter_models(self, models: list[Model] | None = None) -> list[Model]:
        return list(models if models is not None else self.get_models())


def create_provider(
    *,
    id: str,
    name: str,
    env_vars: list[str],
    base_url: str | None = None,
    api: str | None = None,
    models: list[Model] | None = None,
    auth: ProviderAuth | None = None,
) -> Provider:
    catalog = models if models is not None else [m for m in MODELS.values() if m.provider == id]
    return Provider(
        id=id,
        name=name,
        auth=auth or ProviderAuth(api_key=env_api_key_auth(f"{name} API key", env_vars)),
        base_url=base_url,
        models=catalog,
        api=api,
    )


def models_for_provider(provider_id: str) -> list[Model]:
    return [m for m in MODELS.values() if m.provider == provider_id]
