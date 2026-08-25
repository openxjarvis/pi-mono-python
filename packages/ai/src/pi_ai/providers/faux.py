"""Test faux provider factory. Mirrors packages/ai/src/providers/faux.ts"""
from __future__ import annotations

from pi_ai.auth.helpers import env_api_key_auth
from pi_ai.auth.types import ProviderAuth
from pi_ai.provider_factory import Provider, create_provider
from pi_ai.types import Model, ModelCost


def faux_provider(models: list[Model] | None = None) -> Provider:
    return create_provider(
        id="faux",
        name="Faux",
        env_vars=[],
        models=models or [
            Model(
                id="faux",
                name="Faux",
                api="openai-completions",
                provider="faux",
                base_url="",
                reasoning=False,
                input=["text"],
                cost=ModelCost(),
                context_window=8192,
                max_tokens=1024,
            )
        ],
        auth=ProviderAuth(api_key=env_api_key_auth("Faux", [])),
    )
