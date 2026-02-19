"""
Model registry and utilities — mirrors packages/ai/src/models.ts
"""
from __future__ import annotations

from .models_generated import MODELS
from .types import Model, Usage


def get_model(provider: str, model_id: str) -> Model:
    """Get a model by provider and model ID. Raises KeyError if not found."""
    key = f"{provider}/{model_id}"
    if key not in MODELS:
        raise KeyError(f"Model not found: {key}")
    return MODELS[key]


def get_providers() -> list[str]:
    """Return list of all registered providers."""
    seen: set[str] = set()
    result: list[str] = []
    for model in MODELS.values():
        if model.provider not in seen:
            seen.add(model.provider)
            result.append(model.provider)
    return sorted(result)


def get_models(provider: str | None = None) -> list[Model]:
    """Return all models, optionally filtered by provider."""
    models = list(MODELS.values())
    if provider is not None:
        models = [m for m in models if m.provider == provider]
    return models


def calculate_cost(model: Model, usage: Usage) -> float:
    """Calculate total cost in USD from usage and model pricing."""
    m = usage.input / 1_000_000 * model.cost.input
    m += usage.output / 1_000_000 * model.cost.output
    m += usage.cache_read / 1_000_000 * model.cost.cache_read
    m += usage.cache_write / 1_000_000 * model.cost.cache_write
    return m


def supports_xhigh(model: Model) -> bool:
    """Check if a model supports xhigh reasoning (currently only a few OpenAI models)."""
    xhigh_models = {
        "gpt-5.1-codex-max",
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-5.3",
        "gpt-5.3-codex",
    }
    return model.id in xhigh_models
