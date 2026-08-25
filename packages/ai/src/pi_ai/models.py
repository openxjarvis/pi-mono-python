"""
Model registry and utilities — mirrors packages/ai/src/models.ts
"""
from __future__ import annotations

from .models_generated import MODELS
from .types import Model, Usage


def get_model(provider: str, model_id: str) -> Model | None:
    """Get a model by provider and model ID. Returns None if not found."""
    key = f"{provider}/{model_id}"
    return MODELS.get(key)


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


def calculate_cost(model: Model, usage: Usage) -> "UsageCost":
    """Calculate cost in USD from usage and model pricing. Mutates usage.cost fields.

    Applies request-wide pricing tiers (highest matching input threshold wins)
    and Anthropic 1h cache-write charging (2x base input).
    """
    from .types import UsageCost

    if not isinstance(usage.cost, UsageCost):
        usage.cost = UsageCost()

    input_tokens = usage.input + usage.cache_read + usage.cache_write
    rates = model.cost
    matched_threshold = -1
    for tier in model.cost.tiers or []:
        if input_tokens > tier.input_tokens_above and tier.input_tokens_above > matched_threshold:
            rates = tier
            matched_threshold = tier.input_tokens_above

    long_write = usage.cache_write_1h or 0
    short_write = usage.cache_write - long_write
    usage.cost.input = (rates.input / 1_000_000) * usage.input
    usage.cost.output = (rates.output / 1_000_000) * usage.output
    usage.cost.cache_read = (rates.cache_read / 1_000_000) * usage.cache_read
    usage.cost.cache_write = (rates.cache_write * short_write + rates.input * 2 * long_write) / 1_000_000
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    return usage.cost


EXTENDED_THINKING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"]


def get_supported_thinking_levels(model: Model) -> list[str]:
    """Return thinking levels supported by this model. Mirrors TS getSupportedThinkingLevels."""
    if not model.reasoning:
        return ["off"]

    result: list[str] = []
    for level in EXTENDED_THINKING_LEVELS:
        mapped = model.thinking_level_map.get(level) if model.thinking_level_map else None
        if model.thinking_level_map and level in model.thinking_level_map and model.thinking_level_map[level] is None:
            continue
        if level in ("xhigh", "max"):
            if model.thinking_level_map is None or level not in model.thinking_level_map:
                continue
        result.append(level)
    return result or ["off"]


def clamp_thinking_level(model: Model, level: str) -> str:
    """Clamp a requested thinking level to one the model supports."""
    available = get_supported_thinking_levels(model)
    if level in available:
        return level
    try:
        requested_index = EXTENDED_THINKING_LEVELS.index(level)
    except ValueError:
        return available[0] if available else "off"
    for i in range(requested_index, len(EXTENDED_THINKING_LEVELS)):
        if EXTENDED_THINKING_LEVELS[i] in available:
            return EXTENDED_THINKING_LEVELS[i]
    for i in range(requested_index - 1, -1, -1):
        if EXTENDED_THINKING_LEVELS[i] in available:
            return EXTENDED_THINKING_LEVELS[i]
    return available[0] if available else "off"


def supports_xhigh(model: Model) -> bool:
    """Check if a model supports xhigh reasoning."""
    if model.thinking_level_map and "xhigh" in model.thinking_level_map:
        return model.thinking_level_map["xhigh"] is not None
    if "gpt-5.2" in model.id or "gpt-5.3" in model.id or "gpt-5.4" in model.id or "gpt-5.5" in model.id:
        return True
    if model.api == "anthropic-messages":
        return "opus-4-6" in model.id or "opus-4.6" in model.id or "opus-4-7" in model.id
    return False


def models_are_equal(a: Model | None, b: Model | None) -> bool:
    """Check if two models are equal by comparing both id and provider."""
    if a is None or b is None:
        return False
    return a.id == b.id and a.provider == b.provider
