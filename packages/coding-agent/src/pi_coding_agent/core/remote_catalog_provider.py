"""
Remote model catalog overlay — mirrors packages/coding-agent/src/core/remote-catalog-provider.ts
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

DEFAULT_CATALOG_BASE_URL = "https://pi.dev"
REMOTE_CATALOG_ATTEMPT_TIMEOUT_MS = 4_000
REMOTE_CATALOG_REFRESH_INTERVAL_MS = 4 * 60 * 60 * 1000


def merge_models(baseline: list[Any], dynamic: list[Any]) -> list[Any]:
    merged = list(baseline)
    for model in dynamic:
        model_id = model.get("id") if isinstance(model, dict) else getattr(model, "id", None)
        index = next(
            (
                i
                for i, entry in enumerate(merged)
                if (entry.get("id") if isinstance(entry, dict) else getattr(entry, "id", None)) == model_id
            ),
            -1,
        )
        if index >= 0:
            merged[index] = model
        else:
            merged.append(model)
    return merged


def parse_catalog(provider_id: str, value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        entries = value
    elif isinstance(value, dict) and isinstance(value.get("models"), list):
        entries = value["models"]
    elif isinstance(value, dict):
        entries = list(value.values())
    else:
        raise ValueError(f'Invalid model catalog for provider "{provider_id}"')
    models: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, dict) and "id" in entry:
            model = dict(entry)
            model["provider"] = provider_id
            models.append(model)
    return models


def remote_models(
    entry: dict[str, Any] | None,
    local_generated_at: int | None = None,
) -> list[Any]:
    if not entry:
        return []
    last_modified = entry.get("lastModified")
    if local_generated_at is not None and (last_modified is None or last_modified <= local_generated_at):
        return []
    models = entry.get("models") or []
    return list(models) if isinstance(models, list) else []


def with_remote_catalog(
    provider: Any,
    catalog_base_url: str = DEFAULT_CATALOG_BASE_URL,
    local_generated_at: int | None = None,
) -> Any:
    """Add a persisted catalog overlay to a static built-in provider."""
    dynamic_models: list[Any] = []
    original_get_models: Callable[[], list[Any]] | None = None
    if isinstance(provider, dict):
        original_get_models = provider.get("getModels") or provider.get("get_models")
        wrapped = dict(provider)
    else:
        original_get_models = getattr(provider, "get_models", None) or getattr(provider, "getModels", None)
        wrapped = provider

    def get_models() -> list[Any]:
        baseline = []
        if callable(original_get_models):
            baseline = list(original_get_models())
        elif isinstance(provider, dict):
            baseline = list(provider.get("models") or [])
        return merge_models(baseline, dynamic_models)

    async def refresh_models(context: Any | None = None) -> None:
        stored = None
        if context is not None:
            stored = getattr(context, "stored", None)
            if stored is None and isinstance(context, dict):
                stored = context.get("stored")
        restored = remote_models(stored, local_generated_at)
        dynamic_models[:] = restored

    if isinstance(wrapped, dict):
        wrapped = deepcopy(wrapped)
        wrapped["getModels"] = get_models
        wrapped["get_models"] = get_models
        wrapped["refreshModels"] = refresh_models
        wrapped["refresh_models"] = refresh_models
        wrapped["catalogBaseUrl"] = catalog_base_url
        return wrapped

    setattr(wrapped, "get_models", get_models)
    setattr(wrapped, "refresh_models", refresh_models)
    setattr(wrapped, "catalog_base_url", catalog_base_url)
    return wrapped
