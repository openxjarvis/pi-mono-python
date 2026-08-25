"""
Provider composition helpers — mirrors packages/coding-agent/src/core/provider-composer.ts
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .model_config import ModelConfig
from .resolve_config_value import resolve_config_value


@dataclass
class AuthStatus:
    configured: bool
    source: str | None = None
    label: str | None = None


@dataclass
class ProviderConfigInput:
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    models: list[dict[str, Any]] = field(default_factory=list)
    stream_simple: Callable[..., Any] | None = None
    oauth: Any = None
    refresh_models: Callable[..., Any] | None = None


def configured_request_auth_status(
    provider_id: str,
    *,
    has_stored: bool = False,
    has_runtime: bool = False,
    env_key: str | None = None,
    models_json_key: str | None = None,
) -> AuthStatus:
    if has_runtime:
        return AuthStatus(configured=True, source="runtime", label=provider_id)
    if has_stored:
        return AuthStatus(configured=True, source="stored", label=provider_id)
    if env_key:
        return AuthStatus(configured=True, source="environment", label=env_key)
    if models_json_key:
        return AuthStatus(configured=True, source="models_json_key", label=models_json_key)
    return AuthStatus(configured=False)


def resolve_configured_model_headers(
    headers: dict[str, str] | None,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for name, value in (headers or {}).items():
        try:
            resolved[name] = resolve_config_value(value) or str(value)
        except Exception:
            resolved[name] = str(value)
    return resolved


def resolve_compatibility_request_config(
    model: Any,
    provider_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    compat = getattr(model, "compat", None) or {}
    if hasattr(compat, "model_dump"):
        compat = compat.model_dump()
    elif not isinstance(compat, dict):
        compat = getattr(compat, "__dict__", {}) or {}
    merged = dict(compat)
    if provider_config:
        provider_compat = provider_config.get("compat") or {}
        if isinstance(provider_compat, dict):
            merged = {**provider_compat, **merged}
    return merged


def compose_model_provider(
    provider_id: str,
    config: ModelConfig | ProviderConfigInput | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(config, ModelConfig):
        provider = config.get_provider(provider_id) or {}
        return {"id": provider_id, **provider}
    if isinstance(config, ProviderConfigInput):
        return {
            "id": provider_id,
            "name": config.name or provider_id,
            "baseUrl": config.base_url,
            "apiKey": config.api_key,
            "api": config.api,
            "headers": config.headers,
            "models": config.models,
        }
    if isinstance(config, dict):
        return {"id": provider_id, **config}
    return {"id": provider_id}


def validate_extension_provider(config: ProviderConfigInput | dict[str, Any]) -> list[str]:
    errors: list[str] = []
    name = config.name if isinstance(config, ProviderConfigInput) else config.get("name")
    if not name:
        errors.append("Extension provider requires a name")
    return errors


def clear_api_key_cache() -> None:
    try:
        from .resolve_config_value import clear_config_value_cache

        clear_config_value_cache()
    except Exception:
        pass
