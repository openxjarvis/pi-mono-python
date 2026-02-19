"""
Model registry — mirrors packages/coding-agent/src/core/model-registry.ts

Manages built-in and custom models, provides API key resolution.
Supports loading custom models from ~/.pi/agent/models.json,
applying provider-level and per-model overrides.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

from pi_ai import get_model, get_models, get_providers
from pi_ai.types import Model


# ─── Config schema (runtime validation without AJV) ───────────────────────────

def _validate_models_config(config: Any) -> str | None:
    """Validate models.json structure. Returns error string or None."""
    if not isinstance(config, dict):
        return "models.json must be a JSON object"
    providers = config.get("providers")
    if providers is None:
        return '"providers" key is required in models.json'
    if not isinstance(providers, dict):
        return '"providers" must be an object'

    for provider_name, provider_config in providers.items():
        if not isinstance(provider_config, dict):
            return f"Provider {provider_name}: config must be an object"

        models = provider_config.get("models") or []
        model_overrides = provider_config.get("modelOverrides") or {}

        if not models:
            # Override-only: needs baseUrl or modelOverrides
            if not provider_config.get("baseUrl") and not model_overrides:
                return (
                    f'Provider {provider_name}: must specify "baseUrl", '
                    '"modelOverrides", or "models".'
                )
        else:
            # Custom models: needs baseUrl + apiKey
            if not provider_config.get("baseUrl"):
                return f'Provider {provider_name}: "baseUrl" is required when defining custom models.'
            if not provider_config.get("apiKey"):
                return f'Provider {provider_name}: "apiKey" is required when defining custom models.'

        for model_def in models:
            provider_api = provider_config.get("api")
            model_api = model_def.get("api")
            if not provider_api and not model_api:
                mid = model_def.get("id", "?")
                return (
                    f'Provider {provider_name}, model {mid}: no "api" specified. '
                    "Set at provider or model level."
                )
            if not model_def.get("id"):
                return f'Provider {provider_name}: model missing "id"'
            cw = model_def.get("contextWindow")
            if cw is not None and cw <= 0:
                return f'Provider {provider_name}, model {model_def["id"]}: invalid contextWindow'
            mt = model_def.get("maxTokens")
            if mt is not None and mt <= 0:
                return f'Provider {provider_name}, model {model_def["id"]}: invalid maxTokens'

    return None


def _resolve_config_value(value: str) -> str | None:
    """
    Resolve a config value: env var reference (e.g. "$MY_KEY" or "$(cmd)")
    or a plain string.
    Mirrors resolveConfigValue() in TypeScript.
    """
    if not value:
        return None

    # Shell command substitution: $(cmd)
    m = re.match(r"^\$\((.+)\)$", value.strip())
    if m:
        try:
            result = subprocess.run(
                m.group(1), shell=True, capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip() or None
        except Exception:
            return None

    # Environment variable: $VAR_NAME or ${VAR_NAME}
    m2 = re.match(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$", value.strip())
    if m2:
        return os.environ.get(m2.group(1))

    # Plain string
    return value


def _resolve_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    """Resolve header values (env vars, shell commands)."""
    if not headers:
        return None
    resolved = {}
    for k, v in headers.items():
        val = _resolve_config_value(v)
        if val is not None:
            resolved[k] = val
    return resolved or None


def _apply_model_override(model: Model, override: dict[str, Any]) -> Model:
    """
    Deep merge a model override into a model.
    Mirrors applyModelOverride() in TypeScript.
    """
    result = Model(
        id=model.id,
        name=model.name,
        api=model.api,
        provider=model.provider,
        reasoning=model.reasoning,
        input=list(model.input),
        cost=dict(model.cost) if hasattr(model.cost, "__iter__") else model.cost,
        context_window=model.context_window,
        max_tokens=model.max_tokens,
        base_url=getattr(model, "base_url", None),
        headers=dict(model.headers) if model.headers else None,
        compat=dict(model.compat) if model.compat else None,
    )

    if override.get("name") is not None:
        result = Model(**{**result.__dict__, "name": override["name"]})
    if override.get("reasoning") is not None:
        result = Model(**{**result.__dict__, "reasoning": override["reasoning"]})
    if override.get("input") is not None:
        result = Model(**{**result.__dict__, "input": override["input"]})
    if override.get("contextWindow") is not None:
        result = Model(**{**result.__dict__, "context_window": override["contextWindow"]})
    if override.get("maxTokens") is not None:
        result = Model(**{**result.__dict__, "max_tokens": override["maxTokens"]})

    # Merge cost
    if override.get("cost"):
        base_cost = dict(result.cost) if result.cost else {}
        ov_cost = override["cost"]
        new_cost = {
            "input": ov_cost.get("input", base_cost.get("input", 0)),
            "output": ov_cost.get("output", base_cost.get("output", 0)),
            "cacheRead": ov_cost.get("cacheRead", base_cost.get("cacheRead", 0)),
            "cacheWrite": ov_cost.get("cacheWrite", base_cost.get("cacheWrite", 0)),
        }
        result = Model(**{**result.__dict__, "cost": new_cost})

    # Merge headers
    if override.get("headers"):
        resolved = _resolve_headers(override["headers"])
        if resolved:
            base_headers = dict(result.headers) if result.headers else {}
            result = Model(**{**result.__dict__, "headers": {**base_headers, **resolved}})

    # Merge compat
    if override.get("compat"):
        base_compat = dict(result.compat) if result.compat else {}
        result = Model(**{**result.__dict__, "compat": {**base_compat, **override["compat"]}})

    return result


@dataclass
class ProviderConfig:
    """Configuration for a registered provider."""
    name: str
    base_url: str | None = None
    api_key: str | None = None
    api: str | None = None
    headers: dict[str, str] | None = None
    auth_header: bool = False
    models: list[dict[str, Any]] = field(default_factory=list)
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


class ModelRegistry:
    """
    Model registry — loads and manages models, resolves API keys.
    Mirrors ModelRegistry in TypeScript.

    Supports:
    - Built-in models from pi_ai
    - Custom models/overrides from ~/.pi/agent/models.json
    - Runtime provider registration (for extensions)
    - API key resolution via env vars / shell commands
    """

    def __init__(
        self,
        auth_storage: Any = None,
        models_json_path: str | None = None,
    ) -> None:
        self._auth_storage = auth_storage
        if models_json_path is None:
            models_json_path = os.path.join(
                os.path.expanduser("~"), ".pi", "agent", "models.json"
            )
        self._models_json_path = models_json_path
        self._models: list[Model] = []
        self._custom_provider_api_keys: dict[str, str] = {}
        self._registered_providers: dict[str, ProviderConfig] = {}
        self._load_error: str | None = None
        self._extra_models: list[Model] = []

        self._load_models()

    def _load_models(self) -> None:
        """(Re)load built-in + custom models."""
        self._custom_provider_api_keys.clear()
        self._load_error = None

        custom_models, overrides, model_overrides, error = self._load_custom_models()
        if error:
            self._load_error = error

        built_in = self._load_built_in_models(overrides, model_overrides)
        combined = self._merge_custom_models(built_in, custom_models)

        # Apply OAuth provider modifications if auth_storage supports it
        if self._auth_storage and hasattr(self._auth_storage, "get_oauth_providers"):
            for provider in self._auth_storage.get_oauth_providers():
                cred = self._auth_storage.get(provider.id)
                if cred and getattr(cred, "type", None) == "oauth":
                    modify = getattr(provider, "modify_models", None)
                    if callable(modify):
                        combined = modify(combined, cred)

        # Apply registered extension providers
        for prov_name, prov_config in self._registered_providers.items():
            combined = self._apply_provider_config_to_models(combined, prov_name, prov_config)

        self._models = combined + self._extra_models

    def _load_built_in_models(
        self,
        overrides: dict[str, dict[str, Any]],
        model_overrides: dict[str, dict[str, dict[str, Any]]],
    ) -> list[Model]:
        """Load built-in models and apply provider/model overrides."""
        result: list[Model] = []
        for provider_name in get_providers():
            try:
                models = get_models(provider_name)
            except Exception:
                continue

            prov_override = overrides.get(provider_name, {})
            per_model = model_overrides.get(provider_name, {})

            for model in models:
                m = model

                # Apply provider-level baseUrl/headers
                if prov_override.get("baseUrl") or prov_override.get("headers"):
                    resolved_hdrs = _resolve_headers(prov_override.get("headers"))
                    base_url = prov_override.get("baseUrl", getattr(m, "base_url", None))
                    updates: dict[str, Any] = {"base_url": base_url}
                    if resolved_hdrs:
                        base_hdrs = dict(m.headers) if m.headers else {}
                        updates["headers"] = {**base_hdrs, **resolved_hdrs}
                    try:
                        m = Model(**{**m.__dict__, **updates})
                    except Exception:
                        pass

                # Apply per-model override
                if m.id in per_model:
                    try:
                        m = _apply_model_override(m, per_model[m.id])
                    except Exception:
                        pass

                result.append(m)

        return result

    def _merge_custom_models(self, built_in: list[Model], custom: list[Model]) -> list[Model]:
        """Merge custom models into built-in list (custom wins on ID collision)."""
        merged = list(built_in)
        for custom_model in custom:
            idx = next(
                (i for i, m in enumerate(merged)
                 if m.provider == custom_model.provider and m.id == custom_model.id),
                -1,
            )
            if idx >= 0:
                merged[idx] = custom_model
            else:
                merged.append(custom_model)
        return merged

    def _load_custom_models(
        self,
    ) -> tuple[list[Model], dict[str, Any], dict[str, Any], str | None]:
        """
        Load and parse models.json.
        Returns (models, provider_overrides, model_overrides, error).
        """
        path = self._models_json_path
        if not path or not os.path.exists(path):
            return [], {}, {}, None

        try:
            with open(path, encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            return [], {}, {}, f"Failed to parse models.json: {e}\n\nFile: {path}"
        except OSError as e:
            return [], {}, {}, f"Failed to load models.json: {e}\n\nFile: {path}"

        err = _validate_models_config(config)
        if err:
            return [], {}, {}, f"Invalid models.json schema:\n  - {err}\n\nFile: {path}"

        overrides: dict[str, Any] = {}
        model_overrides: dict[str, dict[str, Any]] = {}
        models: list[Model] = []

        for provider_name, prov_cfg in config.get("providers", {}).items():
            # Provider-level overrides for built-in models
            if prov_cfg.get("baseUrl") or prov_cfg.get("headers") or prov_cfg.get("apiKey"):
                overrides[provider_name] = {
                    "baseUrl": prov_cfg.get("baseUrl"),
                    "headers": prov_cfg.get("headers"),
                    "apiKey": prov_cfg.get("apiKey"),
                }

            if prov_cfg.get("apiKey"):
                self._custom_provider_api_keys[provider_name] = prov_cfg["apiKey"]

            if prov_cfg.get("modelOverrides"):
                model_overrides[provider_name] = prov_cfg["modelOverrides"]

            # Custom model definitions
            for model_def in prov_cfg.get("models") or []:
                api = model_def.get("api") or prov_cfg.get("api")
                if not api:
                    continue

                # Resolve headers
                prov_headers = _resolve_headers(prov_cfg.get("headers")) or {}
                model_headers = _resolve_headers(model_def.get("headers")) or {}
                merged_headers: dict[str, str] | None = {**prov_headers, **model_headers} or None

                # Auth header injection
                if prov_cfg.get("authHeader") and prov_cfg.get("apiKey"):
                    resolved_key = _resolve_config_value(prov_cfg["apiKey"])
                    if resolved_key:
                        merged_headers = {**(merged_headers or {}), "Authorization": f"Bearer {resolved_key}"}

                default_cost = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}

                try:
                    m = Model(
                        id=model_def["id"],
                        name=model_def.get("name") or model_def["id"],
                        api=api,
                        provider=provider_name,
                        reasoning=model_def.get("reasoning", False),
                        input=model_def.get("input", ["text"]),
                        cost=model_def.get("cost") or default_cost,
                        context_window=model_def.get("contextWindow", 128000),
                        max_tokens=model_def.get("maxTokens", 16384),
                        base_url=prov_cfg.get("baseUrl"),
                        headers=merged_headers,
                        compat=model_def.get("compat"),
                    )
                    models.append(m)
                except Exception:
                    pass

        return models, overrides, model_overrides, None

    def _apply_provider_config_to_models(
        self,
        models: list[Model],
        provider_name: str,
        prov_config: ProviderConfig,
    ) -> list[Model]:
        """Apply a registered provider config to the model list."""
        new_models = list(models)
        for model_def in prov_config.models:
            api = model_def.get("api") or prov_config.api
            if not api:
                continue
            default_cost = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
            try:
                m = Model(
                    id=model_def["id"],
                    name=model_def.get("name") or model_def["id"],
                    api=api,
                    provider=provider_name,
                    reasoning=model_def.get("reasoning", False),
                    input=model_def.get("input", ["text"]),
                    cost=model_def.get("cost") or default_cost,
                    context_window=model_def.get("contextWindow", 128000),
                    max_tokens=model_def.get("maxTokens", 16384),
                    base_url=prov_config.base_url,
                    headers=prov_config.headers,
                    compat=model_def.get("compat"),
                )
                # Check if it replaces an existing entry
                idx = next(
                    (i for i, em in enumerate(new_models)
                     if em.provider == provider_name and em.id == m.id),
                    -1,
                )
                if idx >= 0:
                    new_models[idx] = m
                else:
                    new_models.append(m)
            except Exception:
                pass
        return new_models

    # ── Public API ────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Reload models from disk (built-in + custom from models.json)."""
        self._load_models()

    def get_error(self) -> str | None:
        """Get any error from loading models.json."""
        return self._load_error

    def get_all(self) -> list[Model]:
        """Get all models (built-in + custom)."""
        return list(self._models)

    def get_all_models(self) -> list[Model]:
        """Alias for get_all()."""
        return self.get_all()

    async def get_available(self) -> list[Model]:
        """
        Get only models that have auth configured.
        Mirrors getAvailable() in TypeScript.
        """
        if self._auth_storage and hasattr(self._auth_storage, "has_auth"):
            return [m for m in self._models if self._auth_storage.has_auth(m.provider)]
        # Fallback: check environment variables
        return [m for m in self._models if self._has_env_auth(m.provider)]

    def _has_env_auth(self, provider: str) -> bool:
        """Check if a provider has auth via environment variable."""
        env_map = {
            "anthropic": ["ANTHROPIC_API_KEY"],
            "openai": ["OPENAI_API_KEY"],
            "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            "mistral": ["MISTRAL_API_KEY"],
            "groq": ["GROQ_API_KEY"],
            "cohere": ["COHERE_API_KEY"],
            "bedrock": ["AWS_ACCESS_KEY_ID"],
            "vertex": ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"],
        }
        keys = env_map.get(provider.lower(), [])
        return any(os.environ.get(k) for k in keys) or bool(
            self._custom_provider_api_keys.get(provider)
        )

    def get_api_key(self, provider: str) -> str | None:
        """
        Resolve API key for a provider.
        Checks custom config first, then auth_storage, then env vars.
        """
        # Custom key from models.json
        key_config = self._custom_provider_api_keys.get(provider)
        if key_config:
            return _resolve_config_value(key_config)

        # Auth storage
        if self._auth_storage:
            if hasattr(self._auth_storage, "get_api_key"):
                return self._auth_storage.get_api_key(provider)
            cred = self._auth_storage.get(provider) if hasattr(self._auth_storage, "get") else None
            if cred and hasattr(cred, "api_key"):
                return cred.api_key

        # Environment variables — delegate to the canonical resolver in pi_ai
        from pi_ai.env_api_keys import get_env_api_key
        return get_env_api_key(provider)

    def resolve_headers(self, model: Model) -> dict[str, str] | None:
        """
        Resolve headers for a model, interpolating env vars and shell commands.
        Mirrors resolveHeaders() in TypeScript.
        """
        if not model.headers:
            return None
        return _resolve_headers(model.headers)

    def register_provider(self, name: str, config: dict[str, Any]) -> None:
        """
        Register a provider from an extension.
        Mirrors registerProvider() in TypeScript.
        """
        prov = ProviderConfig(
            name=name,
            base_url=config.get("baseUrl"),
            api_key=config.get("apiKey"),
            api=config.get("api"),
            headers=config.get("headers"),
            auth_header=config.get("authHeader", False),
            models=config.get("models") or [],
            model_overrides=config.get("modelOverrides") or {},
        )
        self._registered_providers[name] = prov

        if prov.api_key:
            self._custom_provider_api_keys[name] = prov.api_key

        # Re-merge with new provider
        new_models = self._apply_provider_config_to_models(self._models, name, prov)
        self._models = new_models

    def register_model(self, model: Model) -> None:
        """Register an individual extra model."""
        self._extra_models.append(model)
        self._models.append(model)

    def get_model(self, provider: str, model_id: str) -> Model:
        """Get a model by provider and ID."""
        for m in self._models:
            if m.provider == provider and m.id == model_id:
                return m
        return get_model(provider, model_id)

    def find(self, provider: str, model_id: str) -> Model | None:
        """Find a model or return None."""
        for m in self._models:
            if m.provider == provider and m.id == model_id:
                return m
        try:
            return get_model(provider, model_id)
        except Exception:
            return None

    def get_providers(self) -> list[str]:
        """Get all available providers."""
        return list({m.provider for m in self._models})

    def resolve_model(
        self,
        model_id: str | None = None,
        provider: str | None = None,
    ) -> Model:
        """Resolve a model by optional ID and provider."""
        if model_id and provider:
            m = self.find(provider, model_id)
            if m:
                return m

        if model_id:
            for m in self._models:
                if m.id == model_id:
                    return m

        # Auto-select default based on which providers have API keys configured.
        # Prefer Google (Gemini) when GEMINI_API_KEY / GOOGLE_API_KEY is present,
        # then Anthropic, then OpenAI, then whatever is available.
        _default_preference = [
            ("google", "gemini-2.0-flash"),
            ("anthropic", "claude-3-5-sonnet-20241022"),
            ("openai", "gpt-4o"),
        ]
        for prov, mid in _default_preference:
            if self._has_env_auth(prov):
                m = self.find(prov, mid)
                if m:
                    return m

        # Last resort: first available model
        if self._models:
            return self._models[0]
        raise RuntimeError("No models available")
