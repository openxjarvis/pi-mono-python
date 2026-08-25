"""
Configured model collection — mirrors packages/coding-agent/src/core/model-runtime.ts
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .auth_storage import AuthStorage
from .model_config import ModelConfig
from .model_registry import ModelRegistry
from .models_store import FileModelsStore, InMemoryCodingAgentModelsStore
from .provider_composer import AuthStatus, ProviderConfigInput, compose_model_provider, validate_extension_provider
from .remote_catalog_provider import with_remote_catalog
from .runtime_credentials import RuntimeCredentials


class CredentialSynchronizationError(Exception):
    def __init__(self, provider_id: str, operation: str, credential: Any | None = None) -> None:
        super().__init__(f"Credential {operation} committed for {provider_id}, but local synchronization failed")
        self.provider_id = provider_id
        self.operation = operation
        self.credential = credential


@dataclass
class CreateModelRuntimeOptions:
    credentials: Any | None = None
    auth_path: str | None = None
    models_path: str | None = None
    models_store: Any | None = None
    models_store_path: str | None = None
    allow_model_network: bool = False
    catalog_base_url: str | None = None
    refresh_on_create: bool = True


@dataclass
class ModelRuntimeSnapshot:
    all: list[Any] = field(default_factory=list)
    available: list[Any] = field(default_factory=list)
    configured_providers: set[str] = field(default_factory=set)
    stored_providers: set[str] = field(default_factory=set)
    auth: dict[str, AuthStatus] = field(default_factory=dict)


class ModelRuntime:
    """Configured Models collection used by coding-agent and SDK consumers."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        credentials: RuntimeCredentials | None = None,
        config: ModelConfig | None = None,
        models_store: Any | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.credentials = credentials or RuntimeCredentials()
        self.config = config or ModelConfig()
        self.models_store = models_store or InMemoryCodingAgentModelsStore()
        self.extension_providers: dict[str, ProviderConfigInput | dict[str, Any]] = {}
        self.native_extension_providers: dict[str, Any] = {}
        self.composition_errors: dict[str, str] = {}
        self.snapshot = ModelRuntimeSnapshot(all=list(self.registry.get_all()))

    @classmethod
    async def create(cls, options: CreateModelRuntimeOptions | None = None) -> "ModelRuntime":
        opts = options or CreateModelRuntimeOptions()
        auth = AuthStorage()
        if opts.auth_path:
            auth.AUTH_FILE = opts.auth_path
            auth.AUTH_DIR = os.path.dirname(opts.auth_path)
        registry = ModelRegistry()
        config = ModelConfig.load(opts.models_path)
        store = opts.models_store or (
            FileModelsStore(opts.models_store_path) if opts.models_store_path else InMemoryCodingAgentModelsStore()
        )
        runtime = cls(
            registry=registry,
            credentials=RuntimeCredentials(auth),
            config=config,
            models_store=store,
        )
        if opts.refresh_on_create:
            await runtime.refresh(allow_network=opts.allow_model_network)
        return runtime

    async def refresh(self, *, allow_network: bool = False, providers: list[str] | None = None) -> dict[str, Any]:
        available = []
        if hasattr(self.registry, "get_available"):
            result = self.registry.get_available()
            available = await result if hasattr(result, "__await__") else result
        else:
            available = self.registry.get_all()
        all_models = self.registry.get_all()
        self.snapshot = ModelRuntimeSnapshot(
            all=list(all_models),
            available=list(available),
            configured_providers={getattr(m, "provider", "") for m in all_models},
        )
        return {"aborted": False, "errors": {}}

    def get_all(self) -> list[Any]:
        return list(self.snapshot.all or self.registry.get_all())

    def get_available(self) -> list[Any]:
        return list(self.snapshot.available or self.get_all())

    def get_model(self, provider: str, model_id: str) -> Any | None:
        if hasattr(self.registry, "find"):
            return self.registry.find(provider, model_id)
        return next(
            (
                m
                for m in self.get_all()
                if getattr(m, "provider", None) == provider and getattr(m, "id", None) == model_id
            ),
            None,
        )

    def get_api_key(self, provider: str) -> str | None:
        if hasattr(self.registry, "get_api_key"):
            return self.registry.get_api_key(provider)
        return None

    def register_provider(self, name: str, config: ProviderConfigInput | dict[str, Any]) -> None:
        errors = validate_extension_provider(config)
        if errors:
            self.composition_errors[name] = "; ".join(errors)
            raise ValueError(self.composition_errors[name])
        self.extension_providers[name] = config
        compose_model_provider(name, config)

    def register_native_provider(self, provider: Any) -> None:
        provider_id = getattr(provider, "id", None) or getattr(provider, "name", "unknown")
        self.native_extension_providers[str(provider_id)] = with_remote_catalog(provider)

    def set_runtime_api_key(self, provider_id: str, api_key: str) -> None:
        self.credentials.set_runtime_api_key(provider_id, api_key)

    def remove_runtime_api_key(self, provider_id: str) -> None:
        self.credentials.remove_runtime_api_key(provider_id)
