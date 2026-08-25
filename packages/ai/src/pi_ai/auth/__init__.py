"""
Auth substrate — mirrors packages/ai/src/auth/
"""
from .context import default_provider_auth_context
from .credential_store import InMemoryCredentialStore
from .helpers import env_api_key_auth, lazy_oauth
from .resolve import AuthResolutionOverrides, ModelsError, resolve_provider_auth
from .types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthCheck,
    AuthContext,
    AuthEvent,
    AuthInteraction,
    AuthPrompt,
    AuthResult,
    Credential,
    CredentialInfo,
    CredentialStore,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuth,
)

__all__ = [
    "ApiKeyAuth",
    "ApiKeyCredential",
    "AuthCheck",
    "AuthContext",
    "AuthEvent",
    "AuthInteraction",
    "AuthPrompt",
    "AuthResolutionOverrides",
    "AuthResult",
    "Credential",
    "CredentialInfo",
    "CredentialStore",
    "InMemoryCredentialStore",
    "ModelAuth",
    "ModelsError",
    "OAuthAuth",
    "OAuthCredential",
    "ProviderAuth",
    "default_provider_auth_context",
    "env_api_key_auth",
    "lazy_oauth",
    "resolve_provider_auth",
]
