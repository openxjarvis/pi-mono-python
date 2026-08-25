"""Shared Cloudflare auth helpers. Mirrors packages/ai/src/providers/cloudflare-auth.ts"""
from __future__ import annotations

from pi_ai.utils.provider_env import get_provider_env_value


def resolve_cloudflare_account_id(env: dict[str, str] | None = None) -> str | None:
    return (env or {}).get("CLOUDFLARE_ACCOUNT_ID") or get_provider_env_value("CLOUDFLARE_ACCOUNT_ID")


def resolve_cloudflare_gateway_id(env: dict[str, str] | None = None) -> str | None:
    return (env or {}).get("CLOUDFLARE_GATEWAY_ID") or get_provider_env_value("CLOUDFLARE_GATEWAY_ID")
