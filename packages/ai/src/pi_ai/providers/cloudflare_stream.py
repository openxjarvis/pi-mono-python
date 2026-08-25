"""Cloudflare streaming URL helpers. Mirrors packages/ai/src/providers/cloudflare-stream.ts"""
from __future__ import annotations

from .cloudflare_auth import resolve_cloudflare_account_id, resolve_cloudflare_gateway_id


def workers_ai_url(account_id: str | None = None, model_id: str = "") -> str:
    account = account_id or resolve_cloudflare_account_id() or ""
    return f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model_id}"


def gateway_url(account_id: str | None = None, gateway_id: str | None = None) -> str:
    account = account_id or resolve_cloudflare_account_id() or ""
    gateway = gateway_id or resolve_cloudflare_gateway_id() or ""
    return f"https://gateway.ai.cloudflare.com/v1/{account}/{gateway}"
