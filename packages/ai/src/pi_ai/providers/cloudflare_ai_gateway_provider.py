"""Provider factory for cloudflare-ai-gateway — mirrors packages/ai/src/providers/cloudflare-ai-gateway.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def cloudflare_ai_gateway_provider():
    return create_builtin_provider("cloudflare-ai-gateway")
