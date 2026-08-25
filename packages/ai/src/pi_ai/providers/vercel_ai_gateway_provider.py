"""Provider factory for vercel-ai-gateway — mirrors packages/ai/src/providers/vercel-ai-gateway.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def vercel_ai_gateway_provider():
    return create_builtin_provider("vercel-ai-gateway")
