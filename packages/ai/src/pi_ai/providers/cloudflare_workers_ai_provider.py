"""Provider factory for cloudflare-workers-ai — mirrors packages/ai/src/providers/cloudflare-workers-ai.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def cloudflare_workers_ai_provider():
    return create_builtin_provider("cloudflare-workers-ai")
