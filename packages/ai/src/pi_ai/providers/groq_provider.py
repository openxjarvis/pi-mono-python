"""Provider factory for groq — mirrors packages/ai/src/providers/groq.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def groq_provider():
    return create_builtin_provider("groq")
