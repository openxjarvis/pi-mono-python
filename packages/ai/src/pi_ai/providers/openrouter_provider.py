"""Provider factory for openrouter — mirrors packages/ai/src/providers/openrouter.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def openrouter_provider():
    return create_builtin_provider("openrouter")
