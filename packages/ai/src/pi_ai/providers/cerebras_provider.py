"""Provider factory for cerebras — mirrors packages/ai/src/providers/cerebras.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def cerebras_provider():
    return create_builtin_provider("cerebras")
