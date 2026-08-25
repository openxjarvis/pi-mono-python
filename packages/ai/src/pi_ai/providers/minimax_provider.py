"""Provider factory for minimax — mirrors packages/ai/src/providers/minimax.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def minimax_provider():
    return create_builtin_provider("minimax")
