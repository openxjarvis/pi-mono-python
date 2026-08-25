"""Provider factory for moonshotai — mirrors packages/ai/src/providers/moonshotai.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def moonshotai_provider():
    return create_builtin_provider("moonshotai")
