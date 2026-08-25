"""Provider factory for together — mirrors packages/ai/src/providers/together.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def together_provider():
    return create_builtin_provider("together")
