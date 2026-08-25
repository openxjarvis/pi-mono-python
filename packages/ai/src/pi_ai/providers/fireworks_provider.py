"""Provider factory for fireworks — mirrors packages/ai/src/providers/fireworks.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def fireworks_provider():
    return create_builtin_provider("fireworks")
