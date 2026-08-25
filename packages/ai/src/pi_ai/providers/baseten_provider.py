"""Provider factory for baseten — mirrors packages/ai/src/providers/baseten.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def baseten_provider():
    return create_builtin_provider("baseten")
