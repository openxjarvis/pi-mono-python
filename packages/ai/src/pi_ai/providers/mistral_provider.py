"""Provider factory for mistral — mirrors packages/ai/src/providers/mistral.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def mistral_provider():
    return create_builtin_provider("mistral")
