"""Provider factory for openai — mirrors packages/ai/src/providers/openai.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def openai_provider():
    return create_builtin_provider("openai")
