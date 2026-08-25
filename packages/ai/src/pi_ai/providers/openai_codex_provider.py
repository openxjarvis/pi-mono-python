"""Provider factory for openai-codex — mirrors packages/ai/src/providers/openai-codex.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def openai_codex_provider():
    return create_builtin_provider("openai-codex")
