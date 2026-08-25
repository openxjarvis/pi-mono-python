"""Provider factory for anthropic — mirrors packages/ai/src/providers/anthropic.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def anthropic_provider():
    return create_builtin_provider("anthropic")
