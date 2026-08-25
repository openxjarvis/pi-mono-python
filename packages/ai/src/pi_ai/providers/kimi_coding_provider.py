"""Provider factory for kimi-coding — mirrors packages/ai/src/providers/kimi-coding.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def kimi_coding_provider():
    return create_builtin_provider("kimi-coding")
