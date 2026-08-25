"""Provider factory for zai — mirrors packages/ai/src/providers/zai.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def zai_provider():
    return create_builtin_provider("zai")
