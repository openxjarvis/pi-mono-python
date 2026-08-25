"""Provider factory for deepseek — mirrors packages/ai/src/providers/deepseek.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def deepseek_provider():
    return create_builtin_provider("deepseek")
