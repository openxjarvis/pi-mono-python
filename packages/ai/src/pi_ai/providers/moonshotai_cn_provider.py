"""Provider factory for moonshotai-cn — mirrors packages/ai/src/providers/moonshotai-cn.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def moonshotai_cn_provider():
    return create_builtin_provider("moonshotai-cn")
