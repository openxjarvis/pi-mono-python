"""Provider factory for minimax-cn — mirrors packages/ai/src/providers/minimax-cn.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def minimax_cn_provider():
    return create_builtin_provider("minimax-cn")
