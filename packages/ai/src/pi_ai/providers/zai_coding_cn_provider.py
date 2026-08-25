"""Provider factory for zai-coding-cn — mirrors packages/ai/src/providers/zai-coding-cn.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def zai_coding_cn_provider():
    return create_builtin_provider("zai-coding-cn")
