"""Provider factory for qwen-token-plan-cn — mirrors packages/ai/src/providers/qwen-token-plan-cn.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def qwen_token_plan_cn_provider():
    return create_builtin_provider("qwen-token-plan-cn")
