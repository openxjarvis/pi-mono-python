"""Provider factory for xiaomi-token-plan-cn — mirrors packages/ai/src/providers/xiaomi-token-plan-cn.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def xiaomi_token_plan_cn_provider():
    return create_builtin_provider("xiaomi-token-plan-cn")
