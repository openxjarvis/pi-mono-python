"""Provider factory for xiaomi-token-plan-sgp — mirrors packages/ai/src/providers/xiaomi-token-plan-sgp.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def xiaomi_token_plan_sgp_provider():
    return create_builtin_provider("xiaomi-token-plan-sgp")
