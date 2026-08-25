"""Provider factory for qwen-token-plan-individual — mirrors packages/ai/src/providers/qwen-token-plan-individual.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def qwen_token_plan_individual_provider():
    return create_builtin_provider("qwen-token-plan-individual")
