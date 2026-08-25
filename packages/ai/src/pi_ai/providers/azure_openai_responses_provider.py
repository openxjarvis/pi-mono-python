"""Provider factory for azure-openai-responses — mirrors packages/ai/src/providers/azure-openai-responses.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def azure_openai_responses_provider():
    return create_builtin_provider("azure-openai-responses")
