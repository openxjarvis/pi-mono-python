"""Provider factory for amazon-bedrock — mirrors packages/ai/src/providers/amazon-bedrock.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def amazon_bedrock_provider():
    return create_builtin_provider("amazon-bedrock")
