"""Provider factory for huggingface — mirrors packages/ai/src/providers/huggingface.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def huggingface_provider():
    return create_builtin_provider("huggingface")
