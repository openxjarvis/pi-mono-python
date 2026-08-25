"""Provider factory for xai — mirrors packages/ai/src/providers/xai.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def xai_provider():
    return create_builtin_provider("xai")
