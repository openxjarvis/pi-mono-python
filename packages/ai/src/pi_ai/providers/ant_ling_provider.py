"""Provider factory for ant-ling — mirrors packages/ai/src/providers/ant-ling.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def ant_ling_provider():
    return create_builtin_provider("ant-ling")
