"""Provider factory for xiaomi — mirrors packages/ai/src/providers/xiaomi.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def xiaomi_provider():
    return create_builtin_provider("xiaomi")
