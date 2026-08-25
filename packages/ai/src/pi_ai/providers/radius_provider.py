"""Provider factory for radius — mirrors packages/ai/src/providers/radius.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def radius_provider():
    return create_builtin_provider("radius")
