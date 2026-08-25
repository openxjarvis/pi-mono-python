"""Provider factory for opencode — mirrors packages/ai/src/providers/opencode.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def opencode_provider():
    return create_builtin_provider("opencode")
