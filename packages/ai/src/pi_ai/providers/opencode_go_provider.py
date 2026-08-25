"""Provider factory for opencode-go — mirrors packages/ai/src/providers/opencode-go.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def opencode_go_provider():
    return create_builtin_provider("opencode-go")
