"""Provider factory for google — mirrors packages/ai/src/providers/google.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def google_provider():
    return create_builtin_provider("google")
