"""Provider factory for github-copilot — mirrors packages/ai/src/providers/github-copilot.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def github_copilot_provider():
    return create_builtin_provider("github-copilot")
