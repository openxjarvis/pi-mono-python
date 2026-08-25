"""Provider factory for nvidia — mirrors packages/ai/src/providers/nvidia.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def nvidia_provider():
    return create_builtin_provider("nvidia")
