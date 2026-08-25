"""Provider factory for google-vertex — mirrors packages/ai/src/providers/google-vertex.ts"""
from __future__ import annotations

from pi_ai.providers.catalog import create_builtin_provider


def google_vertex_provider():
    return create_builtin_provider("google-vertex")
