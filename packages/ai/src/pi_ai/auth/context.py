"""
Default auth context.
Mirrors packages/ai/src/auth/context.ts
"""
from __future__ import annotations

import os
from pathlib import Path


class DefaultAuthContext:
    async def env(self, name: str) -> str | None:
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value
        return None

    async def file_exists(self, path: str) -> bool:
        resolved = path
        if resolved.startswith("~"):
            resolved = str(Path.home()) + resolved[1:]
        return os.path.exists(resolved)


def default_provider_auth_context() -> DefaultAuthContext:
    return DefaultAuthContext()
