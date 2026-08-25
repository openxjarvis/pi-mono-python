"""
Resolve a provider env value from scoped overrides then process env.
Mirrors packages/ai/src/utils/provider-env.ts
"""
from __future__ import annotations

import os
from typing import Mapping

ProviderEnv = Mapping[str, str]


def get_provider_env_value(name: str, env: ProviderEnv | None = None) -> str | None:
    if env and env.get(name):
        return env[name]
    return os.environ.get(name) or None
