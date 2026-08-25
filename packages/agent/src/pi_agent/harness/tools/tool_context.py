"""Execution tool context. Mirrors packages/agent/src/harness/tools/tool-context.ts"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pi_agent.harness.types import ExecutionEnv


@dataclass
class ExecutionToolContext:
    env: ExecutionEnv
    extras: dict[str, Any] | None = None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
