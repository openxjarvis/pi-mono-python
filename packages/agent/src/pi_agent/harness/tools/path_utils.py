"""Path helpers for harness tools. Mirrors packages/agent/src/harness/tools/path-utils.ts"""
from __future__ import annotations

import asyncio

from pi_agent.harness.result import get_or_throw
from pi_agent.harness.types import ExecutionEnv


async def resolve_read_tool_path(env: ExecutionEnv, path: str, abort: asyncio.Event | None = None) -> str:
    result = await env.absolute_path(path, abort)
    return get_or_throw(result)
