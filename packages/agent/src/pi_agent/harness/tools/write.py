"""Write tool. Mirrors packages/agent/src/harness/tools/write.ts"""
from __future__ import annotations

from typing import Any

from pi_agent.harness.result import get_or_throw
from pi_agent.harness.tools.path_utils import resolve_read_tool_path
from pi_agent.types import AgentTool, AgentToolResult


def create_write_tool() -> AgentTool:
    async def execute(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None, context=None):
        env = context.env
        path = await resolve_read_tool_path(env, params["path"], signal)
        get_or_throw(await env.write_file(path, params.get("content", ""), signal))
        return AgentToolResult(content=[{"type": "text", "text": f"Wrote {path}"}], details=None)

    return AgentTool(
        name="write",
        label="write",
        description="Write a file, creating parent directories when needed.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
        execute=execute,
    )
