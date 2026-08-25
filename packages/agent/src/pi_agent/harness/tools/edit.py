"""Edit tool. Mirrors packages/agent/src/harness/tools/edit.ts"""
from __future__ import annotations

from typing import Any

from pi_agent.harness.result import get_or_throw
from pi_agent.harness.tools.edit_diff import apply_edit, format_diff
from pi_agent.harness.tools.path_utils import resolve_read_tool_path
from pi_agent.types import AgentTool, AgentToolResult


def create_edit_tool() -> AgentTool:
    async def execute(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None, context=None):
        env = context.env
        path = await resolve_read_tool_path(env, params["path"], signal)
        before = get_or_throw(await env.read_text_file(path, signal))
        after = apply_edit(before, params["oldText"], params["newText"])
        get_or_throw(await env.write_file(path, after, signal))
        return AgentToolResult(
            content=[{"type": "text", "text": format_diff(before, after, path) or f"Edited {path}"}],
            details={"path": path},
        )

    return AgentTool(
        name="edit",
        label="edit",
        description="Replace oldText with newText in a file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "oldText": {"type": "string"},
                "newText": {"type": "string"},
            },
            "required": ["path", "oldText", "newText"],
        },
        execute=execute,
    )
