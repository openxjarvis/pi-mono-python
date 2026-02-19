"""
Write file tool — mirrors packages/coding-agent/src/core/tools/write.ts
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Callable

import aiofiles

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent

from .path_utils import resolve_to_cwd


def create_write_tool(cwd: str) -> AgentTool:
    """
    Create a write tool for the given working directory.
    Mirrors createWriteTool() in TypeScript.
    """

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: asyncio.Event | None = None,
        on_update: Callable | None = None,
    ) -> AgentToolResult:
        path: str = params["path"]
        content: str = params["content"]

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation aborted")

        absolute_path = resolve_to_cwd(path, cwd)
        dir_path = os.path.dirname(absolute_path)

        # Create parent directories
        os.makedirs(dir_path, exist_ok=True)

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation aborted")

        async with aiofiles.open(absolute_path, "w", encoding="utf-8") as f:
            await f.write(content)

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation aborted")

        return AgentToolResult(
            content=[TextContent(type="text", text=f"Successfully wrote {len(content)} bytes to {path}")],
            details=None,
        )

    return AgentTool(
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, "
            "overwrites if it does. Automatically creates parent directories."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        },
        execute=execute,
    )


write_tool = create_write_tool(os.getcwd())
