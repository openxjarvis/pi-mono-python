"""Read tool. Mirrors packages/agent/src/harness/tools/read.ts"""
from __future__ import annotations

from typing import Any

from pi_agent.harness.result import get_or_throw
from pi_agent.harness.tools.image import detect_supported_image_mime_type, encode_base64
from pi_agent.harness.tools.path_utils import resolve_read_tool_path
from pi_agent.harness.utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_head
from pi_agent.types import AgentTool, AgentToolResult

ReadImageProcessor = Any
ReadImageProcessorResult = dict[str, Any]
ReadToolDetails = dict[str, Any]
ReadToolOptions = dict[str, Any]


def create_read_tool(options: dict[str, Any] | None = None) -> AgentTool:
    options = options or {}

    async def execute(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None, context=None):
        env = context.env
        path = await resolve_read_tool_path(env, params["path"], signal)
        data = get_or_throw(await env.read_binary_file(path, signal))
        mime = detect_supported_image_mime_type(data)
        if mime:
            return AgentToolResult(
                content=[{"type": "text", "text": f"Read image file [{mime}]"}, {"type": "image", "data": encode_base64(data), "mimeType": mime}],
                details=None,
            )
        text = data.decode("utf-8", errors="replace")
        offset = max(int(params.get("offset") or 1) - 1, 0)
        limit = params.get("limit")
        lines = text.splitlines()
        selected = lines[offset : offset + int(limit)] if limit is not None else lines[offset:]
        joined = "\n".join(selected)
        truncation = truncate_head(joined, {"max_lines": DEFAULT_MAX_LINES, "max_bytes": DEFAULT_MAX_BYTES})
        numbered = []
        for i, line in enumerate(truncation["content"].splitlines(), start=offset + 1):
            numbered.append(f"{i:>6}|{line}")
        return AgentToolResult(content=[{"type": "text", "text": "\n".join(numbered)}], details={"truncation": truncation})

    return AgentTool(
        name="read",
        label="read",
        description=f"Read the contents of a file. Truncated to {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "number"}, "limit": {"type": "number"}}},
        execute=execute,
    )
