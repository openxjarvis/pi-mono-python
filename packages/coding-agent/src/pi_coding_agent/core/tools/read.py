"""
Read file tool — mirrors packages/coding-agent/src/core/tools/read.ts
"""
from __future__ import annotations

import asyncio
import base64
import imghdr
import os
from dataclasses import dataclass
from typing import Any, Callable

import aiofiles

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import ImageContent, TextContent

from .path_utils import resolve_read_path
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
)

SUPPORTED_IMAGE_MIME_TYPES = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


@dataclass
class ReadToolDetails:
    truncation: TruncationResult | None = None


def detect_image_mime_type(path: str) -> str | None:
    """Detect image MIME type from file content."""
    try:
        kind = imghdr.what(path)
        return SUPPORTED_IMAGE_MIME_TYPES.get(kind)
    except Exception:
        return None


def create_read_tool(cwd: str, auto_resize_images: bool = True) -> AgentTool:
    """
    Create a read tool for the given working directory.
    Mirrors createReadTool() in TypeScript.
    """

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: asyncio.Event | None = None,
        on_update: Callable | None = None,
    ) -> AgentToolResult:
        path: str = params["path"]
        offset: int | None = params.get("offset")
        limit: int | None = params.get("limit")

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation aborted")

        absolute_path = resolve_read_path(path, cwd)

        # Check file exists and is readable
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"File not found: {path}")
        if not os.access(absolute_path, os.R_OK):
            raise PermissionError(f"Cannot read file: {path}")

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation aborted")

        # Detect image type
        mime_type = detect_image_mime_type(absolute_path)

        if mime_type:
            # Read as image
            async with aiofiles.open(absolute_path, "rb") as f:
                data = await f.read()
            b64 = base64.b64encode(data).decode("ascii")
            text_note = f"Read image file [{mime_type}]"
            return AgentToolResult(
                content=[
                    TextContent(type="text", text=text_note),
                    ImageContent(type="image", data=b64, mime_type=mime_type),
                ],
                details=ReadToolDetails(),
            )
        else:
            # Read as text
            async with aiofiles.open(absolute_path, "r", encoding="utf-8", errors="replace") as f:
                text_content = await f.read()

            all_lines = text_content.split("\n")
            total_file_lines = len(all_lines)

            start_line = max(0, (offset or 1) - 1) if offset is not None else 0
            start_line_display = start_line + 1

            if start_line >= len(all_lines):
                raise ValueError(f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)")

            user_limited_lines: int | None = None
            if limit is not None:
                end_line = min(start_line + limit, len(all_lines))
                selected_content = "\n".join(all_lines[start_line:end_line])
                user_limited_lines = end_line - start_line
            else:
                selected_content = "\n".join(all_lines[start_line:])

            truncation = truncate_head(selected_content)

            output_text: str
            details: ReadToolDetails | None = None

            if truncation.first_line_exceeds_limit:
                first_line_size = format_size(len(all_lines[start_line].encode("utf-8")))
                output_text = (
                    f"[Line {start_line_display} is {first_line_size}, exceeds "
                    f"{format_size(DEFAULT_MAX_BYTES)} limit. "
                    f"Use bash: sed -n '{start_line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
                )
                details = ReadToolDetails(truncation=truncation)
            elif truncation.truncated:
                end_line_display = start_line_display + truncation.output_lines - 1
                next_offset = end_line_display + 1
                output_text = truncation.content
                if truncation.truncated_by == "lines":
                    output_text += f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. Use offset={next_offset} to continue.]"
                else:
                    output_text += f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} ({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                details = ReadToolDetails(truncation=truncation)
            elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
                remaining = len(all_lines) - (start_line + user_limited_lines)
                next_offset = start_line + user_limited_lines + 1
                output_text = truncation.content
                output_text += f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            else:
                output_text = truncation.content

            return AgentToolResult(
                content=[TextContent(type="text", text=output_text)],
                details=details,
            )

    return AgentTool(
        name="read",
        label="read",
        description=(
            f"Read the contents of a file. Supports text files and images (jpg, png, gif, webp). "
            f"Images are sent as attachments. For text files, output is truncated to "
            f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
            f"Use offset/limit for large files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                "limit": {"type": "number", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
        },
        execute=execute,
    )


read_tool = create_read_tool(os.getcwd())
