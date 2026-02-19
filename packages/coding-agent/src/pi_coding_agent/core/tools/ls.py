"""
List directory tool — mirrors packages/coding-agent/src/core/tools/ls.ts
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Callable

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent

from .path_utils import resolve_to_cwd
from .truncate import DEFAULT_MAX_BYTES, TruncationResult, format_size, truncate_head

DEFAULT_LIMIT = 500


@dataclass
class LsToolDetails:
    truncation: TruncationResult | None = None
    entry_limit_reached: int | None = None


def create_ls_tool(cwd: str) -> AgentTool:
    """
    Create a directory listing tool.
    Mirrors createLsTool() in TypeScript.
    """

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: asyncio.Event | None = None,
        on_update: Callable | None = None,
    ) -> AgentToolResult:
        path: str | None = params.get("path")
        limit: int = params.get("limit", DEFAULT_LIMIT)

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation aborted")

        dir_path = resolve_to_cwd(path or ".", cwd)
        effective_limit = limit

        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Path not found: {dir_path}")

        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        try:
            raw_entries = os.listdir(dir_path)
        except PermissionError as e:
            raise RuntimeError(f"Cannot read directory: {e}")

        # Sort case-insensitive
        raw_entries.sort(key=lambda e: e.lower())

        results: list[str] = []
        entry_limit_reached = False

        for entry in raw_entries:
            if len(results) >= effective_limit:
                entry_limit_reached = True
                break

            full_path = os.path.join(dir_path, entry)
            try:
                suffix = "/" if os.path.isdir(full_path) else ""
            except OSError:
                continue

            results.append(entry + suffix)

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation aborted")

        if not results:
            return AgentToolResult(
                content=[TextContent(type="text", text="(empty directory)")],
                details=None,
            )

        import sys
        raw_output = "\n".join(results)
        truncation = truncate_head(raw_output, max_lines=sys.maxsize)

        output = truncation.content
        details = LsToolDetails()
        notices: list[str] = []

        if entry_limit_reached:
            notices.append(f"{effective_limit} entries limit reached. Use limit={effective_limit * 2} for more")
            details.entry_limit_reached = effective_limit

        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation

        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        has_details = details.truncation or details.entry_limit_reached
        return AgentToolResult(
            content=[TextContent(type="text", text=output)],
            details=details if has_details else None,
        )

    return AgentTool(
        name="ls",
        label="ls",
        description=(
            f"List directory contents. Returns entries sorted alphabetically, "
            f"with '/' suffix for directories. Includes dotfiles. "
            f"Output is truncated to {DEFAULT_LIMIT} entries or {DEFAULT_MAX_BYTES // 1024}KB."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list (default: current directory)"},
                "limit": {"type": "number", "description": "Maximum number of entries to return (default: 500)"},
            },
        },
        execute=execute,
    )


ls_tool = create_ls_tool(os.getcwd())
