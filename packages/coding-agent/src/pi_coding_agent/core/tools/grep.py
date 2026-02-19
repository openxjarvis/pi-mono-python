"""
Grep tool using ripgrep — mirrors packages/coding-agent/src/core/tools/grep.ts
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Callable

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent

from .path_utils import resolve_to_cwd
from .truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_line,
)

DEFAULT_LIMIT = 100


@dataclass
class GrepToolDetails:
    truncation: TruncationResult | None = None
    match_limit_reached: int | None = None
    lines_truncated: bool = False


def _find_rg() -> str | None:
    """Find ripgrep executable."""
    return shutil.which("rg")


def create_grep_tool(cwd: str) -> AgentTool:
    """
    Create a grep tool using ripgrep.
    Mirrors createGrepTool() in TypeScript.
    """

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: asyncio.Event | None = None,
        on_update: Callable | None = None,
    ) -> AgentToolResult:
        pattern: str = params["pattern"]
        search_dir: str | None = params.get("path")
        glob: str | None = params.get("glob")
        ignore_case: bool = params.get("ignoreCase", False)
        literal: bool = params.get("literal", False)
        context: int = params.get("context", 0)
        limit: int = params.get("limit", DEFAULT_LIMIT)

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation aborted")

        rg_path = _find_rg()
        if not rg_path:
            raise RuntimeError("ripgrep (rg) is not available. Install it: https://github.com/BurntSushi/ripgrep")

        search_path = resolve_to_cwd(search_dir or ".", cwd)

        if not os.path.exists(search_path):
            raise FileNotFoundError(f"Path not found: {search_path}")

        is_directory = os.path.isdir(search_path)
        effective_limit = max(1, limit)
        context_value = max(0, context)

        # Build rg args
        args = ["--json", "--line-number", "--color=never", "--hidden"]
        if ignore_case:
            args.append("--ignore-case")
        if literal:
            args.append("--fixed-strings")
        if glob:
            args.extend(["--glob", glob])
        args.extend([pattern, search_path])

        # File cache for context lines
        file_cache: dict[str, list[str]] = {}

        def get_file_lines(file_path: str) -> list[str]:
            if file_path not in file_cache:
                try:
                    with open(file_path, encoding="utf-8", errors="replace") as f:
                        content = f.read().replace("\r\n", "\n").replace("\r", "\n")
                    file_cache[file_path] = content.split("\n")
                except Exception:
                    file_cache[file_path] = []
            return file_cache[file_path]

        def format_path(file_path: str) -> str:
            if is_directory:
                rel = os.path.relpath(file_path, search_path)
                if rel and not rel.startswith(".."):
                    return rel.replace("\\", "/")
            return os.path.basename(file_path)

        def format_block(file_path: str, line_number: int) -> list[str]:
            relative_path = format_path(file_path)
            lines = get_file_lines(file_path)
            if not lines:
                return [f"{relative_path}:{line_number}: (unable to read file)"]

            block: list[str] = []
            start = max(1, line_number - context_value) if context_value > 0 else line_number
            end = min(len(lines), line_number + context_value) if context_value > 0 else line_number

            for current in range(start, end + 1):
                line_text = lines[current - 1] if current <= len(lines) else ""
                sanitized = line_text.replace("\r", "")
                is_match_line = current == line_number
                truncated_text, was_truncated = truncate_line(sanitized)
                if was_truncated:
                    nonlocal lines_truncated
                    lines_truncated = True
                if is_match_line:
                    block.append(f"{relative_path}:{current}: {truncated_text}")
                else:
                    block.append(f"{relative_path}-{current}- {truncated_text}")

            return block

        lines_truncated = False
        match_count = 0
        match_limit_reached = False
        matches: list[dict[str, Any]] = []

        proc = await asyncio.create_subprocess_exec(
            rg_path, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stderr_data = b""

        async def read_stderr():
            nonlocal stderr_data
            if proc.stderr:
                stderr_data = await proc.stderr.read()

        stderr_task = asyncio.create_task(read_stderr())

        if proc.stdout:
            async for raw_line in proc.stdout:
                if cancel_event and cancel_event.is_set():
                    proc.kill()
                    raise RuntimeError("Operation aborted")

                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or match_count >= effective_limit:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "match":
                    match_count += 1
                    file_path = event.get("data", {}).get("path", {}).get("text")
                    line_number = event.get("data", {}).get("line_number")
                    if file_path and isinstance(line_number, int):
                        matches.append({"file_path": file_path, "line_number": line_number})

                    if match_count >= effective_limit:
                        match_limit_reached = True
                        proc.kill()
                        break

        await stderr_task
        await proc.wait()

        if match_count == 0:
            return AgentToolResult(
                content=[TextContent(type="text", text="No matches found")],
                details=None,
            )

        # Format matches
        output_lines: list[str] = []
        for match in matches:
            block = format_block(match["file_path"], match["line_number"])
            output_lines.extend(block)

        raw_output = "\n".join(output_lines)
        import sys
        truncation = truncate_head(raw_output, max_lines=sys.maxsize)

        output = truncation.content
        details = GrepToolDetails()
        notices: list[str] = []

        if match_limit_reached:
            notices.append(f"{effective_limit} matches limit reached. Use limit={effective_limit * 2} for more, or refine pattern")
            details.match_limit_reached = effective_limit

        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation

        if lines_truncated:
            notices.append(f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines")
            details.lines_truncated = True

        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        has_details = details.truncation or details.match_limit_reached or details.lines_truncated
        return AgentToolResult(
            content=[TextContent(type="text", text=output)],
            details=details if has_details else None,
        )

    return AgentTool(
        name="grep",
        label="grep",
        description=(
            f"Search file contents for a pattern. Returns matching lines with file paths and line numbers. "
            f"Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} matches or "
            f"{DEFAULT_MAX_BYTES // 1024}KB. Long lines are truncated to {GREP_MAX_LINE_LENGTH} chars."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
                "path": {"type": "string", "description": "Directory or file to search (default: current directory)"},
                "glob": {"type": "string", "description": "Filter files by glob pattern, e.g. '*.ts'"},
                "ignoreCase": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
                "literal": {"type": "boolean", "description": "Treat pattern as literal string (default: false)"},
                "context": {"type": "number", "description": "Lines to show before/after each match (default: 0)"},
                "limit": {"type": "number", "description": "Maximum number of matches (default: 100)"},
            },
            "required": ["pattern"],
        },
        execute=execute,
    )


grep_tool = create_grep_tool(os.getcwd())
