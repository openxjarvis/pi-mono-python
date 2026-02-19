"""
Find files tool — mirrors packages/coding-agent/src/core/tools/find.ts
"""
from __future__ import annotations

import asyncio
import fnmatch
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent

from .path_utils import resolve_to_cwd
from .truncate import DEFAULT_MAX_BYTES, TruncationResult, format_size, truncate_head

DEFAULT_LIMIT = 1000


@dataclass
class FindToolDetails:
    truncation: TruncationResult | None = None
    result_limit_reached: int | None = None


def _find_fd() -> str | None:
    """Find the fd/fdfind executable."""
    return shutil.which("fd") or shutil.which("fdfind")


def _glob_files(pattern: str, search_path: str, limit: int) -> list[str]:
    """Fallback glob using Python's os.walk if fd is not available."""
    results: list[str] = []
    ignore_dirs = {".git", "node_modules", "__pycache__", ".venv"}

    for root, dirs, files in os.walk(search_path):
        # Filter out ignored directories in-place
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for filename in files:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, search_path)
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(rel_path, pattern):
                results.append(rel_path)
                if len(results) >= limit:
                    return results

    return results


def create_find_tool(cwd: str) -> AgentTool:
    """
    Create a find tool.
    Mirrors createFindTool() in TypeScript.
    """

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: asyncio.Event | None = None,
        on_update: Callable | None = None,
    ) -> AgentToolResult:
        pattern: str = params["pattern"]
        search_dir: str | None = params.get("path")
        limit: int = params.get("limit", DEFAULT_LIMIT)

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation aborted")

        search_path = resolve_to_cwd(search_dir or ".", cwd)
        effective_limit = limit

        if not os.path.exists(search_path):
            raise FileNotFoundError(f"Path not found: {search_path}")

        # Try fd first, fall back to Python glob
        fd_path = _find_fd()
        relativized: list[str] = []

        if fd_path:
            args = [
                fd_path,
                "--glob",
                "--color=never",
                "--hidden",
                "--max-results", str(effective_limit),
                pattern,
                search_path,
            ]
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            output = (result.stdout or "").strip()
            if output:
                for raw_line in output.split("\n"):
                    line = raw_line.strip().rstrip("/\\").rstrip("/")
                    if not line:
                        continue
                    if line.startswith(search_path):
                        rel = line[len(search_path):].lstrip(os.sep)
                    else:
                        rel = os.path.relpath(line, search_path)
                    relativized.append(rel)
        else:
            # Fallback
            relativized = await asyncio.get_event_loop().run_in_executor(
                None, _glob_files, pattern, search_path, effective_limit
            )

        if not relativized:
            return AgentToolResult(
                content=[TextContent(type="text", text="No files found matching pattern")],
                details=None,
            )

        result_limit_reached = len(relativized) >= effective_limit
        raw_output = "\n".join(relativized)
        import sys
        truncation = truncate_head(raw_output, max_lines=sys.maxsize)

        result_output = truncation.content
        details = FindToolDetails()
        notices: list[str] = []

        if result_limit_reached:
            notices.append(f"{effective_limit} results limit reached. Use limit={effective_limit * 2} for more, or refine pattern")
            details.result_limit_reached = effective_limit

        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation

        if notices:
            result_output += f"\n\n[{'. '.join(notices)}]"

        has_details = details.truncation or details.result_limit_reached
        return AgentToolResult(
            content=[TextContent(type="text", text=result_output)],
            details=details if has_details else None,
        )

    return AgentTool(
        name="find",
        label="find",
        description=(
            f"Search for files by glob pattern. Returns matching file paths relative to the search directory. "
            f"Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} results or "
            f"{DEFAULT_MAX_BYTES // 1024}KB."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json'"},
                "path": {"type": "string", "description": "Directory to search in (default: current directory)"},
                "limit": {"type": "number", "description": "Maximum number of results (default: 1000)"},
            },
            "required": ["pattern"],
        },
        execute=execute,
    )


find_tool = create_find_tool(os.getcwd())
