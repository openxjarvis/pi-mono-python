"""
Bash execution tool — mirrors packages/coding-agent/src/core/tools/bash.ts
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent

from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size, truncate_tail


@dataclass
class BashToolDetails:
    truncation: TruncationResult | None = None
    full_output_path: str | None = None


def _get_shell() -> tuple[str, list[str]]:
    """Get the shell command and args for the current platform."""
    if sys.platform == "win32":
        return "cmd.exe", ["/c"]
    return os.environ.get("SHELL", "/bin/bash"), ["-c"]


def _kill_process_tree(pid: int) -> None:
    """Kill a process and its entire child tree — mirrors TS killProcessTree."""
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
            )
        except Exception:
            pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def create_bash_tool(cwd: str, command_prefix: str | None = None) -> AgentTool:
    """
    Create a bash execution tool.
    Mirrors createBashTool() in TypeScript.
    """

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: asyncio.Event | None = None,
        on_update: Callable | None = None,
    ) -> AgentToolResult:
        command: str = params["command"]
        timeout: float | None = params.get("timeout")

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation aborted")

        if not os.path.exists(cwd):
            raise RuntimeError(f"Working directory does not exist: {cwd}")

        resolved_command = f"{command_prefix}\n{command}" if command_prefix else command

        shell, args = _get_shell()

        # Track output
        chunks: list[bytes] = []
        chunks_bytes = 0
        max_chunks_bytes = DEFAULT_MAX_BYTES * 2
        total_bytes = 0
        temp_file_path: str | None = None
        temp_file = None

        async def run() -> int | None:
            nonlocal total_bytes, temp_file_path, temp_file, chunks_bytes

            # start_new_session=True creates a new process group so we can
            # kill the whole tree with killpg (mirrors TS detached: true)
            kwargs: dict[str, Any] = {}
            if sys.platform != "win32":
                kwargs["start_new_session"] = True

            process = await asyncio.create_subprocess_exec(
                shell, *args, resolved_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                env={**os.environ},
                **kwargs,
            )

            timed_out = False

            async def read_output():
                nonlocal total_bytes, temp_file_path, temp_file, chunks_bytes
                if process.stdout is None:
                    return
                async for chunk in process.stdout:
                    total_bytes += len(chunk)

                    if total_bytes > DEFAULT_MAX_BYTES and temp_file_path is None:
                        fd, temp_file_path = tempfile.mkstemp(prefix="pi-bash-", suffix=".log")
                        temp_file = os.fdopen(fd, "wb")
                        for existing in chunks:
                            temp_file.write(existing)

                    if temp_file is not None:
                        temp_file.write(chunk)

                    chunks.append(chunk)
                    chunks_bytes += len(chunk)
                    while chunks_bytes > max_chunks_bytes and len(chunks) > 1:
                        removed = chunks.pop(0)
                        chunks_bytes -= len(removed)

                    if on_update:
                        full_buf = b"".join(chunks)
                        full_text = full_buf.decode("utf-8", errors="replace")
                        trunc = truncate_tail(full_text)
                        on_update(AgentToolResult(
                            content=[TextContent(type="text", text=trunc.content or "")],
                            details=BashToolDetails(
                                truncation=trunc if trunc.truncated else None,
                                full_output_path=temp_file_path,
                            ),
                        ))

            read_task = asyncio.create_task(read_output())

            def _kill_proc():
                if process.pid is not None:
                    _kill_process_tree(process.pid)

            # Set up cancellation
            cancel_task = None
            if cancel_event:
                async def watch_cancel():
                    await cancel_event.wait()
                    _kill_proc()
                cancel_task = asyncio.create_task(watch_cancel())

            timeout_task = None
            if timeout is not None and timeout > 0:
                async def do_timeout():
                    nonlocal timed_out
                    await asyncio.sleep(timeout)
                    timed_out = True
                    _kill_proc()
                timeout_task = asyncio.create_task(do_timeout())

            await read_task
            exit_code = await process.wait()

            if cancel_task:
                cancel_task.cancel()
            if timeout_task:
                timeout_task.cancel()

            if temp_file is not None:
                temp_file.close()
                temp_file = None

            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Command aborted")

            if timed_out:
                raise RuntimeError(f"Command timed out after {timeout} seconds")

            return exit_code

        exit_code = await run()

        # Combine all chunks for final output
        full_buf = b"".join(chunks)
        full_output = full_buf.decode("utf-8", errors="replace")

        truncation = truncate_tail(full_output)
        output_text = truncation.content or "(no output)"
        details: BashToolDetails | None = None

        if truncation.truncated:
            details = BashToolDetails(truncation=truncation, full_output_path=temp_file_path)
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                last_line_size = format_size(len((full_output.split("\n") or [""])[-1].encode("utf-8")))
                output_text += f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line}. Full output: {temp_file_path}]"
            elif truncation.truncated_by == "lines":
                output_text += f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. Full output: {temp_file_path}]"
            else:
                output_text += f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} ({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {temp_file_path}]"

        if exit_code is not None and exit_code != 0:
            output_text += f"\n\nCommand exited with code {exit_code}"
            raise RuntimeError(output_text)

        return AgentToolResult(
            content=[TextContent(type="text", text=output_text)],
            details=details,
        )

    return AgentTool(
        name="bash",
        label="bash",
        description=(
            f"Execute a bash command in the current working directory. "
            f"Returns stdout and stderr. Output is truncated to last "
            f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB. "
            f"Optionally provide a timeout in seconds."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to execute"},
                "timeout": {"type": "number", "description": "Timeout in seconds (optional)"},
            },
            "required": ["command"],
        },
        execute=execute,
    )


bash_tool = create_bash_tool(os.getcwd())
