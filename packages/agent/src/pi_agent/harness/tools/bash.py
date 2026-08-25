from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, TypedDict

from pi_ai.types import TextContent

from pi_agent.harness.tools.tool_context import ExecutionToolContext
from pi_agent.harness.types import AgentHarnessTool, get_or_throw
from pi_agent.harness.utils.shell_output import ShellCaptureProgress, execute_shell_with_capture
from pi_agent.harness.utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size
from pi_agent.types import AgentToolResult

MAX_TIMEOUT_SECONDS = 2_147_483_647 / 1000
BASH_UPDATE_THROTTLE_MS = 100

BASH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
    },
    "required": ["command"],
}


class BashToolDetails(TypedDict, total=False):
    truncation: TruncationResult
    full_output_path: str


class BashExecution(TypedDict):
    command: str
    cwd: str
    env: dict[str, str]
    inherit_env: bool


BashPrepare = Callable[[BashExecution, ExecutionToolContext, asyncio.Event | None], Awaitable[None] | None]


class BashToolOptions(TypedDict, total=False):
    command_prefix: str
    prepare: BashPrepare


def _validate_timeout(timeout: float | None) -> None:
    if timeout is None:
        return
    if not isinstance(timeout, (int, float)) or timeout != timeout or timeout <= 0:  # noqa: PLR0124
        raise RuntimeError("Invalid timeout: must be a finite number of seconds")
    if timeout > MAX_TIMEOUT_SECONDS:
        raise RuntimeError(f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds")


def create_bash_tool(options: BashToolOptions | None = None) -> AgentHarnessTool:
    options = options or {}

    async def execute(
        _tool_call_id: str,
        params: dict[str, Any],
        abort: asyncio.Event | None,
        on_update: Any,
        context: ExecutionToolContext,
    ) -> AgentToolResult:
        command = params["command"]
        timeout = params.get("timeout")
        _validate_timeout(timeout)
        env = context["env"]
        prefix = options.get("command_prefix")
        execution: BashExecution = {
            "command": f"{prefix}\n{command}" if prefix else command,
            "cwd": env.cwd,
            "env": {},
            "inherit_env": True,
        }
        prepare = options.get("prepare")
        if prepare is not None:
            prepared = prepare(execution, context, abort)
            if asyncio.iscoroutine(prepared):
                await prepared

        latest_progress: Callable[[], ShellCaptureProgress] | None = None
        update_timer: asyncio.TimerHandle | None = None
        update_dirty = False
        last_update_at = 0.0
        loop = asyncio.get_running_loop()

        def emit_output_update() -> None:
            nonlocal update_dirty, last_update_at
            if on_update is None or not update_dirty or latest_progress is None:
                return
            update_dirty = False
            last_update_at = loop.time() * 1000
            progress = latest_progress()
            details: BashToolDetails = {}
            if progress["truncation"]["truncated"]:
                details["truncation"] = progress["truncation"]
            if progress.get("full_output_path"):
                details["full_output_path"] = progress["full_output_path"]
            on_update(
                AgentToolResult(
                    content=[TextContent(type="text", text=progress["output"])],
                    details=details or None,
                )
            )

        def clear_update_timer() -> None:
            nonlocal update_timer
            if update_timer is None:
                return
            update_timer.cancel()
            update_timer = None

        def on_timer() -> None:
            nonlocal update_timer
            update_timer = None
            emit_output_update()

        def schedule_output_update() -> None:
            nonlocal update_dirty, update_timer
            if on_update is None:
                return
            update_dirty = True
            delay = BASH_UPDATE_THROTTLE_MS - (loop.time() * 1000 - last_update_at)
            if delay <= 0:
                clear_update_timer()
                emit_output_update()
                return
            if update_timer is None:
                update_timer = loop.call_later(delay / 1000, on_timer)

        def on_chunk(_chunk: str, get_progress: Callable[[], ShellCaptureProgress]) -> None:
            nonlocal latest_progress
            latest_progress = get_progress
            schedule_output_update()

        if on_update is not None:
            on_update(AgentToolResult(content=[], details=None))
        try:
            capture = get_or_throw(
                await execute_shell_with_capture(
                    env,
                    execution["command"],
                    {
                        "cwd": execution["cwd"],
                        "env": execution["env"],
                        "inherit_env": execution["inherit_env"],
                        "timeout": timeout,
                        "abort": abort,
                        "return_execution_errors": True,
                        "on_chunk": on_chunk,
                    },
                )
            )
            clear_update_timer()
            latest_progress = lambda: capture  # noqa: E731
            update_dirty = True
            emit_output_update()

            output_text = capture["output"]
            details: BashToolDetails | None = None
            if capture["truncation"]["truncated"]:
                details = {"truncation": capture["truncation"]}
                if capture.get("full_output_path"):
                    details["full_output_path"] = capture["full_output_path"]
                start_line = capture["truncation"]["total_lines"] - capture["truncation"]["output_lines"] + 1
                end_line = capture["truncation"]["total_lines"]
                if capture["truncation"]["last_line_partial"]:
                    last_line_size = format_size(capture["last_line_bytes"])
                    output_text += (
                        f"\n\n[Showing last {format_size(capture['truncation']['output_bytes'])} of line {end_line} "
                        f"(line is {last_line_size}). Full output: {capture.get('full_output_path')}]"
                    )
                elif capture["truncation"]["truncated_by"] == "lines":
                    output_text += (
                        f"\n\n[Showing lines {start_line}-{end_line} of {capture['truncation']['total_lines']}. "
                        f"Full output: {capture.get('full_output_path')}]"
                    )
                else:
                    output_text += (
                        f"\n\n[Showing lines {start_line}-{end_line} of {capture['truncation']['total_lines']} "
                        f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {capture.get('full_output_path')}]"
                    )

            def append_status(status: str) -> str:
                return f"{output_text}\n\n{status}" if output_text else status

            if capture.get("cancelled"):
                raise RuntimeError(append_status("Command aborted"))
            execution_error = capture.get("execution_error")
            if execution_error is not None and getattr(execution_error, "code", None) == "timeout":
                raise RuntimeError(append_status(f"Command timed out after {timeout} seconds"))
            if execution_error is not None:
                raise execution_error
            if capture.get("exit_code") not in (0, None):
                raise RuntimeError(append_status(f"Command exited with code {capture['exit_code']}"))
            return AgentToolResult(
                content=[TextContent(type="text", text=output_text or "(no output)")],
                details=details,
            )
        finally:
            clear_update_timer()

    return AgentHarnessTool(
        name="bash",
        label="bash",
        description=(
            f"Execute a bash command in the current working directory. Returns stdout and stderr. "
            f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES / 1024}KB "
            f"(whichever is hit first). If truncated, full output is saved to a temp file. "
            f"Optionally provide a timeout in seconds."
        ),
        parameters=BASH_SCHEMA,
        execute=execute,
    )
