"""
Bash command execution with streaming support and cancellation.

Used by AgentSession.execute_bash() for interactive and RPC modes.

Mirrors core/bash-executor.ts
"""

from __future__ import annotations

import asyncio
import os
import re
import secrets
import signal
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable

from pi_coding_agent.core.tools.truncate import DEFAULT_MAX_BYTES, truncate_tail

_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_MAX_OUTPUT_BYTES = DEFAULT_MAX_BYTES * 2


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def _sanitize_binary(text: str) -> str:
    """Remove control characters except tab/newline/CR."""
    return _CTRL_CHARS.sub("", text)


def _get_shell_config() -> tuple[str, list[str]]:
    """Return (shell, args) for the current platform."""
    if sys.platform == "win32":
        return "cmd.exe", ["/c"]
    shell = os.environ.get("SHELL", "/bin/bash")
    return shell, ["-c"]


@dataclass
class BashResult:
    output: str
    exit_code: int | None
    cancelled: bool
    truncated: bool
    full_output_path: str | None = None


async def execute_bash(
    command: str,
    on_chunk: Callable[[str], None] | None = None,
    cancel_event: asyncio.Event | None = None,
) -> BashResult:
    """Execute a bash command with optional streaming and cancellation support.

    Features:
    - Streams sanitized output via on_chunk callback.
    - Writes large output to temp file for later retrieval.
    - Supports cancellation via asyncio.Event.
    - Sanitizes output (strips ANSI, removes binary garbage).
    - Truncates output if it exceeds DEFAULT_MAX_BYTES * 2.
    """
    shell, shell_args = _get_shell_config()

    env = dict(os.environ)
    # Ensure terminal-safe env for non-interactive commands
    env.setdefault("TERM", "dumb")

    process = await asyncio.create_subprocess_exec(
        shell,
        *shell_args,
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )

    output_chunks: list[str] = []
    output_bytes = 0
    temp_file_path: str | None = None
    temp_file = None
    total_bytes = 0
    cancelled = False

    async def _read_output() -> None:
        nonlocal output_bytes, temp_file_path, temp_file, total_bytes
        assert process.stdout
        while True:
            chunk = await process.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            sanitized = _sanitize_binary(_strip_ansi(text))
            total_bytes += len(sanitized.encode("utf-8"))

            # Write to temp file if large output
            if total_bytes > _MAX_OUTPUT_BYTES * 2:
                if temp_file is None:
                    fd, temp_file_path = tempfile.mkstemp(prefix="pi_bash_", suffix=".txt")
                    temp_file = os.fdopen(fd, "w", encoding="utf-8")
                temp_file.write(sanitized)

            # Keep in-memory buffer up to threshold
            if output_bytes <= _MAX_OUTPUT_BYTES:
                output_chunks.append(sanitized)
                output_bytes += len(sanitized.encode("utf-8"))

            if on_chunk:
                on_chunk(sanitized)

    async def _wait_for_cancel() -> None:
        nonlocal cancelled
        if cancel_event:
            await cancel_event.wait()
            cancelled = True
            try:
                if process.returncode is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

    try:
        tasks = [asyncio.ensure_future(_read_output())]
        if cancel_event:
            tasks.append(asyncio.ensure_future(_wait_for_cancel()))

        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED if cancel_event else asyncio.ALL_COMPLETED)

        if not cancelled:
            await process.wait()

    except Exception:
        cancelled = True
    finally:
        if temp_file:
            temp_file.close()

    output = "".join(output_chunks)
    truncated = output_bytes > DEFAULT_MAX_BYTES
    if truncated:
        output = truncate_tail(output, DEFAULT_MAX_BYTES)

    exit_code = process.returncode if not cancelled else None
    return BashResult(
        output=output,
        exit_code=exit_code,
        cancelled=cancelled,
        truncated=truncated,
        full_output_path=temp_file_path,
    )
