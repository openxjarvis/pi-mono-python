"""Shell output capture. Mirrors packages/agent/src/harness/utils/shell-output.ts"""
from __future__ import annotations

from typing import Any

from pi_agent.harness.types import ExecutionEnv, ExecutionError, ShellExecOptions
from pi_agent.harness.utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, truncate_tail

ShellCaptureProgress = dict[str, Any]


async def execute_shell_with_capture(
    env: ExecutionEnv, command: str, options: ShellExecOptions | None = None
) -> dict[str, Any]:
    return await capture_shell_output(env, command, options)


def sanitize_binary_output(text: str) -> str:
    kept: list[str] = []
    for char in text:
        code = ord(char)
        if code in (0x09, 0x0A, 0x0D) or (code > 0x1F and not (0xFFF9 <= code <= 0xFFFB)):
            kept.append(char)
    return "".join(kept)


async def capture_shell_output(
    env: ExecutionEnv, command: str, options: ShellExecOptions | None = None
) -> dict[str, Any]:
    result = await env.exec(command, options)
    if not (result.get("ok") if isinstance(result, dict) else getattr(result, "ok", False)):
        error = result.get("error") if isinstance(result, dict) else getattr(result, "error", None)
        return {
            "output": "",
            "exitCode": None,
            "cancelled": getattr(error, "code", "") == "aborted",
            "truncated": False,
            "executionError": error,
        }
    value = result.value
    raw = sanitize_binary_output((value.get("stdout") or "") + (value.get("stderr") or ""))
    truncation = truncate_tail(raw, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES)
    return {
        "output": truncation.text,
        "exitCode": value.get("exitCode"),
        "cancelled": False,
        "truncated": truncation.truncated,
        "truncation": truncation,
        "fullOutputPath": None,
        "lastLineBytes": len(truncation.text.encode()),
    }
