"""Child process helpers. Mirrors packages/coding-agent/src/utils/child-process.ts"""
from __future__ import annotations

import asyncio
from typing import Any


async def run_command(command: list[str], cwd: str | None = None, timeout: float | None = None) -> dict[str, Any]:
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        return {"stdout": "", "stderr": "timed out", "exitCode": 124}
    return {
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
        "exitCode": proc.returncode or 0,
    }
