"""
Stdout takeover helpers — mirrors packages/coding-agent/src/core/output-guard.ts
"""
from __future__ import annotations

import sys
from typing import Any, TextIO

_original_stdout_write = None
_taken_over = False
_raw_stdout: TextIO | None = None


def take_over_stdout() -> None:
    """Redirect accidental stdout writes to stderr so JSON/RPC streams stay clean."""
    global _original_stdout_write, _taken_over, _raw_stdout
    if _taken_over:
        return
    _raw_stdout = sys.stdout
    _original_stdout_write = sys.stdout.write

    def _write(chunk: Any) -> int:
        text = chunk if isinstance(chunk, str) else str(chunk)
        return sys.stderr.write(text)

    sys.stdout.write = _write  # type: ignore[method-assign]
    _taken_over = True


def restore_stdout() -> None:
    global _original_stdout_write, _taken_over
    if not _taken_over:
        return
    if _original_stdout_write is not None:
        sys.stdout.write = _original_stdout_write  # type: ignore[method-assign]
    _original_stdout_write = None
    _taken_over = False


def is_stdout_taken_over() -> bool:
    return _taken_over


def write_raw_stdout(text: str) -> None:
    if not text:
        return
    target = _raw_stdout or sys.stdout
    if _original_stdout_write is not None:
        _original_stdout_write(text)
        if hasattr(target, "flush"):
            target.flush()
        return
    target.write(text)
    target.flush()


async def wait_for_raw_stdout_backpressure() -> None:
    return None


async def flush_raw_stdout() -> None:
    target = _raw_stdout or sys.stdout
    if hasattr(target, "flush"):
        target.flush()
