"""
Text truncation utilities — mirrors packages/coding-agent/src/core/tools/truncate.ts
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 30 * 1024  # 30 KB
GREP_MAX_LINE_LENGTH = 500


@dataclass
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: str | None  # "lines" or "bytes"
    output_lines: int
    total_lines: int
    output_bytes: int
    first_line_exceeds_limit: bool = False
    last_line_partial: bool = False


def format_size(bytes_: int) -> str:
    if bytes_ < 1024:
        return f"{bytes_}B"
    kb = bytes_ / 1024
    if kb < 1024:
        return f"{kb:.0f}KB"
    mb = kb / 1024
    return f"{mb:.1f}MB"


def truncate_head(
    text: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """
    Truncate text from the head (start), keeping the beginning.
    Mirrors truncateHead() in TypeScript.
    """
    lines = text.split("\n")
    total_lines = len(lines)
    selected_lines: list[str] = []
    current_bytes = 0
    truncated_by: str | None = None

    # Check first line for byte-exceeds
    if lines and len(lines[0].encode("utf-8")) > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            output_lines=0,
            total_lines=total_lines,
            output_bytes=0,
            first_line_exceeds_limit=True,
        )

    for i, line in enumerate(lines):
        if len(selected_lines) >= max_lines:
            truncated_by = "lines"
            break
        line_bytes = len(line.encode("utf-8")) + 1  # +1 for newline
        if current_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        selected_lines.append(line)
        current_bytes += line_bytes

    truncated = truncated_by is not None
    content = "\n".join(selected_lines)

    return TruncationResult(
        content=content,
        truncated=truncated,
        truncated_by=truncated_by,
        output_lines=len(selected_lines),
        total_lines=total_lines,
        output_bytes=current_bytes,
    )


def truncate_tail(
    text: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """
    Truncate text from the tail (end), keeping the end.
    Mirrors truncateTail() in TypeScript.
    """
    lines = text.split("\n")
    total_lines = len(lines)

    # Work from the end
    selected_lines: list[str] = []
    current_bytes = 0
    truncated_by: str | None = None
    last_line_partial = False

    # Check last line
    if lines and len(lines[-1].encode("utf-8")) > max_bytes:
        last_line_partial = True
        # Truncate the last line itself
        last_line = lines[-1]
        truncated_last = last_line.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
        return TruncationResult(
            content=truncated_last,
            truncated=True,
            truncated_by="bytes",
            output_lines=1,
            total_lines=total_lines,
            output_bytes=len(truncated_last.encode("utf-8")),
            last_line_partial=True,
        )

    for line in reversed(lines):
        if len(selected_lines) >= max_lines:
            truncated_by = "lines"
            break
        line_bytes = len(line.encode("utf-8")) + 1
        if current_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        selected_lines.insert(0, line)
        current_bytes += line_bytes

    truncated = truncated_by is not None
    content = "\n".join(selected_lines)

    return TruncationResult(
        content=content,
        truncated=truncated,
        truncated_by=truncated_by,
        output_lines=len(selected_lines),
        total_lines=total_lines,
        output_bytes=current_bytes,
    )


def truncate_line(line: str, max_length: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    """Truncate a single line to max_length. Returns (truncated_text, was_truncated)."""
    if len(line) <= max_length:
        return line, False
    return line[:max_length] + "...", True
