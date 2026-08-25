"""
Shared truncation utilities for tool outputs.

Mirrors harness/utils/truncate.ts.
"""
from __future__ import annotations

from typing import Literal, TypedDict

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024
GREP_MAX_LINE_LENGTH = 500

NON_ASCII_PATTERN = None


class TruncationResult(TypedDict):
    content: str
    truncated: bool
    truncated_by: Literal["lines", "bytes"] | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int


class TruncationOptions(TypedDict, total=False):
    max_lines: int
    max_bytes: int


def utf8_byte_length(content: str) -> int:
    return len(content.encode("utf-8"))


def split_lines_for_counting(content: str) -> list[str]:
    if len(content) == 0:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def replace_unpaired_surrogates(content: str) -> str:
    # Python str cannot hold unpaired UTF-16 surrogates the same way JS can,
    # but encode/decode replacement already maps invalid sequences. Keep the
    # walk so tests that pass lone-surrogate-like code units stay well-defined.
    output: list[str] = []
    i = 0
    while i < len(content):
        code = ord(content[i])
        if 0xD800 <= code <= 0xDBFF:
            if i + 1 < len(content):
                next_code = ord(content[i + 1])
                if 0xDC00 <= next_code <= 0xDFFF:
                    output.append(content[i])
                    output.append(content[i + 1])
                    i += 2
                    continue
            output.append("\ufffd")
            i += 1
        elif 0xDC00 <= code <= 0xDFFF:
            output.append("\ufffd")
            i += 1
        else:
            output.append(content[i])
            i += 1
    return "".join(output)


def format_size(bytes_count: int) -> str:
    if bytes_count < 1024:
        return f"{bytes_count}B"
    if bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f}KB"
    return f"{bytes_count / (1024 * 1024):.1f}MB"


def truncate_head(content: str, options: TruncationOptions | None = None) -> TruncationResult:
    options = options or {}
    max_lines = options.get("max_lines", DEFAULT_MAX_LINES)
    max_bytes = options.get("max_bytes", DEFAULT_MAX_BYTES)

    total_bytes = utf8_byte_length(content)
    lines = split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    first_line_bytes = utf8_byte_length(lines[0]) if lines else 0
    if first_line_bytes > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            last_line_partial=False,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines_arr: list[str] = []
    output_bytes_count = 0
    truncated_by: Literal["lines", "bytes"] = "lines"

    for i in range(min(len(lines), max_lines)):
        line = lines[i]
        line_bytes = utf8_byte_length(line) + (1 if i > 0 else 0)
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines_arr.append(line)
        output_bytes_count += line_bytes

    if len(output_lines_arr) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines_arr)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines_arr),
        output_bytes=utf8_byte_length(output_content),
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    start = len(encoded) - max_bytes
    while start < len(encoded) and (encoded[start] & 0xC0) == 0x80:
        start += 1
    return encoded[start:].decode("utf-8")


def truncate_tail(content: str, options: TruncationOptions | None = None) -> TruncationResult:
    options = options or {}
    max_lines = options.get("max_lines", DEFAULT_MAX_LINES)
    max_bytes = options.get("max_bytes", DEFAULT_MAX_BYTES)

    total_bytes = utf8_byte_length(content)
    lines = split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines_arr: list[str] = []
    output_bytes_count = 0
    truncated_by: Literal["lines", "bytes"] = "lines"
    last_line_partial = False

    i = len(lines) - 1
    while i >= 0 and len(output_lines_arr) < max_lines:
        line = lines[i]
        line_bytes = utf8_byte_length(line) + (1 if output_lines_arr else 0)
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            if len(output_lines_arr) == 0:
                truncated_line = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines_arr.insert(0, truncated_line)
                output_bytes_count = utf8_byte_length(truncated_line)
                last_line_partial = True
            break
        output_lines_arr.insert(0, line)
        output_bytes_count += line_bytes
        i -= 1

    if len(output_lines_arr) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines_arr)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines_arr),
        output_bytes=utf8_byte_length(output_content),
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> dict[str, object]:
    if len(line) <= max_chars:
        return {"text": line, "was_truncated": False}
    return {"text": f"{line[:max_chars]}... [truncated]", "was_truncated": True}
