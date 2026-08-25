"""
Strict JSONL framing — mirrors packages/coding-agent/src/modes/rpc/jsonl.ts

Splits on LF (\\n) only, preserving U+2028/U+2029 and other Unicode separators
that appear inside JSON strings. Python's readline() is LF-safe but this module
adds explicit error handling for malformed lines and proper UTF-8 boundary
handling on raw byte streams.
"""
from __future__ import annotations

import json
from typing import Any, BinaryIO, Callable, Iterator


def serialize_json_line(value: Any) -> str:
    """Serialize a value as a single strict JSONL record (LF terminated)."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def serialize_json_line_bytes(value: Any) -> bytes:
    return serialize_json_line(value).encode("utf-8")


def parse_json_line(line: str) -> Any | None:
    """Parse a single JSONL line. Returns None for blank/malformed lines."""
    stripped = line.rstrip("\r\n")
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def iter_json_lines(stream: BinaryIO) -> Iterator[Any]:
    """Iterate parsed JSON objects from a byte stream using strict LF framing.

    Handles UTF-8 multi-byte characters that may be split across read
    boundaries by accumulating a line buffer and splitting only on 0x0A.
    """
    buffer = b""
    while True:
        chunk = stream.read(8192)
        if not chunk:
            break
        buffer += chunk
        while b"\n" in buffer:
            line_bytes, buffer = buffer.split(b"\n", 1)
            try:
                text = line_bytes.decode("utf-8").rstrip("\r")
            except UnicodeDecodeError:
                continue
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError:
                continue

    if buffer:
        try:
            text = buffer.decode("utf-8").rstrip("\r\n")
            if text:
                yield json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass


class JsonlLineReader:
    """Stateful JSONL line reader that can be fed chunks incrementally.

    Mirrors TS attachJsonlLineReader: splits on LF only, strips trailing CR,
    and handles UTF-8 boundaries correctly.
    """

    def __init__(self, on_line: Callable[[str], None]) -> None:
        self._on_line = on_line
        self._buffer = ""

    def feed(self, data: str | bytes) -> None:
        """Feed a chunk of data (str or bytes). Complete lines are emitted."""
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
        else:
            text = data
        self._buffer += text
        while "\n" in self._buffer:
            idx = self._buffer.index("\n")
            line = self._buffer[:idx]
            self._buffer = self._buffer[idx + 1:]
            self._on_line(line.rstrip("\r"))

    def flush(self) -> None:
        """Emit any remaining partial line in the buffer."""
        if self._buffer:
            self._on_line(self._buffer.rstrip("\r"))
            self._buffer = ""
