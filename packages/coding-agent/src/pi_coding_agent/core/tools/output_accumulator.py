"""
Streaming output accumulator — mirrors packages/coding-agent/src/core/tools/output-accumulator.ts
"""
from __future__ import annotations

import os
import secrets
import tempfile
from dataclasses import dataclass
from typing import BinaryIO

from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, truncate_tail


@dataclass
class OutputAccumulatorOptions:
    max_lines: int = DEFAULT_MAX_LINES
    max_bytes: int = DEFAULT_MAX_BYTES
    temp_file_prefix: str = "pi-output"


@dataclass
class OutputSnapshot:
    content: str
    truncation: TruncationResult
    full_output_path: str | None = None


def _default_temp_file_path(prefix: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"{prefix}-{secrets.token_hex(8)}.log")


class OutputAccumulator:
    """Incrementally tracks streaming output with bounded memory."""

    def __init__(self, options: OutputAccumulatorOptions | None = None) -> None:
        opts = options or OutputAccumulatorOptions()
        self.max_lines = opts.max_lines
        self.max_bytes = opts.max_bytes
        self.max_rolling_bytes = max(self.max_bytes * 2, 1)
        self.temp_file_prefix = opts.temp_file_prefix
        self._tail_text = ""
        self._tail_bytes = 0
        self._tail_starts_at_line_boundary = True
        self._total_raw_bytes = 0
        self._total_decoded_bytes = 0
        self._completed_lines = 0
        self._total_lines = 0
        self._current_line_bytes = 0
        self._has_open_line = False
        self._finished = False
        self._raw_chunks: list[bytes] = []
        self._temp_file_path: str | None = None
        self._temp_file: BinaryIO | None = None

    def append(self, data: bytes | str) -> None:
        if self._finished:
            raise RuntimeError("Cannot append to a finished output accumulator")
        raw = data.encode("utf-8") if isinstance(data, str) else data
        self._total_raw_bytes += len(raw)
        text = raw.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else data
        self._append_decoded_text(text)
        if self._temp_file or self._should_use_temp_file():
            self._ensure_temp_file()
            if self._temp_file is not None:
                self._temp_file.write(raw)
        elif raw:
            self._raw_chunks.append(raw)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._should_use_temp_file():
            self._ensure_temp_file()

    def snapshot(self, persist_if_truncated: bool = False) -> OutputSnapshot:
        tail = truncate_tail(self._get_snapshot_text(), self.max_lines, self.max_bytes)
        truncated = self._total_lines > self.max_lines or self._total_decoded_bytes > self.max_bytes
        if persist_if_truncated and truncated:
            self._ensure_temp_file()
        return OutputSnapshot(
            content=tail.content,
            truncation=tail,
            full_output_path=self._temp_file_path,
        )

    async def close_temp_file(self) -> None:
        if self._temp_file is None:
            return
        self._temp_file.close()
        self._temp_file = None

    def get_last_line_bytes(self) -> int:
        return self._current_line_bytes

    def _append_decoded_text(self, text: str) -> None:
        if not text:
            return
        encoded = text.encode("utf-8")
        self._total_decoded_bytes += len(encoded)
        self._tail_text += text
        self._tail_bytes += len(encoded)
        if self._tail_bytes > self.max_rolling_bytes * 2:
            self._trim_tail()
        newlines = text.count("\n")
        if newlines == 0:
            self._current_line_bytes += len(encoded)
            self._has_open_line = True
        else:
            self._completed_lines += newlines
            tail = text.rsplit("\n", 1)[-1]
            self._current_line_bytes = len(tail.encode("utf-8"))
            self._has_open_line = bool(tail)
        self._total_lines = self._completed_lines + (1 if self._has_open_line else 0)

    def _trim_tail(self) -> None:
        buffer = self._tail_text.encode("utf-8")
        if len(buffer) <= self.max_rolling_bytes:
            self._tail_bytes = len(buffer)
            return
        start = len(buffer) - self.max_rolling_bytes
        while start < len(buffer) and (buffer[start] & 0xC0) == 0x80:
            start += 1
        self._tail_starts_at_line_boundary = start == 0 or buffer[start - 1] == 0x0A
        self._tail_text = buffer[start:].decode("utf-8", errors="ignore")
        self._tail_bytes = len(self._tail_text.encode("utf-8"))

    def _get_snapshot_text(self) -> str:
        if self._tail_starts_at_line_boundary:
            return self._tail_text
        first_newline = self._tail_text.find("\n")
        return self._tail_text if first_newline == -1 else self._tail_text[first_newline + 1 :]

    def _should_use_temp_file(self) -> bool:
        return (
            self._total_raw_bytes > self.max_bytes
            or self._total_decoded_bytes > self.max_bytes
            or self._total_lines > self.max_lines
        )

    def _ensure_temp_file(self) -> None:
        if self._temp_file_path:
            return
        self._temp_file_path = _default_temp_file_path(self.temp_file_prefix)
        self._temp_file = open(self._temp_file_path, "wb")
        for chunk in self._raw_chunks:
            self._temp_file.write(chunk)
        self._raw_chunks = []
