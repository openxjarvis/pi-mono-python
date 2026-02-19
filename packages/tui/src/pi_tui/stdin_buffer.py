"""
StdinBuffer — mirrors packages/tui/src/stdin-buffer.ts

Buffers terminal input and emits complete sequences.
Handles partial escape sequences, bracketed paste mode, and sequence splitting.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Callable

ESC = "\x1b"
BRACKETED_PASTE_START = "\x1b[200~"
BRACKETED_PASTE_END = "\x1b[201~"


# ─────────────────────────────────────────────────────────────────────────────
# Sequence completeness detection
# ─────────────────────────────────────────────────────────────────────────────

def _is_complete_csi(data: str) -> str:
    """Returns 'complete', 'incomplete', or 'not-escape'."""
    if not data.startswith(ESC + "["):
        return "complete"
    if len(data) < 3:
        return "incomplete"
    payload = data[2:]
    last = payload[-1]
    code = ord(last)
    if 0x40 <= code <= 0x7e:
        if payload.startswith("<"):
            import re
            if re.match(r"^<\d+;\d+;\d+[Mm]$", payload):
                return "complete"
            if last in ("M", "m"):
                parts = payload[1:-1].split(";")
                if len(parts) == 3 and all(p.isdigit() for p in parts):
                    return "complete"
            return "incomplete"
        return "complete"
    return "incomplete"


def _is_complete_sequence(data: str) -> str:
    """Returns 'complete', 'incomplete', or 'not-escape'."""
    if not data.startswith(ESC):
        return "not-escape"
    if len(data) == 1:
        return "incomplete"
    after = data[1:]

    if after.startswith("["):
        if after.startswith("[M"):
            return "complete" if len(data) >= 6 else "incomplete"
        return _is_complete_csi(data)

    if after.startswith("]"):
        if data.endswith(ESC + "\\") or data.endswith("\x07"):
            return "complete"
        return "incomplete"

    if after.startswith("P"):
        if data.endswith(ESC + "\\"):
            return "complete"
        return "incomplete"

    if after.startswith("_"):
        if data.endswith(ESC + "\\"):
            return "complete"
        return "incomplete"

    if after.startswith("O"):
        return "complete" if len(after) >= 2 else "incomplete"

    if len(after) == 1:
        return "complete"

    return "complete"


def _extract_complete_sequences(buffer: str) -> tuple[list[str], str]:
    """
    Split buffer into complete sequences.
    Returns (sequences, remainder).
    """
    sequences: list[str] = []
    pos = 0

    while pos < len(buffer):
        remaining = buffer[pos:]

        if remaining.startswith(ESC):
            seq_end = 1
            while seq_end <= len(remaining):
                candidate = remaining[:seq_end]
                status = _is_complete_sequence(candidate)
                if status == "complete":
                    sequences.append(candidate)
                    pos += seq_end
                    break
                elif status == "incomplete":
                    seq_end += 1
                else:
                    sequences.append(candidate)
                    pos += seq_end
                    break
            else:
                # Ran off end — incomplete sequence
                return sequences, remaining
        else:
            sequences.append(remaining[0])
            pos += 1

    return sequences, ""


# ─────────────────────────────────────────────────────────────────────────────
# StdinBuffer
# ─────────────────────────────────────────────────────────────────────────────

class StdinBuffer:
    """
    Buffers stdin input and emits complete sequences via callbacks.
    Handles partial escape sequences that arrive across multiple chunks.
    Also handles bracketed paste mode.
    Mirrors StdinBuffer in stdin-buffer.ts.
    """

    def __init__(
        self,
        timeout_ms: int = 10,
        on_data: Callable[[str], None] | None = None,
        on_paste: Callable[[str], None] | None = None,
    ) -> None:
        self._timeout_ms = timeout_ms
        self._buffer = ""
        self._paste_mode = False
        self._paste_buffer = ""
        self._timer: threading.Timer | None = None
        self._on_data: list[Callable[[str], None]] = []
        self._on_paste: list[Callable[[str], None]] = []
        if on_data:
            self._on_data.append(on_data)
        if on_paste:
            self._on_paste.append(on_paste)

    def on(self, event: str, callback: Callable[[str], None]) -> None:
        """Register a callback for 'data' or 'paste' events."""
        if event == "data":
            self._on_data.append(callback)
        elif event == "paste":
            self._on_paste.append(callback)

    def _emit_data(self, seq: str) -> None:
        for cb in self._on_data:
            cb(seq)

    def _emit_paste(self, content: str) -> None:
        for cb in self._on_paste:
            cb(content)

    def _cancel_timer(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def process(self, data: str | bytes) -> None:
        """Feed input data into the buffer."""
        self._cancel_timer()

        # Handle bytes (high-byte conversion for compatibility)
        if isinstance(data, bytes):
            if len(data) == 1 and data[0] > 127:
                s = ESC + chr(data[0] - 128)
            else:
                s = data.decode("utf-8", errors="replace")
        else:
            s = data

        if not s and not self._buffer:
            self._emit_data("")
            return

        self._buffer += s

        if self._paste_mode:
            self._paste_buffer += self._buffer
            self._buffer = ""
            end_idx = self._paste_buffer.find(BRACKETED_PASTE_END)
            if end_idx != -1:
                pasted = self._paste_buffer[:end_idx]
                remaining = self._paste_buffer[end_idx + len(BRACKETED_PASTE_END):]
                self._paste_mode = False
                self._paste_buffer = ""
                self._emit_paste(pasted)
                if remaining:
                    self.process(remaining)
            return

        start_idx = self._buffer.find(BRACKETED_PASTE_START)
        if start_idx != -1:
            if start_idx > 0:
                before = self._buffer[:start_idx]
                seqs, _ = _extract_complete_sequences(before)
                for seq in seqs:
                    self._emit_data(seq)

            self._buffer = self._buffer[start_idx + len(BRACKETED_PASTE_START):]
            self._paste_mode = True
            self._paste_buffer = self._buffer
            self._buffer = ""

            end_idx = self._paste_buffer.find(BRACKETED_PASTE_END)
            if end_idx != -1:
                pasted = self._paste_buffer[:end_idx]
                remaining = self._paste_buffer[end_idx + len(BRACKETED_PASTE_END):]
                self._paste_mode = False
                self._paste_buffer = ""
                self._emit_paste(pasted)
                if remaining:
                    self.process(remaining)
            return

        seqs, remainder = _extract_complete_sequences(self._buffer)
        self._buffer = remainder

        for seq in seqs:
            self._emit_data(seq)

        if self._buffer:
            def _flush_timer():
                flushed = self.flush()
                for seq in flushed:
                    self._emit_data(seq)

            self._timer = threading.Timer(self._timeout_ms / 1000.0, _flush_timer)
            self._timer.daemon = True
            self._timer.start()

    def flush(self) -> list[str]:
        """Flush the buffer, returning any pending sequences."""
        self._cancel_timer()
        if not self._buffer:
            return []
        seqs = [self._buffer]
        self._buffer = ""
        return seqs

    def clear(self) -> None:
        """Clear buffer and cancel pending timer."""
        self._cancel_timer()
        self._buffer = ""
        self._paste_mode = False
        self._paste_buffer = ""

    def get_buffer(self) -> str:
        return self._buffer

    def destroy(self) -> None:
        self.clear()
