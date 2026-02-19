"""
Terminal abstraction — mirrors packages/tui/src/terminal.ts

Provides:
- Terminal: abstract base class (interface)
- ProcessTerminal: real terminal using sys.stdin/sys.stdout + raw mode
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Callable

from .keys import set_kitty_protocol_active
from .stdin_buffer import StdinBuffer

# ─────────────────────────────────────────────────────────────────────────────
# Terminal ABC — mirrors Terminal interface in terminal.ts
# ─────────────────────────────────────────────────────────────────────────────

class Terminal(ABC):
    """
    Minimal terminal interface for TUI.
    Mirrors the Terminal interface in terminal.ts.
    """

    @abstractmethod
    def start(
        self,
        on_input: Callable[[str], None],
        on_resize: Callable[[], None],
    ) -> None:
        """Start the terminal with input and resize handlers."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the terminal and restore state."""

    @abstractmethod
    async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
        """
        Drain stdin before exiting to prevent Kitty key release events from
        leaking to the parent shell.
        """

    @abstractmethod
    def write(self, data: str) -> None:
        """Write output to the terminal."""

    @property
    @abstractmethod
    def columns(self) -> int:
        """Terminal width in columns."""

    @property
    @abstractmethod
    def rows(self) -> int:
        """Terminal height in rows."""

    @property
    @abstractmethod
    def kitty_protocol_active(self) -> bool:
        """Whether Kitty keyboard protocol is active."""

    @abstractmethod
    def move_by(self, lines: int) -> None:
        """Move cursor up (negative) or down (positive) by N lines."""

    @abstractmethod
    def hide_cursor(self) -> None:
        """Hide the cursor."""

    @abstractmethod
    def show_cursor(self) -> None:
        """Show the cursor."""

    @abstractmethod
    def clear_line(self) -> None:
        """Clear current line."""

    @abstractmethod
    def clear_from_cursor(self) -> None:
        """Clear from cursor to end of screen."""

    @abstractmethod
    def clear_screen(self) -> None:
        """Clear entire screen and move cursor to (0,0)."""

    @abstractmethod
    def set_title(self, title: str) -> None:
        """Set terminal window title."""


# ─────────────────────────────────────────────────────────────────────────────
# ProcessTerminal — mirrors ProcessTerminal in terminal.ts
# ─────────────────────────────────────────────────────────────────────────────

class ProcessTerminal(Terminal):
    """
    Real terminal using sys.stdin/sys.stdout.
    Enables raw mode, Kitty protocol detection, and bracketed paste.
    Mirrors ProcessTerminal in terminal.ts.
    """

    def __init__(self) -> None:
        self._was_raw = False
        self._input_handler: Callable[[str], None] | None = None
        self._resize_handler: Callable[[], None] | None = None
        self._kitty_protocol_active = False
        self._stdin_buffer: StdinBuffer | None = None
        self._write_log_path = os.environ.get("PI_TUI_WRITE_LOG", "")
        self._old_termios: object | None = None

    @property
    def kitty_protocol_active(self) -> bool:
        return self._kitty_protocol_active

    def start(
        self,
        on_input: Callable[[str], None],
        on_resize: Callable[[], None],
    ) -> None:
        self._input_handler = on_input
        self._resize_handler = on_resize

        self._enable_raw_mode()

        # Enable bracketed paste mode
        sys.stdout.write("\x1b[?2004h")
        sys.stdout.flush()

        # Set up SIGWINCH (resize) handler on Unix
        if hasattr(signal := __import__("signal"), "SIGWINCH"):
            import signal as _signal
            self._prev_sigwinch = _signal.signal(
                _signal.SIGWINCH,
                lambda *_: on_resize(),
            )

        # Query and enable Kitty keyboard protocol
        self._setup_stdin_buffer()
        sys.stdout.write("\x1b[?u")
        sys.stdout.flush()

    def _enable_raw_mode(self) -> None:
        """Put stdin in raw mode (no echo, no line buffering)."""
        try:
            import tty
            import termios
            fd = sys.stdin.fileno()
            self._old_termios = termios.tcgetattr(fd)
            tty.setraw(fd)
        except Exception:
            pass

    def _disable_raw_mode(self) -> None:
        try:
            import termios
            if self._old_termios is not None:
                fd = sys.stdin.fileno()
                termios.tcsetattr(fd, termios.TCSADRAIN, self._old_termios)
                self._old_termios = None
        except Exception:
            pass

    def _setup_stdin_buffer(self) -> None:
        """Set up StdinBuffer to split batched input into individual sequences."""
        self._stdin_buffer = StdinBuffer(timeout_ms=10)
        kitty_response_pattern = __import__("re").compile(r"^\x1b\[\?(\d+)u$")

        def on_data(sequence: str) -> None:
            if not self._kitty_protocol_active:
                m = kitty_response_pattern.match(sequence)
                if m:
                    self._kitty_protocol_active = True
                    set_kitty_protocol_active(True)
                    # Enable Kitty keyboard protocol flags 1+2+4
                    sys.stdout.write("\x1b[>7u")
                    sys.stdout.flush()
                    return  # Don't forward protocol response to TUI

            if self._input_handler:
                self._input_handler(sequence)

        def on_paste(content: str) -> None:
            if self._input_handler:
                self._input_handler(f"\x1b[200~{content}\x1b[201~")

        self._stdin_buffer.on("data", on_data)
        self._stdin_buffer.on("paste", on_paste)

        # Start reading stdin in a background thread
        import threading

        def _read_loop():
            import os as _os
            fd = sys.stdin.fileno()
            while self._stdin_buffer is not None:
                try:
                    import select
                    r, _, _ = select.select([fd], [], [], 0.05)
                    if r:
                        data = _os.read(fd, 1024)
                        if data:
                            buf = self._stdin_buffer
                            if buf:
                                buf.process(data)
                except (OSError, ValueError):
                    break

        t = threading.Thread(target=_read_loop, daemon=True)
        t.start()
        self._read_thread = t

    async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
        """Drain stdin before exiting."""
        if self._kitty_protocol_active:
            sys.stdout.write("\x1b[<u")
            sys.stdout.flush()
            self._kitty_protocol_active = False
            set_kitty_protocol_active(False)

        prev_handler = self._input_handler
        self._input_handler = None

        last_data_time = time.monotonic()
        end_time = last_data_time + max_ms / 1000.0

        async def _noop(_: str) -> None:
            nonlocal last_data_time
            last_data_time = time.monotonic()

        try:
            while True:
                now = time.monotonic()
                if now >= end_time:
                    break
                if now - last_data_time >= idle_ms / 1000.0:
                    break
                await asyncio.sleep(min(idle_ms / 1000.0, end_time - now))
        finally:
            self._input_handler = prev_handler

    def stop(self) -> None:
        """Disable bracketed paste, Kitty protocol, remove handlers, restore raw mode."""
        sys.stdout.write("\x1b[?2004l")

        if self._kitty_protocol_active:
            sys.stdout.write("\x1b[<u")
            self._kitty_protocol_active = False
            set_kitty_protocol_active(False)

        if self._stdin_buffer:
            self._stdin_buffer.destroy()
            self._stdin_buffer = None

        self._input_handler = None
        self._resize_handler = None

        # Restore SIGWINCH
        if hasattr(self, "_prev_sigwinch"):
            try:
                import signal as _signal
                _signal.signal(_signal.SIGWINCH, self._prev_sigwinch)
            except Exception:
                pass

        sys.stdout.flush()
        self._disable_raw_mode()

    def write(self, data: str) -> None:
        sys.stdout.write(data)
        sys.stdout.flush()
        if self._write_log_path:
            try:
                with open(self._write_log_path, "a", encoding="utf-8") as f:
                    f.write(data)
            except Exception:
                pass

    @property
    def columns(self) -> int:
        try:
            return os.get_terminal_size().columns
        except Exception:
            return int(os.environ.get("COLUMNS", "80"))

    @property
    def rows(self) -> int:
        try:
            return os.get_terminal_size().lines
        except Exception:
            return int(os.environ.get("LINES", "24"))

    def move_by(self, lines: int) -> None:
        if lines > 0:
            sys.stdout.write(f"\x1b[{lines}B")
        elif lines < 0:
            sys.stdout.write(f"\x1b[{-lines}A")
        sys.stdout.flush()

    def hide_cursor(self) -> None:
        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()

    def show_cursor(self) -> None:
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()

    def clear_line(self) -> None:
        sys.stdout.write("\x1b[K")
        sys.stdout.flush()

    def clear_from_cursor(self) -> None:
        sys.stdout.write("\x1b[J")
        sys.stdout.flush()

    def clear_screen(self) -> None:
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()

    def set_title(self, title: str) -> None:
        sys.stdout.write(f"\x1b]0;{title}\x07")
        sys.stdout.flush()
