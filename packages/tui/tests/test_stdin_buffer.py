"""Tests for pi_tui.stdin_buffer"""
import pytest

from pi_tui.stdin_buffer import StdinBuffer


class TestStdinBuffer:
    def test_basic_char_emitted(self):
        received = []
        buf = StdinBuffer()
        buf.on("data", received.append)
        buf.process(b"a")
        buf.flush()
        assert "a" in received

    def test_complete_escape_sequence(self):
        received = []
        buf = StdinBuffer()
        buf.on("data", received.append)
        buf.process(b"\x1b[A")  # up arrow
        buf.flush()
        assert "\x1b[A" in received

    def test_bracketed_paste(self):
        pasted = []
        buf = StdinBuffer()
        buf.on("paste", pasted.append)
        buf.process(b"\x1b[200~hello\x1b[201~")
        buf.flush()
        assert pasted
        assert "hello" in pasted[0]

    def test_multiple_chars(self):
        received = []
        buf = StdinBuffer()
        buf.on("data", received.append)
        buf.process(b"abc")
        buf.flush()
        combined = "".join(received)
        assert "a" in combined
        assert "b" in combined
        assert "c" in combined
