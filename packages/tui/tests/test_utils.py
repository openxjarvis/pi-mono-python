"""Tests for pi_tui.utils — mirrors utils.ts tests"""
import pytest

from pi_tui.utils import (
    AnsiCodeTracker,
    apply_background_to_line,
    truncate_to_width,
    visible_width,
    wrap_text_with_ansi,
    is_whitespace_char,
    is_punctuation_char,
    slice_by_column,
)


class TestVisibleWidth:
    def test_ascii(self):
        assert visible_width("hello") == 5

    def test_empty(self):
        assert visible_width("") == 0

    def test_ansi_stripped(self):
        assert visible_width("\x1b[31mhello\x1b[0m") == 5

    def test_unicode_cjk(self):
        # CJK characters are double-width
        assert visible_width("中文") == 4

    def test_newline_not_counted(self):
        # Newline: width should be 0 (control char)
        assert visible_width("\n") == 0

    def test_spaces(self):
        assert visible_width("   ") == 3


class TestTruncateToWidth:
    def test_short_string_unchanged(self):
        assert truncate_to_width("hello", 10) == "hello"

    def test_truncates_with_ellipsis(self):
        result = truncate_to_width("hello world", 8)
        assert visible_width(result) <= 8
        assert "..." in result

    def test_empty_ellipsis(self):
        result = truncate_to_width("hello world", 8, "")
        assert visible_width(result) <= 8

    def test_exactly_fits(self):
        result = truncate_to_width("hello", 5)
        assert result == "hello"

    def test_zero_width(self):
        result = truncate_to_width("hello", 0)
        assert visible_width(result) == 0


class TestWrapTextWithAnsi:
    def test_short_line_unchanged(self):
        lines = wrap_text_with_ansi("hello", 20)
        assert lines == ["hello"]

    def test_wraps_at_word_boundary(self):
        lines = wrap_text_with_ansi("hello world foo", 8)
        assert len(lines) > 1
        for ln in lines:
            assert visible_width(ln) <= 8

    def test_preserves_ansi_across_wraps(self):
        text = "\x1b[31mhello world and more text here\x1b[0m"
        lines = wrap_text_with_ansi(text, 10)
        assert len(lines) > 1

    def test_empty_string(self):
        lines = wrap_text_with_ansi("", 20)
        assert lines == [""]

    def test_multiline_input(self):
        lines = wrap_text_with_ansi("line1\nline2\nline3", 80)
        assert len(lines) == 3

    def test_long_word_forced_break(self):
        word = "a" * 20
        lines = wrap_text_with_ansi(word, 10)
        for ln in lines:
            assert visible_width(ln) <= 10


class TestAnsiCodeTracker:
    def test_reset_on_empty(self):
        tracker = AnsiCodeTracker()
        assert tracker.get_active_codes() == ""

    def test_tracks_bold(self):
        tracker = AnsiCodeTracker()
        tracker.process("\x1b[1m")
        codes = tracker.get_active_codes()
        assert "1" in codes

    def test_resets_on_sgr_0(self):
        tracker = AnsiCodeTracker()
        tracker.process("\x1b[1m")
        tracker.process("\x1b[0m")
        assert tracker.get_active_codes() == ""

    def test_tracks_color(self):
        tracker = AnsiCodeTracker()
        tracker.process("\x1b[31m")
        codes = tracker.get_active_codes()
        assert "31" in codes

    def test_non_m_sequence_ignored(self):
        tracker = AnsiCodeTracker()
        tracker.process("\x1b[2J")  # clear screen
        assert tracker.get_active_codes() == ""

    def test_has_active_codes(self):
        tracker = AnsiCodeTracker()
        assert not tracker.has_active_codes()
        tracker.process("\x1b[1m")
        assert tracker.has_active_codes()


class TestApplyBackgroundToLine:
    def test_pads_to_width(self):
        result = apply_background_to_line("hi", 10, lambda x: x)
        assert visible_width(result) >= 10

    def test_applies_bg_fn(self):
        calls = []
        def bg(x: str) -> str:
            calls.append(x)
            return f"\x1b[42m{x}\x1b[0m"
        apply_background_to_line("hi", 5, bg)
        assert calls


class TestSliceByColumn:
    def test_basic_slice(self):
        result = slice_by_column("hello world", 0, 5)
        assert result == "hello"

    def test_slice_middle(self):
        result = slice_by_column("hello world", 6, 5)
        assert result == "world"


class TestCharUtils:
    def test_whitespace(self):
        assert is_whitespace_char(" ")
        assert is_whitespace_char("\t")
        assert not is_whitespace_char("a")

    def test_punctuation(self):
        assert is_punctuation_char(".")
        assert is_punctuation_char("!")
        assert not is_punctuation_char("a")
        assert not is_punctuation_char(" ")
