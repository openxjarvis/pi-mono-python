"""Tests for pi_tui.keys — mirrors keys.ts tests"""
import pytest

from pi_tui.keys import KEY, matches_key, parse_key, set_kitty_protocol_active, is_key_release, is_key_repeat


class TestMatchesKey:
    def test_simple_key(self):
        assert matches_key("\r", "enter")

    def test_ctrl_c(self):
        assert matches_key("\x03", "ctrl+c")

    def test_escape(self):
        assert matches_key("\x1b", "escape")

    def test_backspace(self):
        assert matches_key("\x7f", "backspace")

    def test_up_arrow(self):
        assert matches_key("\x1b[A", "up")

    def test_down_arrow(self):
        assert matches_key("\x1b[B", "down")

    def test_left_arrow(self):
        assert matches_key("\x1b[D", "left")

    def test_right_arrow(self):
        assert matches_key("\x1b[C", "right")

    def test_no_match(self):
        assert not matches_key("a", "enter")

    def test_home_key(self):
        assert matches_key("\x1b[H", "home")

    def test_end_key(self):
        assert matches_key("\x1b[F", "end")

    def test_tab(self):
        assert matches_key("\t", "tab")


class TestParseKey:
    def test_parse_enter(self):
        key = parse_key("\r")
        assert key is not None
        assert "enter" in key.lower() or key == "\r"

    def test_parse_ctrl_c(self):
        key = parse_key("\x03")
        assert key is not None

    def test_parse_regular_char(self):
        key = parse_key("a")
        assert key is not None

    def test_parse_arrow(self):
        key = parse_key("\x1b[A")
        assert key is not None


class TestKeyConstants:
    def test_key_helper_attributes(self):
        assert KEY.enter == "enter"
        assert KEY.escape == "escape"
        assert KEY.tab == "tab"

    def test_key_is_string(self):
        assert isinstance(KEY.up, str)
        assert isinstance(KEY.down, str)


class TestKittyProtocol:
    def test_set_kitty_protocol(self):
        # Should not raise
        set_kitty_protocol_active(True)
        set_kitty_protocol_active(False)


class TestKeyRelease:
    def test_not_key_release_regular(self):
        assert not is_key_release("\r")
        assert not is_key_release("a")

    def test_key_release_kitty(self):
        # Kitty key release: CSI u with event type 3 (release)
        set_kitty_protocol_active(True)
        release_seq = "\x1b[97;1:3u"  # 'a' key release
        assert is_key_release(release_seq)
        set_kitty_protocol_active(False)


class TestKeyRepeat:
    def test_not_key_repeat_regular(self):
        assert not is_key_repeat("\r")

    def test_key_repeat_kitty(self):
        set_kitty_protocol_active(True)
        repeat_seq = "\x1b[97;1:2u"  # 'a' key repeat
        assert is_key_repeat(repeat_seq)
        set_kitty_protocol_active(False)
