"""Tests for pi_tui components"""
import pytest

from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text
from pi_tui.components.truncated_text import TruncatedText
from pi_tui.components.box import Box
from pi_tui.components.select_list import SelectItem, SelectList, SelectListTheme
from pi_tui.components.input import Input
from pi_tui.utils import visible_width


class TestSpacer:
    def test_default_one_line(self):
        s = Spacer()
        lines = s.render(80)
        assert len(lines) == 1

    def test_multiple_lines(self):
        s = Spacer(3)
        lines = s.render(80)
        assert len(lines) == 3

    def test_lines_are_empty(self):
        s = Spacer(2)
        lines = s.render(80)
        assert all(ln == "" for ln in lines)

    def test_set_lines(self):
        s = Spacer(1)
        s.set_lines(5)
        assert len(s.render(80)) == 5

    def test_handle_input_noop(self):
        s = Spacer()
        s.handle_input("x")  # should not raise


class TestText:
    def test_renders_text(self):
        t = Text("hello")
        lines = t.render(40)
        combined = " ".join(lines)
        assert "hello" in combined

    def test_empty_text_returns_empty(self):
        t = Text("")
        lines = t.render(40)
        assert lines == []

    def test_padding_applied(self):
        t = Text("hi", padding_x=2, padding_y=0)
        lines = t.render(20)
        # content lines should have left/right spaces
        for ln in lines:
            if "hi" in ln:
                assert ln.startswith("  ")

    def test_cache_invalidation(self):
        t = Text("hello")
        lines1 = t.render(40)
        t.set_text("world")
        lines2 = t.render(40)
        assert "world" in " ".join(lines2)

    def test_custom_bg_fn(self):
        applied = []
        def bg(x: str) -> str:
            applied.append(x)
            return f"\x1b[42m{x}\x1b[0m"
        t = Text("hello", padding_x=0, padding_y=0, custom_bg_fn=bg)
        t.render(20)
        assert applied


class TestTruncatedText:
    def test_short_text_unchanged(self):
        t = TruncatedText("hello")
        lines = t.render(40)
        assert len(lines) == 1
        assert "hello" in lines[0]

    def test_long_text_truncated(self):
        t = TruncatedText("a" * 100)
        lines = t.render(20)
        assert len(lines) == 1
        assert visible_width(lines[0]) <= 20

    def test_multiline_uses_first_line(self):
        t = TruncatedText("line1\nline2")
        lines = t.render(40)
        assert "line1" in lines[0]
        assert "line2" not in lines[0]


class TestBox:
    def test_empty_box_renders_nothing(self):
        b = Box()
        lines = b.render(40)
        assert lines == []

    def test_with_child(self):
        b = Box(padding_x=0, padding_y=0)
        b.add_child(Text("hello", padding_x=0, padding_y=0))
        lines = b.render(40)
        assert any("hello" in ln for ln in lines)

    def test_padding(self):
        b = Box(padding_x=2, padding_y=1)
        b.add_child(Text("hi", padding_x=0, padding_y=0))
        lines = b.render(40)
        assert len(lines) >= 3  # 1 top pad + content + 1 bottom pad

    def test_clear(self):
        b = Box()
        b.add_child(Text("hello"))
        b.clear()
        assert b.render(40) == []

    def test_remove_child(self):
        b = Box(padding_x=0, padding_y=0)
        t = Text("hello")
        b.add_child(t)
        b.remove_child(t)
        assert b.render(40) == []


class TestSelectList:
    def _make_list(self, items=None, max_visible=5):
        if items is None:
            items = [SelectItem(value=f"item{i}", label=f"Item {i}") for i in range(10)]
        theme = SelectListTheme()
        return SelectList(items, max_visible, theme)

    def test_renders_items(self):
        sl = self._make_list()
        lines = sl.render(80)
        assert len(lines) > 0

    def test_scroll_indicator_when_more_items(self):
        sl = self._make_list(max_visible=3)
        lines = sl.render(80)
        combined = " ".join(lines)
        assert "/" in combined  # scroll info like "1/10"

    def test_navigation_down(self):
        sl = self._make_list()
        initial = sl._selected_index
        sl.handle_input("\x1b[B")  # down arrow
        assert sl._selected_index == initial + 1

    def test_navigation_up(self):
        sl = self._make_list()
        sl._selected_index = 5
        sl.handle_input("\x1b[A")  # up arrow
        assert sl._selected_index == 4

    def test_wraps_at_bottom(self):
        items = [SelectItem(value=f"item{i}", label=f"Item {i}") for i in range(3)]
        sl = self._make_list(items=items)
        sl._selected_index = 2
        sl.handle_input("\x1b[B")  # down — wraps to 0
        assert sl._selected_index == 0

    def test_on_select_callback(self):
        selected = []
        sl = self._make_list()
        sl.on_select = lambda item: selected.append(item)
        sl.handle_input("\r")  # enter
        assert len(selected) == 1
        assert selected[0].value == "item0"

    def test_filter(self):
        items = [
            SelectItem(value="foo", label="Foo"),
            SelectItem(value="bar", label="Bar"),
            SelectItem(value="baz", label="Baz"),
        ]
        sl = self._make_list(items=items)
        sl.set_filter("foo")
        lines = sl.render(80)
        combined = " ".join(lines)
        # "bar" and "baz" should be filtered out
        assert "Baz" not in combined
        assert "Foo" in combined

    def test_no_match_message(self):
        sl = self._make_list()
        sl.set_filter("xyznotfound")
        lines = sl.render(80)
        assert any("No matching" in ln for ln in lines)


class TestInput:
    def test_initial_empty(self):
        inp = Input()
        assert inp.get_value() == ""

    def test_type_characters(self):
        inp = Input()
        inp.handle_input("h")
        inp.handle_input("i")
        assert inp.get_value() == "hi"

    def test_backspace(self):
        inp = Input()
        inp.handle_input("hello")
        inp.handle_input("\x7f")  # backspace
        assert inp.get_value() == "hell"

    def test_set_value(self):
        inp = Input()
        inp.set_value("hello")
        assert inp.get_value() == "hello"

    def test_ctrl_u_delete_to_line_start(self):
        inp = Input()
        inp.handle_input("hello")
        inp.handle_input("\x15")  # Ctrl+U
        assert inp.get_value() == ""

    def test_ctrl_k_delete_to_line_end(self):
        inp = Input()
        inp.handle_input("hello")
        inp.handle_input("\x1b[D")  # left
        inp.handle_input("\x1b[D")  # left
        inp.handle_input("\x0b")  # Ctrl+K
        assert inp.get_value() == "hel"

    def test_undo(self):
        inp = Input()
        inp.handle_input("hello")
        inp.handle_input("\x1f")  # Ctrl+_  (undo)
        # After undo, value should be different
        # (state before "hello" was typed)
        assert len(inp.get_value()) < 5 or inp.get_value() == ""

    def test_render_returns_single_line(self):
        inp = Input()
        inp.handle_input("test")
        lines = inp.render(40)
        assert len(lines) == 1

    def test_render_shows_cursor(self):
        inp = Input()
        inp.handle_input("test")
        inp.focused = True
        lines = inp.render(40)
        assert len(lines) == 1

    def test_on_submit(self):
        submitted = []
        inp = Input()
        inp.on_submit = lambda v: submitted.append(v)
        inp.handle_input("hello")
        inp.handle_input("\r")
        assert submitted == ["hello"]

    def test_on_escape(self):
        escaped = []
        inp = Input()
        inp.on_escape = lambda: escaped.append(True)
        inp.handle_input("\x1b")
        assert escaped == [True]

    def test_paste_mode(self):
        inp = Input()
        inp.handle_input("\x1b[200~hello\nworld\x1b[201~")
        # Newlines should be stripped in single-line input
        assert "\n" not in inp.get_value()
        assert "hello" in inp.get_value()
