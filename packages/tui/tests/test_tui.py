"""Tests for pi_tui.tui — core TUI class"""
import asyncio
import pytest
from unittest.mock import MagicMock, patch

from pi_tui.tui import TUI, CURSOR_MARKER


class MockTerminal:
    """Mock terminal for testing TUI without real terminal."""
    def __init__(self):
        self.rows = 24
        self.columns = 80
        self._output: list[str] = []
        self.on_input = None
        self.on_resize = None

    def write(self, text: str) -> None:
        self._output.append(text)

    def hide_cursor(self) -> None:
        pass

    def show_cursor(self) -> None:
        pass

    def move_cursor(self, row: int, col: int) -> None:
        pass

    def clear_line(self) -> None:
        pass

    def clear_to_end_of_screen(self) -> None:
        pass

    def set_title(self, title: str) -> None:
        pass

    def enable_kitty_protocol(self) -> None:
        pass

    def disable_kitty_protocol(self) -> None:
        pass

    def enable_bracketed_paste(self) -> None:
        pass

    def disable_bracketed_paste(self) -> None:
        pass

    def set_raw_mode(self, enabled: bool) -> None:
        pass

    def get_output(self) -> str:
        return "".join(self._output)

    def clear_output(self) -> None:
        self._output.clear()


class AsyncMockTerminal(MockTerminal):
    """Terminal mock with start/stop API used by TUI.start()."""

    kitty_protocol_active = False

    def start(self, on_input, on_resize) -> None:
        self.on_input = on_input
        self.on_resize = on_resize

    def stop(self) -> None:
        pass

    async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
        return

    def move_by(self, lines: int) -> None:
        pass

    def clear_from_cursor(self) -> None:
        pass

    def clear_screen(self) -> None:
        pass


class TestCursorMarker:
    def test_cursor_marker_constant(self):
        assert CURSOR_MARKER == "\x1b_pi:c\x07"


class TestTUIBasic:
    def _make_tui(self):
        terminal = MockTerminal()
        tui = TUI(terminal)
        return tui, terminal

    def test_tui_creates(self):
        tui, terminal = self._make_tui()
        assert tui is not None

    def test_tui_has_terminal(self):
        tui, terminal = self._make_tui()
        assert tui.terminal is terminal

    def test_add_component(self):
        from pi_tui.components.spacer import Spacer
        tui, terminal = self._make_tui()
        spacer = Spacer()
        tui.add_child(spacer)
        assert spacer in tui.children

    def test_remove_component(self):
        from pi_tui.components.spacer import Spacer
        tui, terminal = self._make_tui()
        spacer = Spacer()
        tui.add_child(spacer)
        tui.remove_child(spacer)
        assert spacer not in tui.children

    def test_request_render_does_not_crash(self):
        tui, terminal = self._make_tui()
        tui.request_render()  # Should not raise

    def test_render_produces_output(self):
        from pi_tui.components.text import Text
        tui, terminal = self._make_tui()
        t = Text("hello world", padding_x=0, padding_y=0)
        tui.add_child(t)
        # Manually trigger render
        tui._do_render()
        output = terminal.get_output()
        # Should have written something
        assert len(output) > 0

    def test_focus_management(self):
        from pi_tui.components.input import Input
        tui, terminal = self._make_tui()
        inp = Input()
        tui.add_child(inp)
        tui.set_focus(inp)
        assert inp.focused is True

    def test_overlay_show_hide(self):
        from pi_tui.tui import OverlayOptions
        tui, terminal = self._make_tui()
        from pi_tui.components.spacer import Spacer
        overlay = Spacer(3)
        handle = tui.show_overlay(overlay)
        assert tui.has_overlay()
        handle.hide()
        tui.hide_overlay()


@pytest.mark.asyncio
async def test_start_captures_running_event_loop():
    from pi_tui.components.text import Text

    terminal = AsyncMockTerminal()
    tui = TUI(terminal)
    tui.add_child(Text("hello", padding_x=0, padding_y=0))
    tui.start()
    await asyncio.sleep(0)
    assert tui._main_loop is asyncio.get_running_loop()
    tui.stop()


@pytest.mark.asyncio
async def test_request_render_schedules_render_tick_on_running_loop(monkeypatch):
    terminal = AsyncMockTerminal()
    tui = TUI(terminal)
    tui._main_loop = asyncio.get_running_loop()

    called = asyncio.Event()
    original_tick = tui._render_tick

    def wrapped_tick():
        called.set()
        original_tick()

    monkeypatch.setattr(tui, "_render_tick", wrapped_tick)
    tui.request_render()
    await asyncio.wait_for(called.wait(), timeout=1.0)
