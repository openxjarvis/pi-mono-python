"""Input component — mirrors components/input.ts"""
from __future__ import annotations

from ..keybindings import get_keybindings
from ..kill_ring import KillRing
from ..tui import CURSOR_MARKER
from ..undo_stack import UndoStack
from ..utils import _segment_graphemes, is_whitespace_char, visible_width, slice_by_column
from ..word_navigation import find_word_backward, find_word_forward


class _InputState:
    __slots__ = ("value", "cursor")

    def __init__(self, value: str, cursor: int) -> None:
        self.value = value
        self.cursor = cursor

    def __copy__(self) -> "_InputState":
        return _InputState(self.value, self.cursor)

    def copy(self) -> "_InputState":
        return _InputState(self.value, self.cursor)


class Input:
    """
    Single-line text input with horizontal scrolling.
    Mirrors Input in components/input.ts.
    """

    def __init__(self) -> None:
        self._value = ""
        self._cursor = 0

        self.focused = False
        self.on_submit: "((value: str) -> None) | None" = None  # type: ignore[assignment]
        self.on_escape: "(() -> None) | None" = None  # type: ignore[assignment]

        self._paste_buffer = ""
        self._is_in_paste = False

        self._kill_ring = KillRing()
        self._last_action: str | None = None  # "kill", "yank", "type-word"

        self._undo_stack: UndoStack[_InputState] = UndoStack()

    def get_value(self) -> str:
        return self._value

    def set_value(self, value: str) -> None:
        self._value = value
        self._cursor = min(self._cursor, len(value))

    def invalidate(self) -> None:
        pass

    def handle_input(self, data: str) -> None:
        # Bracketed paste: start
        if "\x1b[200~" in data:
            self._is_in_paste = True
            self._paste_buffer = ""
            data = data.replace("\x1b[200~", "")

        if self._is_in_paste:
            self._paste_buffer += data
            end_idx = self._paste_buffer.find("\x1b[201~")
            if end_idx != -1:
                paste_content = self._paste_buffer[:end_idx]
                self._handle_paste(paste_content)
                self._is_in_paste = False
                remaining = self._paste_buffer[end_idx + 6:]
                self._paste_buffer = ""
                if remaining:
                    self.handle_input(remaining)
            return

        kb = get_keybindings()

        if kb.matches(data, "tui.select.cancel"):
            if self.on_escape:
                self.on_escape()
            return

        if kb.matches(data, "tui.editor.undo"):
            self._undo()
            return

        if kb.matches(data, "tui.input.submit") or data == "\n":
            if self.on_submit:
                self.on_submit(self._value)
            return

        if kb.matches(data, "tui.editor.deleteCharBackward"):
            self._handle_backspace()
            return

        if kb.matches(data, "tui.editor.deleteCharForward"):
            self._handle_forward_delete()
            return

        if kb.matches(data, "tui.editor.deleteWordBackward"):
            self._delete_word_backwards()
            return

        if kb.matches(data, "tui.editor.deleteWordForward"):
            self._delete_word_forward()
            return

        if kb.matches(data, "tui.editor.deleteToLineStart"):
            self._delete_to_line_start()
            return

        if kb.matches(data, "tui.editor.deleteToLineEnd"):
            self._delete_to_line_end()
            return

        if kb.matches(data, "tui.editor.yank"):
            self._yank()
            return

        if kb.matches(data, "tui.editor.yankPop"):
            self._yank_pop()
            return

        if kb.matches(data, "tui.editor.cursorLeft"):
            self._last_action = None
            if self._cursor > 0:
                before = self._value[:self._cursor]
                graphemes = _segment_graphemes(before)
                if graphemes:
                    self._cursor -= len(graphemes[-1])
            return

        if kb.matches(data, "tui.editor.cursorRight"):
            self._last_action = None
            if self._cursor < len(self._value):
                after = self._value[self._cursor:]
                graphemes = _segment_graphemes(after)
                if graphemes:
                    self._cursor += len(graphemes[0])
            return

        if kb.matches(data, "tui.editor.cursorLineStart"):
            self._last_action = None
            self._cursor = 0
            return

        if kb.matches(data, "tui.editor.cursorLineEnd"):
            self._last_action = None
            self._cursor = len(self._value)
            return

        if kb.matches(data, "tui.editor.cursorWordLeft"):
            self._move_word_backwards()
            return

        if kb.matches(data, "tui.editor.cursorWordRight"):
            self._move_word_forwards()
            return

        # Kitty CSI-u printable decode (for terminals with flag 1, e.g. VS Code)
        # Must be checked before the control-char guard because CSI-u contains ESC
        from pi_tui.keys import decode_kitty_printable
        kitty_ch = decode_kitty_printable(data)
        if kitty_ch is not None:
            self._insert_character(kitty_ch)
            return

        # Regular printable characters
        has_control = any(
            ord(ch) < 32 or ord(ch) == 0x7F or (0x80 <= ord(ch) <= 0x9F)
            for ch in data
        )
        if not has_control:
            self._insert_character(data)

    def _insert_character(self, char: str) -> None:
        if is_whitespace_char(char) or self._last_action != "type-word":
            self._push_undo()
        self._last_action = "type-word"
        self._value = self._value[:self._cursor] + char + self._value[self._cursor:]
        self._cursor += len(char)

    def _handle_backspace(self) -> None:
        self._last_action = None
        if self._cursor > 0:
            self._push_undo()
            before = self._value[:self._cursor]
            graphemes = _segment_graphemes(before)
            length = len(graphemes[-1]) if graphemes else 1
            self._value = self._value[:self._cursor - length] + self._value[self._cursor:]
            self._cursor -= length

    def _handle_forward_delete(self) -> None:
        self._last_action = None
        if self._cursor < len(self._value):
            self._push_undo()
            after = self._value[self._cursor:]
            graphemes = _segment_graphemes(after)
            length = len(graphemes[0]) if graphemes else 1
            self._value = self._value[:self._cursor] + self._value[self._cursor + length:]

    def _delete_to_line_start(self) -> None:
        if self._cursor == 0:
            return
        self._push_undo()
        deleted = self._value[:self._cursor]
        self._kill_ring.push(deleted, prepend=True, accumulate=(self._last_action == "kill"))
        self._last_action = "kill"
        self._value = self._value[self._cursor:]
        self._cursor = 0

    def _delete_to_line_end(self) -> None:
        if self._cursor >= len(self._value):
            return
        self._push_undo()
        deleted = self._value[self._cursor:]
        self._kill_ring.push(deleted, prepend=False, accumulate=(self._last_action == "kill"))
        self._last_action = "kill"
        self._value = self._value[:self._cursor]

    def _delete_word_backwards(self) -> None:
        if self._cursor == 0:
            return
        was_kill = self._last_action == "kill"
        self._push_undo()
        old_cursor = self._cursor
        self._move_word_backwards()
        delete_from = self._cursor
        self._cursor = old_cursor
        deleted = self._value[delete_from:self._cursor]
        self._kill_ring.push(deleted, prepend=True, accumulate=was_kill)
        self._last_action = "kill"
        self._value = self._value[:delete_from] + self._value[self._cursor:]
        self._cursor = delete_from

    def _delete_word_forward(self) -> None:
        if self._cursor >= len(self._value):
            return
        was_kill = self._last_action == "kill"
        self._push_undo()
        old_cursor = self._cursor
        self._move_word_forwards()
        delete_to = self._cursor
        self._cursor = old_cursor
        deleted = self._value[self._cursor:delete_to]
        self._kill_ring.push(deleted, prepend=False, accumulate=was_kill)
        self._last_action = "kill"
        self._value = self._value[:self._cursor] + self._value[delete_to:]

    def _yank(self) -> None:
        text = self._kill_ring.peek()
        if not text:
            return
        self._push_undo()
        self._value = self._value[:self._cursor] + text + self._value[self._cursor:]
        self._cursor += len(text)
        self._last_action = "yank"

    def _yank_pop(self) -> None:
        if self._last_action != "yank" or self._kill_ring.length <= 1:
            return
        self._push_undo()
        prev_text = self._kill_ring.peek() or ""
        self._value = self._value[:self._cursor - len(prev_text)] + self._value[self._cursor:]
        self._cursor -= len(prev_text)
        self._kill_ring.rotate()
        text = self._kill_ring.peek() or ""
        self._value = self._value[:self._cursor] + text + self._value[self._cursor:]
        self._cursor += len(text)
        self._last_action = "yank"

    def _push_undo(self) -> None:
        self._undo_stack.push(_InputState(self._value, self._cursor))

    def _undo(self) -> None:
        snapshot = self._undo_stack.pop()
        if snapshot is None:
            return
        self._value = snapshot.value
        self._cursor = snapshot.cursor
        self._last_action = None

    def _move_word_backwards(self) -> None:
        if self._cursor == 0:
            return
        self._last_action = None
        self._cursor = find_word_backward(self._value, self._cursor)

    def _move_word_forwards(self) -> None:
        if self._cursor >= len(self._value):
            return
        self._last_action = None
        self._cursor = find_word_forward(self._value, self._cursor)

    def _handle_paste(self, pasted_text: str) -> None:
        self._last_action = None
        self._push_undo()
        # Strip newlines first, then expand tabs to 4 spaces (mirrors input.ts)
        clean = pasted_text.replace("\r\n", "").replace("\r", "").replace("\n", "").replace("\t", "    ")
        self._value = self._value[:self._cursor] + clean + self._value[self._cursor:]
        self._cursor += len(clean)

    def render(self, width: int) -> list[str]:
        prompt = "> "
        available = width - len(prompt)

        if available <= 0:
            return [prompt]

        value = self._value
        cursor = self._cursor

        visible_text = ""
        cursor_display = cursor
        total_width = visible_width(value)  # Use visible_width instead of len

        if total_width < available:
            # Everything fits
            visible_text = value
        else:
            # Need horizontal scrolling
            scroll_width = available - 1 if cursor == len(value) else available
            cursor_col = visible_width(value[:cursor])  # Use visible_width for cursor position

            if scroll_width > 0:
                half_width = scroll_width // 2
                start_col = 0

                if cursor_col < half_width:
                    # Cursor near start
                    start_col = 0
                elif cursor_col > total_width - half_width:
                    # Cursor near end
                    start_col = max(0, total_width - scroll_width)
                else:
                    # Cursor in middle
                    start_col = max(0, cursor_col - half_width)

                # Use slice_by_column instead of string slicing
                visible_text = slice_by_column(value, start_col, scroll_width, True)
                before_cursor = slice_by_column(value, start_col, max(0, cursor_col - start_col), True)
                cursor_display = len(before_cursor)
            else:
                visible_text = ""
                cursor_display = 0

        graphemes_after = _segment_graphemes(visible_text[cursor_display:])
        cursor_grapheme = graphemes_after[0] if graphemes_after else None

        before_cursor = visible_text[:cursor_display]
        at_cursor = cursor_grapheme if cursor_grapheme else " "
        after_len = len(at_cursor) if cursor_grapheme else 0
        after_cursor = visible_text[cursor_display + after_len:]

        marker = CURSOR_MARKER if self.focused else ""
        cursor_char = f"\x1b[7m{at_cursor}\x1b[27m"
        text_with_cursor = before_cursor + marker + cursor_char + after_cursor

        visual_length = visible_width(text_with_cursor)
        padding = " " * max(0, available - visual_length)
        line = prompt + text_with_cursor + padding

        return [line]
