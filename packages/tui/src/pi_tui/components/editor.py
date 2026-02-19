"""Editor component — mirrors components/editor.ts"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ..autocomplete import AutocompleteProvider
from ..keybindings import get_editor_keybindings
from ..keys import matches_key
from ..kill_ring import KillRing
from ..tui import CURSOR_MARKER
from ..undo_stack import UndoStack
from ..utils import _segment_graphemes, is_punctuation_char, is_whitespace_char, visible_width
from .select_list import SelectItem, SelectList, SelectListTheme

if TYPE_CHECKING:
    from ..tui import TUI


# ─── Kitty CSI-u printable key decoder ───────────────────────────────────────
_KITTY_CSI_U_RE = re.compile(r"^\x1b\[(\d+)(?::(\d*))?(?::(\d+))?(?:;(\d+))?(?::(\d+))?u$")
_KITTY_MOD_SHIFT = 1
_KITTY_MOD_ALT = 2
_KITTY_MOD_CTRL = 4


def _decode_kitty_printable(data: str) -> str | None:
    m = _KITTY_CSI_U_RE.match(data)
    if not m:
        return None
    codepoint = int(m.group(1) or "0")
    shifted_key = int(m.group(2)) if m.group(2) else None
    mod_value = int(m.group(4)) if m.group(4) else 1
    modifier = mod_value - 1 if mod_value else 0
    if modifier & (_KITTY_MOD_ALT | _KITTY_MOD_CTRL):
        return None
    effective = codepoint
    if (modifier & _KITTY_MOD_SHIFT) and shifted_key is not None:
        effective = shifted_key
    if effective < 32:
        return None
    try:
        return chr(effective)
    except (ValueError, OverflowError):
        return None


# ─── Editor state / layout ────────────────────────────────────────────────────

@dataclass
class _EditorState:
    lines: list[str]
    cursor_line: int
    cursor_col: int

    def copy(self) -> "_EditorState":
        return _EditorState(list(self.lines), self.cursor_line, self.cursor_col)


@dataclass
class _LayoutLine:
    text: str
    has_cursor: bool
    cursor_pos: int | None = None


@dataclass
class _VisualLine:
    logical_line: int
    start_col: int
    length: int


@dataclass
class TextChunk:
    text: str
    start_index: int
    end_index: int


@dataclass
class EditorTheme:
    border_color: Callable[[str], str] = lambda x: x  # noqa: E731
    select_list: SelectListTheme = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.select_list is None:
            self.select_list = SelectListTheme()


@dataclass
class EditorOptions:
    padding_x: int = 0
    autocomplete_max_visible: int = 5


def word_wrap_line(line: str, max_width: int) -> list[TextChunk]:
    """
    Split a line into word-wrapped chunks tracking original positions.
    Mirrors wordWrapLine() in editor.ts.
    """
    if not line or max_width <= 0:
        return [TextChunk("", 0, 0)]

    line_vis_w = visible_width(line)
    if line_vis_w <= max_width:
        return [TextChunk(line, 0, len(line))]

    chunks: list[TextChunk] = []
    graphemes = _segment_graphemes(line)

    current_width = 0
    chunk_start = 0
    wrap_opp_idx = -1
    wrap_opp_width = 0

    for i, g in enumerate(graphemes):
        g_width = visible_width(g)
        char_index = sum(len(graphemes[j]) for j in range(i))
        is_ws = is_whitespace_char(g)

        if current_width + g_width > max_width:
            if wrap_opp_idx >= 0:
                chunks.append(TextChunk(line[chunk_start:wrap_opp_idx], chunk_start, wrap_opp_idx))
                chunk_start = wrap_opp_idx
                current_width -= wrap_opp_width
            elif chunk_start < char_index:
                chunks.append(TextChunk(line[chunk_start:char_index], chunk_start, char_index))
                chunk_start = char_index
                current_width = 0
            wrap_opp_idx = -1

        current_width += g_width

        next_g = graphemes[i + 1] if i + 1 < len(graphemes) else None
        if is_ws and next_g and not is_whitespace_char(next_g):
            wrap_opp_idx = sum(len(graphemes[j]) for j in range(i + 1))
            wrap_opp_width = current_width

    chunks.append(TextChunk(line[chunk_start:], chunk_start, len(line)))
    return chunks


class Editor:
    """
    Multi-line text editor with word-wrap, autocomplete, history, kill ring, undo.
    Mirrors Editor in components/editor.ts.
    """

    def __init__(
        self,
        tui: "TUI",
        theme: EditorTheme,
        options: EditorOptions | None = None,
    ) -> None:
        self._tui = tui
        self._theme = theme
        opts = options or EditorOptions()
        self._padding_x = max(0, opts.padding_x)
        self._autocomplete_max_visible = max(3, min(20, opts.autocomplete_max_visible))

        self._state = _EditorState([""], 0, 0)
        self.focused = False
        self._last_width = 80
        self._scroll_offset = 0
        self.border_color: Callable[[str], str] = theme.border_color

        self._autocomplete_provider: AutocompleteProvider | None = None
        self._autocomplete_list: SelectList | None = None
        self._autocomplete_state: str | None = None  # "regular" | "force" | None
        self._autocomplete_prefix = ""

        self._pastes: dict[int, str] = {}
        self._paste_counter = 0
        self._paste_buffer = ""
        self._is_in_paste = False

        self._history: list[str] = []
        self._history_index = -1

        self._kill_ring = KillRing()
        self._last_action: str | None = None  # "kill" | "yank" | "type-word"

        self._jump_mode: str | None = None  # "forward" | "backward"
        self._preferred_visual_col: int | None = None

        self._undo_stack: UndoStack[_EditorState] = UndoStack()

        self.on_submit: Callable[[str], None] | None = None
        self.on_change: Callable[[str], None] | None = None
        self.disable_submit = False

    # ── Public API ──────────────────────────────────────────────────────────────

    def get_padding_x(self) -> int:
        return self._padding_x

    def set_padding_x(self, padding: int) -> None:
        new_padding = max(0, int(padding))
        if self._padding_x != new_padding:
            self._padding_x = new_padding
            self._tui.request_render()

    def set_autocomplete_provider(self, provider: AutocompleteProvider) -> None:
        self._autocomplete_provider = provider

    def add_to_history(self, text: str) -> None:
        trimmed = text.strip()
        if not trimmed:
            return
        if self._history and self._history[0] == trimmed:
            return
        self._history.insert(0, trimmed)
        if len(self._history) > 100:
            self._history.pop()

    def get_text(self) -> str:
        return "\n".join(self._state.lines)

    def get_expanded_text(self) -> str:
        result = self.get_text()
        for paste_id, paste_content in self._pastes.items():
            pattern = re.compile(rf"\[paste #{paste_id}( (\+\d+ lines|\d+ chars))?\]")
            result = pattern.sub(paste_content, result)
        return result

    def get_lines(self) -> list[str]:
        return list(self._state.lines)

    def get_cursor(self) -> dict[str, int]:
        return {"line": self._state.cursor_line, "col": self._state.cursor_col}

    def set_text(self, text: str) -> None:
        self._last_action = None
        self._history_index = -1
        if self.get_text() != text:
            self._push_undo()
        self._set_text_internal(text)

    def insert_text_at_cursor(self, text: str) -> None:
        if not text:
            return
        self._push_undo()
        self._last_action = None
        self._history_index = -1
        self._insert_text_at_cursor_internal(text)

    def is_showing_autocomplete(self) -> bool:
        return self._autocomplete_state is not None

    def invalidate(self) -> None:
        pass

    # ── Render ──────────────────────────────────────────────────────────────────

    def render(self, width: int) -> list[str]:
        max_padding = max(0, (width - 1) // 2)
        padding_x = min(self._padding_x, max_padding)
        content_width = max(1, width - padding_x * 2)
        layout_width = max(1, content_width - (0 if padding_x else 1))
        self._last_width = layout_width

        horizontal = self.border_color("─")

        layout_lines = self._layout_text(layout_width)

        terminal_rows = self._tui.terminal.rows if self._tui.terminal else 24
        max_visible_lines = max(5, int(terminal_rows * 0.3))

        cursor_line_idx = next((i for i, ll in enumerate(layout_lines) if ll.has_cursor), 0)
        if cursor_line_idx < self._scroll_offset:
            self._scroll_offset = cursor_line_idx
        elif cursor_line_idx >= self._scroll_offset + max_visible_lines:
            self._scroll_offset = cursor_line_idx - max_visible_lines + 1

        max_scroll = max(0, len(layout_lines) - max_visible_lines)
        self._scroll_offset = max(0, min(self._scroll_offset, max_scroll))

        visible_lines = layout_lines[self._scroll_offset:self._scroll_offset + max_visible_lines]
        result: list[str] = []
        left_pad = " " * padding_x
        right_pad = " " * padding_x

        # Top border
        if self._scroll_offset > 0:
            indicator = f"─── ↑ {self._scroll_offset} more "
            remaining = width - visible_width(indicator)
            result.append(self.border_color(indicator + "─" * max(0, remaining)))
        else:
            result.append(horizontal * width)

        emit_cursor_marker = self.focused and not self._autocomplete_state
        segmenter = _segment_graphemes

        for ll in visible_lines:
            display_text = ll.text
            line_vis_w = visible_width(display_text)
            cursor_in_padding = False

            if ll.has_cursor and ll.cursor_pos is not None:
                before = display_text[:ll.cursor_pos]
                after = display_text[ll.cursor_pos:]
                marker = CURSOR_MARKER if emit_cursor_marker else ""

                if after:
                    after_graphemes = segmenter(after)
                    first_g = after_graphemes[0] if after_graphemes else ""
                    rest_after = after[len(first_g):]
                    cursor = f"\x1b[7m{first_g}\x1b[0m"
                    display_text = before + marker + cursor + rest_after
                else:
                    cursor = "\x1b[7m \x1b[0m"
                    display_text = before + marker + cursor
                    line_vis_w += 1
                    if line_vis_w > content_width and padding_x > 0:
                        cursor_in_padding = True

            padding = " " * max(0, content_width - visible_width(display_text))
            line_right_pad = right_pad[1:] if cursor_in_padding else right_pad
            result.append(f"{left_pad}{display_text}{padding}{line_right_pad}")

        # Bottom border
        lines_below = len(layout_lines) - (self._scroll_offset + len(visible_lines))
        if lines_below > 0:
            indicator = f"─── ↓ {lines_below} more "
            remaining = width - visible_width(indicator)
            result.append(self.border_color(indicator + "─" * max(0, remaining)))
        else:
            result.append(horizontal * width)

        # Autocomplete
        if self._autocomplete_state and self._autocomplete_list:
            for ln in self._autocomplete_list.render(content_width):
                lw = visible_width(ln)
                lpad = " " * max(0, content_width - lw)
                result.append(f"{left_pad}{ln}{lpad}{right_pad}")

        return result

    # ── Input handling ──────────────────────────────────────────────────────────

    def handle_input(self, data: str) -> None:
        kb = get_editor_keybindings()

        if self._jump_mode is not None:
            if kb.matches(data, "jumpForward") or kb.matches(data, "jumpBackward"):
                self._jump_mode = None
                return
            if data and ord(data[0]) >= 32:
                direction = self._jump_mode
                self._jump_mode = None
                self._jump_to_char(data, direction)
                return
            self._jump_mode = None

        if "\x1b[200~" in data:
            self._is_in_paste = True
            self._paste_buffer = ""
            data = data.replace("\x1b[200~", "")

        if self._is_in_paste:
            self._paste_buffer += data
            end_idx = self._paste_buffer.find("\x1b[201~")
            if end_idx != -1:
                paste_content = self._paste_buffer[:end_idx]
                if paste_content:
                    self._handle_paste(paste_content)
                self._is_in_paste = False
                remaining = self._paste_buffer[end_idx + 6:]
                self._paste_buffer = ""
                if remaining:
                    self.handle_input(remaining)
            return

        if kb.matches(data, "copy"):
            return

        if kb.matches(data, "undo"):
            self._undo()
            return

        # Autocomplete navigation
        if self._autocomplete_state and self._autocomplete_list:
            if kb.matches(data, "selectCancel"):
                self._cancel_autocomplete()
                return
            if kb.matches(data, "selectUp") or kb.matches(data, "selectDown"):
                self._autocomplete_list.handle_input(data)
                return
            if kb.matches(data, "tab"):
                selected = self._autocomplete_list.get_selected_item()
                if selected and self._autocomplete_provider:
                    self._apply_autocomplete(selected)
                    self._cancel_autocomplete()
                return
            if kb.matches(data, "selectConfirm"):
                selected = self._autocomplete_list.get_selected_item()
                if selected and self._autocomplete_provider:
                    prefix = self._autocomplete_prefix
                    self._apply_autocomplete(selected)
                    if prefix.startswith("/"):
                        self._cancel_autocomplete()
                        # fall through to submit
                    else:
                        self._cancel_autocomplete()
                        if self.on_change:
                            self.on_change(self.get_text())
                        return

        if kb.matches(data, "tab") and not self._autocomplete_state:
            self._handle_tab_completion()
            return

        if kb.matches(data, "deleteToLineEnd"):
            self._delete_to_end_of_line()
            return
        if kb.matches(data, "deleteToLineStart"):
            self._delete_to_start_of_line()
            return
        if kb.matches(data, "deleteWordBackward"):
            self._delete_word_backwards()
            return
        if kb.matches(data, "deleteWordForward"):
            self._delete_word_forward()
            return
        if kb.matches(data, "deleteCharBackward") or matches_key(data, "shift+backspace"):
            self._handle_backspace()
            return
        if kb.matches(data, "deleteCharForward") or matches_key(data, "shift+delete"):
            self._handle_forward_delete()
            return

        if kb.matches(data, "yank"):
            self._yank()
            return
        if kb.matches(data, "yankPop"):
            self._yank_pop()
            return

        if kb.matches(data, "cursorLineStart"):
            self._move_to_line_start()
            return
        if kb.matches(data, "cursorLineEnd"):
            self._move_to_line_end()
            return
        if kb.matches(data, "cursorWordLeft"):
            self._move_word_backwards()
            return
        if kb.matches(data, "cursorWordRight"):
            self._move_word_forwards()
            return

        # New line
        is_new_line = (
            kb.matches(data, "newLine") or
            (len(data) > 1 and data.charCodeAt(0) == 10 if hasattr(data, "charCodeAt") else False) or
            data == "\x1b\r" or
            data == "\x1b[13;2~" or
            (len(data) == 1 and data == "\n")
        )
        if is_new_line:
            if self._should_submit_on_backslash_enter(data, kb):
                self._handle_backspace()
                self._submit_value()
                return
            self._add_new_line()
            return

        if kb.matches(data, "submit"):
            if self.disable_submit:
                return
            current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
            if self._state.cursor_col > 0 and current_line and current_line[self._state.cursor_col - 1] == "\\":
                self._handle_backspace()
                self._add_new_line()
                return
            self._submit_value()
            return

        if kb.matches(data, "cursorUp"):
            if self._is_editor_empty():
                self._navigate_history(-1)
            elif self._history_index > -1 and self._is_on_first_visual_line():
                self._navigate_history(-1)
            elif self._is_on_first_visual_line():
                self._move_to_line_start()
            else:
                self._move_cursor(-1, 0)
            return
        if kb.matches(data, "cursorDown"):
            if self._history_index > -1 and self._is_on_last_visual_line():
                self._navigate_history(1)
            elif self._is_on_last_visual_line():
                self._move_to_line_end()
            else:
                self._move_cursor(1, 0)
            return
        if kb.matches(data, "cursorRight"):
            self._move_cursor(0, 1)
            return
        if kb.matches(data, "cursorLeft"):
            self._move_cursor(0, -1)
            return

        if kb.matches(data, "pageUp"):
            self._page_scroll(-1)
            return
        if kb.matches(data, "pageDown"):
            self._page_scroll(1)
            return

        if kb.matches(data, "jumpForward"):
            self._jump_mode = "forward"
            return
        if kb.matches(data, "jumpBackward"):
            self._jump_mode = "backward"
            return

        if matches_key(data, "shift+space"):
            self._insert_character(" ")
            return

        kitty_printable = _decode_kitty_printable(data)
        if kitty_printable is not None:
            self._insert_character(kitty_printable)
            return

        if data and ord(data[0]) >= 32:
            self._insert_character(data)

    # ── Private helpers ──────────────────────────────────────────────────────────

    def _is_editor_empty(self) -> bool:
        return len(self._state.lines) == 1 and self._state.lines[0] == ""

    def _is_on_first_visual_line(self) -> bool:
        vis = self._build_visual_line_map(self._last_width)
        return self._find_current_visual_line(vis) == 0

    def _is_on_last_visual_line(self) -> bool:
        vis = self._build_visual_line_map(self._last_width)
        return self._find_current_visual_line(vis) == len(vis) - 1

    def _navigate_history(self, direction: int) -> None:
        self._last_action = None
        if not self._history:
            return
        new_index = self._history_index - direction
        if new_index < -1 or new_index >= len(self._history):
            return
        if self._history_index == -1 and new_index >= 0:
            self._push_undo()
        self._history_index = new_index
        if self._history_index == -1:
            self._set_text_internal("")
        else:
            self._set_text_internal(self._history[self._history_index])

    def _set_text_internal(self, text: str) -> None:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")
        self._state.lines = lines if lines else [""]
        self._state.cursor_line = len(self._state.lines) - 1
        self._set_cursor_col(len(self._state.lines[-1]) if self._state.lines else 0)
        self._scroll_offset = 0
        if self.on_change:
            self.on_change(self.get_text())

    def _set_cursor_col(self, col: int) -> None:
        self._state.cursor_col = col
        self._preferred_visual_col = None

    def _layout_text(self, content_width: int) -> list[_LayoutLine]:
        layout_lines: list[_LayoutLine] = []

        if not self._state.lines or (len(self._state.lines) == 1 and self._state.lines[0] == ""):
            layout_lines.append(_LayoutLine("", True, 0))
            return layout_lines

        for i, line in enumerate(self._state.lines):
            is_current = i == self._state.cursor_line
            line_vis_w = visible_width(line)

            if line_vis_w <= content_width:
                if is_current:
                    layout_lines.append(_LayoutLine(line, True, self._state.cursor_col))
                else:
                    layout_lines.append(_LayoutLine(line, False))
            else:
                chunks = word_wrap_line(line, content_width)
                for ci, chunk in enumerate(chunks):
                    is_last = ci == len(chunks) - 1
                    if is_current:
                        cursor_pos = self._state.cursor_col
                        if is_last:
                            has_cursor = cursor_pos >= chunk.start_index
                            adj_pos = cursor_pos - chunk.start_index
                        else:
                            has_cursor = chunk.start_index <= cursor_pos < chunk.end_index
                            adj_pos = min(cursor_pos - chunk.start_index, len(chunk.text))
                        if has_cursor:
                            layout_lines.append(_LayoutLine(chunk.text, True, adj_pos))
                        else:
                            layout_lines.append(_LayoutLine(chunk.text, False))
                    else:
                        layout_lines.append(_LayoutLine(chunk.text, False))

        return layout_lines

    def _build_visual_line_map(self, width: int) -> list[_VisualLine]:
        visual_lines: list[_VisualLine] = []
        for i, line in enumerate(self._state.lines):
            if not line:
                visual_lines.append(_VisualLine(i, 0, 0))
            elif visible_width(line) <= width:
                visual_lines.append(_VisualLine(i, 0, len(line)))
            else:
                chunks = word_wrap_line(line, width)
                for chunk in chunks:
                    visual_lines.append(_VisualLine(i, chunk.start_index, chunk.end_index - chunk.start_index))
        return visual_lines

    def _find_current_visual_line(self, visual_lines: list[_VisualLine]) -> int:
        for i, vl in enumerate(visual_lines):
            if vl.logical_line == self._state.cursor_line:
                col_in_seg = self._state.cursor_col - vl.start_col
                is_last = i == len(visual_lines) - 1 or visual_lines[i + 1].logical_line != vl.logical_line
                if col_in_seg >= 0 and (col_in_seg < vl.length or (is_last and col_in_seg <= vl.length)):
                    return i
        return len(visual_lines) - 1

    def _compute_vertical_move_column(
        self, current_visual_col: int, source_max: int, target_max: int
    ) -> int:
        has_preferred = self._preferred_visual_col is not None
        cursor_in_middle = current_visual_col < source_max
        target_too_short = target_max < current_visual_col

        if not has_preferred or cursor_in_middle:
            if target_too_short:
                self._preferred_visual_col = current_visual_col
                return target_max
            self._preferred_visual_col = None
            return current_visual_col

        target_cant_fit = target_max < self._preferred_visual_col  # type: ignore[operator]
        if target_too_short or target_cant_fit:
            return target_max

        result = self._preferred_visual_col  # type: ignore[assignment]
        self._preferred_visual_col = None
        return result  # type: ignore[return-value]

    def _move_to_visual_line(
        self,
        visual_lines: list[_VisualLine],
        current_vi: int,
        target_vi: int,
    ) -> None:
        current_vl = visual_lines[current_vi]
        target_vl = visual_lines[target_vi]

        current_vis_col = self._state.cursor_col - current_vl.start_col

        is_last_src = current_vi == len(visual_lines) - 1 or visual_lines[current_vi + 1].logical_line != current_vl.logical_line
        src_max = current_vl.length if is_last_src else max(0, current_vl.length - 1)

        is_last_tgt = target_vi == len(visual_lines) - 1 or visual_lines[target_vi + 1].logical_line != target_vl.logical_line
        tgt_max = target_vl.length if is_last_tgt else max(0, target_vl.length - 1)

        move_col = self._compute_vertical_move_column(current_vis_col, src_max, tgt_max)

        self._state.cursor_line = target_vl.logical_line
        target_col = target_vl.start_col + move_col
        logical_line = self._state.lines[target_vl.logical_line] if self._state.lines else ""
        self._state.cursor_col = min(target_col, len(logical_line))

    def _move_cursor(self, delta_line: int, delta_col: int) -> None:
        self._last_action = None
        visual_lines = self._build_visual_line_map(self._last_width)
        current_vi = self._find_current_visual_line(visual_lines)

        if delta_line != 0:
            target_vi = current_vi + delta_line
            if 0 <= target_vi < len(visual_lines):
                self._move_to_visual_line(visual_lines, current_vi, target_vi)

        if delta_col != 0:
            current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
            if delta_col > 0:
                if self._state.cursor_col < len(current_line):
                    after = current_line[self._state.cursor_col:]
                    graphemes = _segment_graphemes(after)
                    self._set_cursor_col(self._state.cursor_col + len(graphemes[0]) if graphemes else self._state.cursor_col + 1)
                elif self._state.cursor_line < len(self._state.lines) - 1:
                    self._state.cursor_line += 1
                    self._set_cursor_col(0)
            else:
                if self._state.cursor_col > 0:
                    before = current_line[:self._state.cursor_col]
                    graphemes = _segment_graphemes(before)
                    self._set_cursor_col(self._state.cursor_col - len(graphemes[-1]) if graphemes else self._state.cursor_col - 1)
                elif self._state.cursor_line > 0:
                    self._state.cursor_line -= 1
                    prev_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
                    self._set_cursor_col(len(prev_line))

    def _page_scroll(self, direction: int) -> None:
        self._last_action = None
        terminal_rows = self._tui.terminal.rows if self._tui.terminal else 24
        page_size = max(5, int(terminal_rows * 0.3))
        visual_lines = self._build_visual_line_map(self._last_width)
        current_vi = self._find_current_visual_line(visual_lines)
        target_vi = max(0, min(len(visual_lines) - 1, current_vi + direction * page_size))
        self._move_to_visual_line(visual_lines, current_vi, target_vi)

    def _move_to_line_start(self) -> None:
        self._last_action = None
        self._set_cursor_col(0)

    def _move_to_line_end(self) -> None:
        self._last_action = None
        current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
        self._set_cursor_col(len(current_line))

    def _insert_character(self, char: str, skip_undo_coalescing: bool = False) -> None:
        self._history_index = -1
        if not skip_undo_coalescing:
            if is_whitespace_char(char) or self._last_action != "type-word":
                self._push_undo()
            self._last_action = "type-word"

        line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
        before = line[:self._state.cursor_col]
        after = line[self._state.cursor_col:]
        self._state.lines[self._state.cursor_line] = before + char + after
        self._set_cursor_col(self._state.cursor_col + len(char))

        if self.on_change:
            self.on_change(self.get_text())

        if not self._autocomplete_state:
            if char == "/" and self._is_at_start_of_message():
                self._try_trigger_autocomplete()
            elif char == "@":
                current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
                text_before = current_line[:self._state.cursor_col]
                char_before_at = text_before[-2] if len(text_before) >= 2 else None
                if len(text_before) == 1 or char_before_at in (None, " ", "\t"):
                    self._try_trigger_autocomplete()
            elif re.match(r"[a-zA-Z0-9.\-_]", char):
                current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
                text_before = current_line[:self._state.cursor_col]
                if self._is_in_slash_command_context(text_before):
                    self._try_trigger_autocomplete()
                elif re.search(r"(?:^|[\s])@[^\s]*$", text_before):
                    self._try_trigger_autocomplete()
        else:
            self._update_autocomplete()

    def _handle_paste(self, pasted_text: str) -> None:
        self._history_index = -1
        self._last_action = None
        self._push_undo()

        clean = pasted_text.replace("\r\n", "\n").replace("\r", "\n")
        tab_expanded = clean.replace("\t", "    ")
        filtered = "".join(c for c in tab_expanded if c == "\n" or ord(c) >= 32)

        if re.match(r"^[/~.]", filtered):
            current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
            char_before = current_line[self._state.cursor_col - 1] if self._state.cursor_col > 0 else ""
            if char_before and re.match(r"\w", char_before):
                filtered = " " + filtered

        pasted_lines = filtered.split("\n")
        total_chars = len(filtered)
        if len(pasted_lines) > 10 or total_chars > 1000:
            self._paste_counter += 1
            paste_id = self._paste_counter
            self._pastes[paste_id] = filtered
            marker = (
                f"[paste #{paste_id} +{len(pasted_lines)} lines]"
                if len(pasted_lines) > 10
                else f"[paste #{paste_id} {total_chars} chars]"
            )
            self._insert_text_at_cursor_internal(marker)
            return

        if len(pasted_lines) == 1:
            for char in filtered:
                self._insert_character(char, True)
            return

        self._insert_text_at_cursor_internal(filtered)

    def _add_new_line(self) -> None:
        self._history_index = -1
        self._last_action = None
        self._push_undo()

        current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
        before = current_line[:self._state.cursor_col]
        after = current_line[self._state.cursor_col:]
        self._state.lines[self._state.cursor_line] = before
        self._state.lines.insert(self._state.cursor_line + 1, after)
        self._state.cursor_line += 1
        self._set_cursor_col(0)

        if self.on_change:
            self.on_change(self.get_text())

    def _should_submit_on_backslash_enter(self, data: str, kb: object) -> bool:
        if self.disable_submit:
            return False
        if not matches_key(data, "enter"):
            return False
        submit_keys = kb.get_keys("submit")  # type: ignore[union-attr]
        has_shift_enter = "shift+enter" in submit_keys or "shift+return" in submit_keys
        if not has_shift_enter:
            return False
        current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
        return self._state.cursor_col > 0 and current_line and current_line[self._state.cursor_col - 1] == "\\"

    def _submit_value(self) -> None:
        result = "\n".join(self._state.lines).strip()
        for paste_id, paste_content in self._pastes.items():
            pattern = re.compile(rf"\[paste #{paste_id}( (\+\d+ lines|\d+ chars))?\]")
            result = pattern.sub(paste_content, result)

        self._state = _EditorState([""], 0, 0)
        self._pastes.clear()
        self._paste_counter = 0
        self._history_index = -1
        self._scroll_offset = 0
        self._undo_stack.clear()
        self._last_action = None

        if self.on_change:
            self.on_change("")
        if self.on_submit:
            self.on_submit(result)

    def _handle_backspace(self) -> None:
        self._history_index = -1
        self._last_action = None

        if self._state.cursor_col > 0:
            self._push_undo()
            line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
            before = line[:self._state.cursor_col]
            graphemes = _segment_graphemes(before)
            length = len(graphemes[-1]) if graphemes else 1
            self._state.lines[self._state.cursor_line] = line[:self._state.cursor_col - length] + line[self._state.cursor_col:]
            self._set_cursor_col(self._state.cursor_col - length)
        elif self._state.cursor_line > 0:
            self._push_undo()
            current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
            prev_line = self._state.lines[self._state.cursor_line - 1] if self._state.lines else ""
            self._state.lines[self._state.cursor_line - 1] = prev_line + current_line
            del self._state.lines[self._state.cursor_line]
            self._state.cursor_line -= 1
            self._set_cursor_col(len(prev_line))

        if self.on_change:
            self.on_change(self.get_text())

        if self._autocomplete_state:
            self._update_autocomplete()
        else:
            current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
            text_before = current_line[:self._state.cursor_col]
            if self._is_in_slash_command_context(text_before):
                self._try_trigger_autocomplete()
            elif re.search(r"(?:^|[\s])@[^\s]*$", text_before):
                self._try_trigger_autocomplete()

    def _handle_forward_delete(self) -> None:
        self._history_index = -1
        self._last_action = None
        current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""

        if self._state.cursor_col < len(current_line):
            self._push_undo()
            after = current_line[self._state.cursor_col:]
            graphemes = _segment_graphemes(after)
            length = len(graphemes[0]) if graphemes else 1
            self._state.lines[self._state.cursor_line] = current_line[:self._state.cursor_col] + current_line[self._state.cursor_col + length:]
        elif self._state.cursor_line < len(self._state.lines) - 1:
            self._push_undo()
            next_line = self._state.lines[self._state.cursor_line + 1]
            self._state.lines[self._state.cursor_line] = current_line + next_line
            del self._state.lines[self._state.cursor_line + 1]

        if self.on_change:
            self.on_change(self.get_text())

        if self._autocomplete_state:
            self._update_autocomplete()
        else:
            current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
            text_before = current_line[:self._state.cursor_col]
            if self._is_in_slash_command_context(text_before):
                self._try_trigger_autocomplete()
            elif re.search(r"(?:^|[\s])@[^\s]*$", text_before):
                self._try_trigger_autocomplete()

    def _delete_to_start_of_line(self) -> None:
        self._history_index = -1
        current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
        if self._state.cursor_col > 0:
            self._push_undo()
            deleted = current_line[:self._state.cursor_col]
            self._kill_ring.push(deleted, prepend=True, accumulate=(self._last_action == "kill"))
            self._last_action = "kill"
            self._state.lines[self._state.cursor_line] = current_line[self._state.cursor_col:]
            self._set_cursor_col(0)
        elif self._state.cursor_line > 0:
            self._push_undo()
            self._kill_ring.push("\n", prepend=True, accumulate=(self._last_action == "kill"))
            self._last_action = "kill"
            prev_line = self._state.lines[self._state.cursor_line - 1]
            self._state.lines[self._state.cursor_line - 1] = prev_line + current_line
            del self._state.lines[self._state.cursor_line]
            self._state.cursor_line -= 1
            self._set_cursor_col(len(prev_line))
        if self.on_change:
            self.on_change(self.get_text())

    def _delete_to_end_of_line(self) -> None:
        self._history_index = -1
        current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
        if self._state.cursor_col < len(current_line):
            self._push_undo()
            deleted = current_line[self._state.cursor_col:]
            self._kill_ring.push(deleted, prepend=False, accumulate=(self._last_action == "kill"))
            self._last_action = "kill"
            self._state.lines[self._state.cursor_line] = current_line[:self._state.cursor_col]
        elif self._state.cursor_line < len(self._state.lines) - 1:
            self._push_undo()
            self._kill_ring.push("\n", prepend=False, accumulate=(self._last_action == "kill"))
            self._last_action = "kill"
            next_line = self._state.lines[self._state.cursor_line + 1]
            self._state.lines[self._state.cursor_line] = current_line + next_line
            del self._state.lines[self._state.cursor_line + 1]
        if self.on_change:
            self.on_change(self.get_text())

    def _delete_word_backwards(self) -> None:
        self._history_index = -1
        current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
        if self._state.cursor_col == 0:
            if self._state.cursor_line > 0:
                self._push_undo()
                self._kill_ring.push("\n", prepend=True, accumulate=(self._last_action == "kill"))
                self._last_action = "kill"
                prev_line = self._state.lines[self._state.cursor_line - 1]
                self._state.lines[self._state.cursor_line - 1] = prev_line + current_line
                del self._state.lines[self._state.cursor_line]
                self._state.cursor_line -= 1
                self._set_cursor_col(len(prev_line))
        else:
            was_kill = self._last_action == "kill"
            self._push_undo()
            old_col = self._state.cursor_col
            self._move_word_backwards()
            delete_from = self._state.cursor_col
            self._set_cursor_col(old_col)
            deleted = current_line[delete_from:self._state.cursor_col]
            self._kill_ring.push(deleted, prepend=True, accumulate=was_kill)
            self._last_action = "kill"
            self._state.lines[self._state.cursor_line] = current_line[:delete_from] + current_line[self._state.cursor_col:]
            self._set_cursor_col(delete_from)
        if self.on_change:
            self.on_change(self.get_text())

    def _delete_word_forward(self) -> None:
        self._history_index = -1
        current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
        if self._state.cursor_col >= len(current_line):
            if self._state.cursor_line < len(self._state.lines) - 1:
                self._push_undo()
                self._kill_ring.push("\n", prepend=False, accumulate=(self._last_action == "kill"))
                self._last_action = "kill"
                next_line = self._state.lines[self._state.cursor_line + 1]
                self._state.lines[self._state.cursor_line] = current_line + next_line
                del self._state.lines[self._state.cursor_line + 1]
        else:
            was_kill = self._last_action == "kill"
            self._push_undo()
            old_col = self._state.cursor_col
            self._move_word_forwards()
            delete_to = self._state.cursor_col
            self._set_cursor_col(old_col)
            deleted = current_line[self._state.cursor_col:delete_to]
            self._kill_ring.push(deleted, prepend=False, accumulate=was_kill)
            self._last_action = "kill"
            self._state.lines[self._state.cursor_line] = current_line[:self._state.cursor_col] + current_line[delete_to:]
        if self.on_change:
            self.on_change(self.get_text())

    def _yank(self) -> None:
        if self._kill_ring.length == 0:
            return
        self._push_undo()
        text = self._kill_ring.peek()
        if text:
            self._insert_yanked_text(text)
        self._last_action = "yank"

    def _yank_pop(self) -> None:
        if self._last_action != "yank" or self._kill_ring.length <= 1:
            return
        self._push_undo()
        self._delete_yanked_text()
        self._kill_ring.rotate()
        text = self._kill_ring.peek()
        if text:
            self._insert_yanked_text(text)
        self._last_action = "yank"

    def _insert_yanked_text(self, text: str) -> None:
        self._history_index = -1
        lines_to_insert = text.split("\n")
        if len(lines_to_insert) == 1:
            current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
            before = current_line[:self._state.cursor_col]
            after = current_line[self._state.cursor_col:]
            self._state.lines[self._state.cursor_line] = before + text + after
            self._set_cursor_col(self._state.cursor_col + len(text))
        else:
            current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
            before = current_line[:self._state.cursor_col]
            after = current_line[self._state.cursor_col:]
            self._state.lines[self._state.cursor_line] = before + (lines_to_insert[0] or "")
            for i in range(1, len(lines_to_insert) - 1):
                self._state.lines.insert(self._state.cursor_line + i, lines_to_insert[i])
            last_line_idx = self._state.cursor_line + len(lines_to_insert) - 1
            self._state.lines.insert(last_line_idx, (lines_to_insert[-1] or "") + after)
            self._state.cursor_line = last_line_idx
            self._set_cursor_col(len(lines_to_insert[-1] or ""))
        if self.on_change:
            self.on_change(self.get_text())

    def _delete_yanked_text(self) -> None:
        yanked = self._kill_ring.peek()
        if not yanked:
            return
        yank_lines = yanked.split("\n")
        if len(yank_lines) == 1:
            current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
            delete_len = len(yanked)
            before = current_line[:self._state.cursor_col - delete_len]
            after = current_line[self._state.cursor_col:]
            self._state.lines[self._state.cursor_line] = before + after
            self._set_cursor_col(self._state.cursor_col - delete_len)
        else:
            start_line = self._state.cursor_line - (len(yank_lines) - 1)
            start_col = len(self._state.lines[start_line]) - len(yank_lines[0])
            after_cursor = self._state.lines[self._state.cursor_line][self._state.cursor_col:]
            before_yank = self._state.lines[start_line][:start_col]
            del self._state.lines[start_line:start_line + len(yank_lines)]
            self._state.lines.insert(start_line, before_yank + after_cursor)
            self._state.cursor_line = start_line
            self._set_cursor_col(start_col)
        if self.on_change:
            self.on_change(self.get_text())

    def _push_undo(self) -> None:
        self._undo_stack.push(self._state)

    def _undo(self) -> None:
        self._history_index = -1
        snapshot = self._undo_stack.pop()
        if snapshot is None:
            return
        self._state.lines = list(snapshot.lines)
        self._state.cursor_line = snapshot.cursor_line
        self._state.cursor_col = snapshot.cursor_col
        self._last_action = None
        self._preferred_visual_col = None
        if self.on_change:
            self.on_change(self.get_text())

    def _move_word_backwards(self) -> None:
        self._last_action = None
        current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
        if self._state.cursor_col == 0:
            if self._state.cursor_line > 0:
                self._state.cursor_line -= 1
                prev_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
                self._set_cursor_col(len(prev_line))
            return

        text_before = current_line[:self._state.cursor_col]
        graphemes = _segment_graphemes(text_before)
        new_col = self._state.cursor_col

        while graphemes and is_whitespace_char(graphemes[-1]):
            new_col -= len(graphemes.pop())

        if graphemes:
            last = graphemes[-1]
            if is_punctuation_char(last):
                while graphemes and is_punctuation_char(graphemes[-1]):
                    new_col -= len(graphemes.pop())
            else:
                while graphemes and not is_whitespace_char(graphemes[-1]) and not is_punctuation_char(graphemes[-1]):
                    new_col -= len(graphemes.pop())

        self._set_cursor_col(new_col)

    def _move_word_forwards(self) -> None:
        self._last_action = None
        current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
        if self._state.cursor_col >= len(current_line):
            if self._state.cursor_line < len(self._state.lines) - 1:
                self._state.cursor_line += 1
                self._set_cursor_col(0)
            return

        text_after = current_line[self._state.cursor_col:]
        graphemes = _segment_graphemes(text_after)
        idx = 0
        new_col = self._state.cursor_col

        while idx < len(graphemes) and is_whitespace_char(graphemes[idx]):
            new_col += len(graphemes[idx])
            idx += 1

        if idx < len(graphemes):
            first = graphemes[idx]
            if is_punctuation_char(first):
                while idx < len(graphemes) and is_punctuation_char(graphemes[idx]):
                    new_col += len(graphemes[idx])
                    idx += 1
            else:
                while idx < len(graphemes) and not is_whitespace_char(graphemes[idx]) and not is_punctuation_char(graphemes[idx]):
                    new_col += len(graphemes[idx])
                    idx += 1

        self._set_cursor_col(new_col)

    def _jump_to_char(self, char: str, direction: str) -> None:
        self._last_action = None
        is_forward = direction == "forward"
        lines = self._state.lines
        step = 1 if is_forward else -1
        end = len(lines) if is_forward else -1

        for line_idx in range(self._state.cursor_line, end, step):
            line = lines[line_idx]
            is_current = line_idx == self._state.cursor_line
            if is_current:
                search_from = self._state.cursor_col + 1 if is_forward else self._state.cursor_col - 1
            else:
                search_from = None

            if is_forward:
                idx = line.find(char, search_from) if search_from is not None else line.find(char)
            else:
                idx = line.rfind(char, 0, search_from + 1) if search_from is not None else line.rfind(char)

            if idx != -1:
                self._state.cursor_line = line_idx
                self._set_cursor_col(idx)
                return

    def _insert_text_at_cursor_internal(self, text: str) -> None:
        if not text:
            return
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        inserted_lines = normalized.split("\n")
        current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
        before = current_line[:self._state.cursor_col]
        after = current_line[self._state.cursor_col:]

        if len(inserted_lines) == 1:
            self._state.lines[self._state.cursor_line] = before + normalized + after
            self._set_cursor_col(self._state.cursor_col + len(normalized))
        else:
            new_lines = (
                self._state.lines[:self._state.cursor_line] +
                [before + inserted_lines[0]] +
                inserted_lines[1:-1] +
                [inserted_lines[-1] + after] +
                self._state.lines[self._state.cursor_line + 1:]
            )
            self._state.lines = new_lines
            self._state.cursor_line += len(inserted_lines) - 1
            self._set_cursor_col(len(inserted_lines[-1]))

        if self.on_change:
            self.on_change(self.get_text())

    # ── Autocomplete ─────────────────────────────────────────────────────────────

    def _is_slash_menu_allowed(self) -> bool:
        return self._state.cursor_line == 0

    def _is_at_start_of_message(self) -> bool:
        if not self._is_slash_menu_allowed():
            return False
        current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
        before = current_line[:self._state.cursor_col]
        return before.strip() in ("", "/")

    def _is_in_slash_command_context(self, text_before: str) -> bool:
        return self._is_slash_menu_allowed() and text_before.lstrip().startswith("/")

    def _try_trigger_autocomplete(self, explicit_tab: bool = False) -> None:
        if not self._autocomplete_provider:
            return
        if explicit_tab:
            provider = self._autocomplete_provider
            if hasattr(provider, "should_trigger_file_completion"):
                should = provider.should_trigger_file_completion(  # type: ignore[union-attr]
                    self._state.lines, self._state.cursor_line, self._state.cursor_col
                )
                if not should:
                    return

        suggestions = self._autocomplete_provider.get_suggestions(
            self._state.lines, self._state.cursor_line, self._state.cursor_col
        )
        if suggestions and suggestions.items:
            self._autocomplete_prefix = suggestions.prefix
            self._autocomplete_list = SelectList(
                suggestions.items, self._autocomplete_max_visible, self._theme.select_list
            )
            self._autocomplete_state = "regular"
        else:
            self._cancel_autocomplete()

    def _handle_tab_completion(self) -> None:
        if not self._autocomplete_provider:
            return
        current_line = self._state.lines[self._state.cursor_line] if self._state.lines else ""
        before = current_line[:self._state.cursor_col]
        if self._is_in_slash_command_context(before) and " " not in before.lstrip():
            self._try_trigger_autocomplete(True)
        else:
            self._force_file_autocomplete(True)

    def _force_file_autocomplete(self, explicit_tab: bool = False) -> None:
        if not self._autocomplete_provider:
            return
        provider = self._autocomplete_provider
        if not hasattr(provider, "get_force_file_suggestions"):
            self._try_trigger_autocomplete(True)
            return

        suggestions = provider.get_force_file_suggestions(  # type: ignore[union-attr]
            self._state.lines, self._state.cursor_line, self._state.cursor_col
        )
        if suggestions and suggestions.items:
            if explicit_tab and len(suggestions.items) == 1:
                item = suggestions.items[0]
                self._push_undo()
                self._last_action = None
                result = self._autocomplete_provider.apply_completion(
                    self._state.lines, self._state.cursor_line, self._state.cursor_col,
                    item, suggestions.prefix
                )
                self._state.lines = result.lines
                self._state.cursor_line = result.cursor_line
                self._set_cursor_col(result.cursor_col)
                if self.on_change:
                    self.on_change(self.get_text())
                return
            self._autocomplete_prefix = suggestions.prefix
            self._autocomplete_list = SelectList(
                suggestions.items, self._autocomplete_max_visible, self._theme.select_list
            )
            self._autocomplete_state = "force"
        else:
            self._cancel_autocomplete()

    def _apply_autocomplete(self, selected: SelectItem) -> None:
        if not self._autocomplete_provider:
            return
        self._push_undo()
        self._last_action = None
        result = self._autocomplete_provider.apply_completion(
            self._state.lines, self._state.cursor_line, self._state.cursor_col,
            selected, self._autocomplete_prefix
        )
        self._state.lines = result.lines
        self._state.cursor_line = result.cursor_line
        self._set_cursor_col(result.cursor_col)

    def _cancel_autocomplete(self) -> None:
        self._autocomplete_state = None
        self._autocomplete_list = None
        self._autocomplete_prefix = ""

    def _update_autocomplete(self) -> None:
        if not self._autocomplete_state or not self._autocomplete_provider:
            return
        if self._autocomplete_state == "force":
            self._force_file_autocomplete()
            return
        suggestions = self._autocomplete_provider.get_suggestions(
            self._state.lines, self._state.cursor_line, self._state.cursor_col
        )
        if suggestions and suggestions.items:
            self._autocomplete_prefix = suggestions.prefix
            self._autocomplete_list = SelectList(
                suggestions.items, self._autocomplete_max_visible, self._theme.select_list
            )
        else:
            self._cancel_autocomplete()
