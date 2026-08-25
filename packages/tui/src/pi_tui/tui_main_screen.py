"""
Main-screen TUI renderer and handoff APIs — mirrors tui-main-screen.ts.

The shared TUI lifecycle lives in tui.py. This module extracts the
main-screen differential renderer plus capture/restore state used when
handing the terminal between TUI instances.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .terminal_image import delete_kitty_image, is_image_line
from .tui import TUI, TuiStopOptions
from .utils import visible_width

_KITTY_SEQUENCE_PREFIX = "\x1b_G"


@dataclass
class _KittyImageHeader:
    ids: list[int]
    rows: int


def _parse_kitty_image_header(line: str) -> _KittyImageHeader | None:
    sequence_start = line.find(_KITTY_SEQUENCE_PREFIX)
    if sequence_start == -1:
        return None
    params_start = sequence_start + len(_KITTY_SEQUENCE_PREFIX)
    params_end = line.find(";", params_start)
    if params_end == -1:
        return None

    ids: list[int] = []
    rows = 1
    for param in line[params_start:params_end].split(","):
        if "=" not in param:
            continue
        key, value = param.split("=", 1)
        try:
            number_value = int(value)
        except ValueError:
            continue
        if number_value <= 0 or number_value > 0xFFFFFFFF:
            continue
        if key == "i":
            ids.append(number_value)
        elif key == "r":
            rows = number_value
    return _KittyImageHeader(ids=ids, rows=rows)


def _extract_kitty_image_ids(line: str) -> list[int]:
    header = _parse_kitty_image_header(line)
    return header.ids if header else []


def _extract_kitty_image_rows(line: str) -> int:
    header = _parse_kitty_image_header(line)
    return header.rows if header else 1


def _is_termux_session() -> bool:
    return bool(os.environ.get("TERMUX_VERSION"))


@dataclass
class TuiMainScreenRenderState:
    previous_lines: list[str]
    previous_width: int
    previous_height: int
    cursor_row: int
    hardware_cursor_row: int
    max_lines_rendered: int
    previous_viewport_top: int


class TuiMainScreen(TUI):
    """TUI implementation that renders into the terminal's main screen and scrollback."""

    def __init__(
        self,
        terminal,
        show_hardware_cursor: bool | None = None,
        log_directory: str | None = None,
    ) -> None:
        super().__init__(terminal, show_hardware_cursor)
        self._previous_kitty_image_ids: set[int] = set()
        self._log_directory = (
            log_directory
            or os.environ.get("PI_CODING_AGENT_DIR")
            or os.path.join(os.path.expanduser("~"), ".pi", "agent")
        )

    @property
    def mode(self) -> str:
        return "regular"

    def capture_render_state(self) -> TuiMainScreenRenderState:
        return TuiMainScreenRenderState(
            previous_lines=list(self._previous_lines),
            previous_width=self._previous_width,
            previous_height=self._previous_height,
            cursor_row=self._cursor_row,
            hardware_cursor_row=self._hardware_cursor_row,
            max_lines_rendered=self._max_lines_rendered,
            previous_viewport_top=self._previous_viewport_top,
        )

    def restore_render_state(self, state: TuiMainScreenRenderState) -> None:
        self._previous_lines = ["" if is_image_line(line) else line for line in state.previous_lines]
        self._previous_kitty_image_ids = set()
        self._previous_width = state.previous_width
        self._previous_height = state.previous_height
        self._cursor_row = state.cursor_row
        self._hardware_cursor_row = state.hardware_cursor_row
        self._max_lines_rendered = state.max_lines_rendered
        self._previous_viewport_top = state.previous_viewport_top

    def _reset_render_state(self) -> None:
        super()._reset_render_state()
        self._previous_kitty_image_ids = set()

    def _before_terminal_stop(self, options: TuiStopOptions) -> None:
        if options.preserve_screen or not self._previous_lines:
            return
        self.terminal.write(" ")
        target_row = len(self._previous_lines)
        line_diff = target_row - self._hardware_cursor_row
        if line_diff > 0:
            self.terminal.write(f"\x1b[{line_diff}B")
        elif line_diff < 0:
            self.terminal.write(f"\x1b[{-line_diff}A")
        self.terminal.write("\r\n")

    def _collect_kitty_image_ids(self, lines: list[str]) -> set[int]:
        ids: set[int] = set()
        for line in lines:
            ids.update(_extract_kitty_image_ids(line))
        return ids

    def _delete_kitty_images(self, ids) -> str:
        return "".join(delete_kitty_image(image_id) for image_id in ids)

    def _get_kitty_image_reserved_rows(
        self,
        lines: list[str],
        index: int,
        max_index: int | None = None,
    ) -> int:
        if max_index is None:
            max_index = len(lines) - 1
        rows = _extract_kitty_image_rows(lines[index] if index < len(lines) else "")
        if rows <= 1:
            return 1
        max_rows = min(rows, max_index - index + 1, len(lines) - index)
        reserved_rows = 1
        while reserved_rows < max_rows:
            line = lines[index + reserved_rows] if index + reserved_rows < len(lines) else ""
            if is_image_line(line) or visible_width(line) > 0:
                break
            reserved_rows += 1
        return reserved_rows

    def _expand_changed_range_for_kitty_images(
        self,
        first_changed: int,
        last_changed: int,
        new_lines: list[str],
    ) -> tuple[int, int]:
        expanded_first = first_changed
        expanded_last = last_changed

        def expand_for_lines(lines: list[str]) -> None:
            nonlocal expanded_first, expanded_last
            for i, line in enumerate(lines):
                if not _extract_kitty_image_ids(line):
                    continue
                block_end = i + self._get_kitty_image_reserved_rows(lines, i) - 1
                if i >= first_changed or (i <= last_changed and block_end >= first_changed):
                    expanded_first = min(expanded_first, i)
                    expanded_last = max(expanded_last, block_end)

        expand_for_lines(self._previous_lines)
        expand_for_lines(new_lines)
        return expanded_first, expanded_last

    def _delete_changed_kitty_images(self, first_changed: int, last_changed: int) -> str:
        if first_changed < 0 or last_changed < first_changed:
            return ""
        ids: set[int] = set()
        max_line = min(last_changed, len(self._previous_lines) - 1)
        for i in range(first_changed, max_line + 1):
            ids.update(_extract_kitty_image_ids(self._previous_lines[i] if i < len(self._previous_lines) else ""))
        return self._delete_kitty_images(ids)

    def _do_render(self) -> None:
        if self._stopped:
            return
        width = self.terminal.columns
        height = self.terminal.rows
        width_changed = self._previous_width != 0 and self._previous_width != width
        height_changed = self._previous_height != 0 and self._previous_height != height
        previous_buffer_length = (
            self._previous_viewport_top + self._previous_height if self._previous_height > 0 else height
        )
        prev_viewport_top = (
            max(0, previous_buffer_length - height) if height_changed else self._previous_viewport_top
        )
        viewport_top = prev_viewport_top
        hardware_cursor_row = self._hardware_cursor_row

        def compute_line_diff(target_row: int) -> int:
            current_screen_row = hardware_cursor_row - prev_viewport_top
            target_screen_row = target_row - viewport_top
            return target_screen_row - current_screen_row

        new_lines = self.render(width)
        if self.has_overlay_entries:
            new_lines = self._composite_overlays(new_lines, width, height)
        cursor_pos = self._extract_cursor_position(new_lines, height)
        new_lines = self._apply_line_resets(new_lines)

        def full_render(clear: bool) -> None:
            nonlocal hardware_cursor_row, viewport_top, prev_viewport_top
            self._full_redraw_count += 1
            buf = "\x1b[?2026h"
            if clear:
                buf += self._delete_kitty_images(self._previous_kitty_image_ids)
                buf += "\x1b[2J\x1b[H\x1b[3J"
            i = 0
            while i < len(new_lines):
                if i > 0:
                    buf += "\r\n"
                line = new_lines[i]
                image_reserved_rows = self._get_kitty_image_reserved_rows(new_lines, i) if is_image_line(line) else 1
                if image_reserved_rows > 1 and image_reserved_rows <= height:
                    for _row in range(1, image_reserved_rows):
                        buf += "\r\n"
                    buf += f"\x1b[{image_reserved_rows - 1}A"
                    buf += line
                    buf += f"\x1b[{image_reserved_rows - 1}B"
                    i += image_reserved_rows
                    continue
                buf += line
                i += 1
            buf += "\x1b[?2026l"
            self.terminal.write(buf)
            self._cursor_row = max(0, len(new_lines) - 1)
            self._hardware_cursor_row = self._cursor_row
            if clear:
                self._max_lines_rendered = len(new_lines)
            else:
                self._max_lines_rendered = max(self._max_lines_rendered, len(new_lines))
            buffer_length = max(height, len(new_lines))
            self._previous_viewport_top = max(0, buffer_length - height)
            self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_lines = new_lines
            self._previous_kitty_image_ids = self._collect_kitty_image_ids(new_lines)
            self._previous_width = width
            self._previous_height = height

        debug_redraw = os.environ.get("PI_DEBUG_REDRAW") == "1"

        def log_redraw(reason: str) -> None:
            if not debug_redraw:
                return
            try:
                from datetime import datetime

                log_path = os.path.join(self._log_directory, "pi-debug.log")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                msg = (
                    f"[{datetime.now().isoformat()}] fullRender: {reason} "
                    f"(prev={len(self._previous_lines)}, new={len(new_lines)}, height={height})\n"
                )
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(msg)
            except Exception:
                pass

        if not self._previous_lines and not width_changed and not height_changed:
            log_redraw("first render")
            full_render(False)
            return

        if width_changed:
            log_redraw(f"terminal width changed ({self._previous_width} -> {width})")
            full_render(True)
            return

        if height_changed and not _is_termux_session():
            log_redraw(f"terminal height changed ({self._previous_height} -> {height})")
            full_render(True)
            return

        if self.get_clear_on_shrink() and len(new_lines) < self._max_lines_rendered and not self.has_overlay_entries:
            log_redraw(f"clearOnShrink (maxLinesRendered={self._max_lines_rendered})")
            full_render(True)
            return

        first_changed = -1
        last_changed = -1
        max_lines = max(len(new_lines), len(self._previous_lines))
        for i in range(max_lines):
            old_line = self._previous_lines[i] if i < len(self._previous_lines) else ""
            new_line = new_lines[i] if i < len(new_lines) else ""
            if old_line != new_line:
                if first_changed == -1:
                    first_changed = i
                last_changed = i
        appended_lines = len(new_lines) > len(self._previous_lines)
        if appended_lines:
            if first_changed == -1:
                first_changed = len(self._previous_lines)
            last_changed = len(new_lines) - 1
        if first_changed != -1:
            first_changed, last_changed = self._expand_changed_range_for_kitty_images(
                first_changed, last_changed, new_lines
            )
        append_start = appended_lines and first_changed == len(self._previous_lines) and first_changed > 0

        if first_changed == -1:
            self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_viewport_top = prev_viewport_top
            self._previous_height = height
            return

        if first_changed >= len(new_lines):
            if len(self._previous_lines) > len(new_lines):
                buf = "\x1b[?2026h"
                buf += self._delete_changed_kitty_images(first_changed, last_changed)
                target_row = max(0, len(new_lines) - 1)
                if target_row < prev_viewport_top:
                    log_redraw(f"deleted lines moved viewport up ({target_row} < {prev_viewport_top})")
                    full_render(True)
                    return
                line_diff = compute_line_diff(target_row)
                if line_diff > 0:
                    buf += f"\x1b[{line_diff}B"
                elif line_diff < 0:
                    buf += f"\x1b[{-line_diff}A"
                buf += "\r"
                extra_lines = len(self._previous_lines) - len(new_lines)
                if extra_lines > height:
                    log_redraw(f"extraLines > height ({extra_lines} > {height})")
                    full_render(True)
                    return
                clear_start_offset = 0 if len(new_lines) == 0 else 1
                if extra_lines > 0 and clear_start_offset > 0:
                    buf += f"\x1b[{clear_start_offset}B"
                for i in range(extra_lines):
                    buf += "\r\x1b[2K"
                    if i < extra_lines - 1:
                        buf += "\x1b[1B"
                move_back = max(0, extra_lines - 1 + clear_start_offset)
                if move_back > 0:
                    buf += f"\x1b[{move_back}A"
                buf += "\x1b[?2026l"
                self.terminal.write(buf)
                self._cursor_row = target_row
                self._hardware_cursor_row = target_row
            self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_lines = new_lines
            self._previous_kitty_image_ids = self._collect_kitty_image_ids(new_lines)
            self._previous_width = width
            self._previous_height = height
            self._previous_viewport_top = prev_viewport_top
            return

        if first_changed < prev_viewport_top:
            log_redraw(f"firstChanged < viewportTop ({first_changed} < {prev_viewport_top})")
            full_render(True)
            return

        buf = "\x1b[?2026h"
        buf += self._delete_changed_kitty_images(first_changed, last_changed)
        prev_viewport_bottom = prev_viewport_top + height - 1
        move_target_row = first_changed - 1 if append_start else first_changed
        if move_target_row > prev_viewport_bottom:
            current_screen_row = max(0, min(height - 1, hardware_cursor_row - prev_viewport_top))
            move_to_bottom = height - 1 - current_screen_row
            if move_to_bottom > 0:
                buf += f"\x1b[{move_to_bottom}B"
            scroll = move_target_row - prev_viewport_bottom
            buf += "\r\n" * scroll
            prev_viewport_top += scroll
            viewport_top += scroll
            hardware_cursor_row = move_target_row

        line_diff = compute_line_diff(move_target_row)
        if line_diff > 0:
            buf += f"\x1b[{line_diff}B"
        elif line_diff < 0:
            buf += f"\x1b[{-line_diff}A"
        buf += "\r\n" if append_start else "\r"

        render_end = min(last_changed, len(new_lines) - 1)
        i = first_changed
        while i <= render_end:
            if i > first_changed:
                buf += "\r\n"
            line = new_lines[i]
            image_reserved_rows = (
                self._get_kitty_image_reserved_rows(new_lines, i, render_end) if is_image_line(line) else 1
            )
            if image_reserved_rows > 1:
                image_start_screen_row = i - viewport_top
                if image_start_screen_row < 0 or image_start_screen_row + image_reserved_rows > height:
                    log_redraw(
                        f"kitty image pre-clear would scroll ({image_start_screen_row} + {image_reserved_rows} > {height})"
                    )
                    full_render(True)
                    return
                buf += "\x1b[2K"
                for _row in range(1, image_reserved_rows):
                    buf += "\r\n\x1b[2K"
                buf += f"\x1b[{image_reserved_rows - 1}A"
                buf += line
                buf += f"\x1b[{image_reserved_rows - 1}B"
                i += image_reserved_rows
                continue

            buf += "\x1b[2K"
            if not is_image_line(line) and visible_width(line) > width:
                from .utils import slice_by_column

                line = slice_by_column(line, 0, width, True)
            buf += line
            i += 1

        final_cursor_row = render_end
        if len(self._previous_lines) > len(new_lines):
            if render_end < len(new_lines) - 1:
                move_down = len(new_lines) - 1 - render_end
                buf += f"\x1b[{move_down}B"
                final_cursor_row = len(new_lines) - 1
            extra_lines = len(self._previous_lines) - len(new_lines)
            for _ in range(extra_lines):
                buf += "\r\n\x1b[2K"
            buf += f"\x1b[{extra_lines}A"

        buf += "\x1b[?2026l"
        self.terminal.write(buf)
        self._cursor_row = max(0, len(new_lines) - 1)
        self._hardware_cursor_row = final_cursor_row
        self._max_lines_rendered = max(self._max_lines_rendered, len(new_lines))
        self._previous_viewport_top = max(prev_viewport_top, final_cursor_row - height + 1)
        self._position_hardware_cursor(cursor_pos, len(new_lines))
        self._previous_lines = new_lines
        self._previous_kitty_image_ids = self._collect_kitty_image_ids(new_lines)
        self._previous_width = width
        self._previous_height = height
