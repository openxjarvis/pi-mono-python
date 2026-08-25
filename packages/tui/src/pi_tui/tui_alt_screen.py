"""Alternate-screen TUI — mirrors packages/tui/src/tui-alt-screen.ts"""
from __future__ import annotations

import base64
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

from .alt_screen_search import (
    AltScreenSearchComponent,
    AltScreenSearchMatch,
    find_alt_screen_search_matches,
    get_alt_screen_search_match_key,
)
from .components.alt_screen_flash import AltScreenFlashContainer
from .components.scroll_view import ScrollView
from .keybindings import get_keybindings
from .keys import is_key_release
from .layout import (
    LayoutFrame,
    ScrollbarGeometry,
    get_scrollbar_geometry,
    get_scroll_view_box,
    get_scroll_views_at,
    render_layout_frame,
)
from .terminal import Terminal
from .terminal_image import (
    TerminalCapabilities,
    delete_all_kitty_images,
    delete_all_kitty_placements,
    delete_kitty_image,
    get_capabilities,
    get_kitty_image_placement,
    is_image_line,
    set_capabilities,
)
from .tui import (
    CURSOR_MARKER,
    OverlayHandle,
    OverlayOptions,
    TUI,
    TuiStopOptions,
    composite_tui_line,
)
from .utils import (
    extract_ansi_code,
    get_grapheme_cell_range,
    get_osc8_link_at_column,
    slice_by_column,
    strip_terminal_sequences,
    visible_width,
)

_ENTER_ALT_SCREEN = "\x1b[?1049h"
_EXIT_ALT_SCREEN = "\x1b[?1049l"
_DISABLE_AUTOWRAP = "\x1b[?7l"
_ENABLE_AUTOWRAP = "\x1b[?7h"
_ENABLE_BUTTON_MOTION_MOUSE = "\x1b[?1000h\x1b[?1002h\x1b[?1004h\x1b[?1006h"
_ENABLE_ALL_MOTION_MOUSE = "\x1b[?1000h\x1b[?1002h\x1b[?1003h\x1b[?1004h\x1b[?1006h"
_DISABLE_MOUSE = "\x1b[?1006l\x1b[?1004l\x1b[?1003l\x1b[?1002l\x1b[?1000l"
_FOCUS_IN = "\x1b[I"
_FOCUS_OUT = "\x1b[O"
_BEGIN_SYNCHRONIZED_OUTPUT = "\x1b[?2026h"
_END_SYNCHRONIZED_OUTPUT = "\x1b[?2026l"
_OSC133_ZONE_PREFIX = re.compile(r"^(?:\x1b\]133;[ABC](?:\x07|\x1b\\))+")
_OSC133_PROMPT_START = re.compile(r"^\x1b\]133;A(?:\x07|\x1b\\)")
_PAGE_SCROLL_OVERLAP = 4
_MAX_CACHED_OFFSCREEN_KITTY_IMAGES = 16
_MAX_CACHED_OFFSCREEN_KITTY_TRANSMISSION_BYTES = 32 * 1024 * 1024
_MAX_CACHED_OFFSCREEN_KITTY_DECODED_BYTES = 64 * 1024 * 1024
_DOUBLE_CLICK_INTERVAL_MS = 500
_WORD_RE = re.compile(r"\w+|[^\w]+", re.UNICODE)
_SGR_MOUSE_RE = re.compile(r"^\x1b\[<(\d+);(\d+);(\d+)([Mm])$")
_SGR_WHEEL_RE = re.compile(r"^\x1b\[<(\d+);(\d+);(\d+)[Mm]$")


@dataclass
class _WordSegment:
    segment: str


def _segment_words(text: str) -> list[_WordSegment]:
    return [_WordSegment(match.group(0)) for match in _WORD_RE.finditer(text)]


@dataclass
class _CachedKittyImage:
    transmission_generation: int
    transmission_bytes: int
    estimated_decoded_bytes: int


@dataclass
class _SelectionPoint:
    row: int
    col: int
    scroll_view: ScrollView | None = None
    boundary: bool = False


@dataclass
class _SelectionRange:
    start: _SelectionPoint
    end: _SelectionPoint


SelectionGranularity = Literal["character", "word", "line"]
SearchSelectionMode = Literal["query", "retain", "next", "previous"]


@dataclass
class _ClickTarget:
    timestamp: float
    count: int
    row: int
    scroll_view: ScrollView | None
    word_start: int
    word_end: int


@dataclass
class _SgrMouseEvent:
    button: int
    x: int
    y: int
    release: bool


@dataclass
class _WheelEvent:
    direction: Literal[-1, 1]
    x: int
    y: int


@dataclass
class _ScrollbarDrag:
    scroll_view: ScrollView
    grab_offset: int


@dataclass
class _ScrollbarTarget:
    scroll_view: ScrollView
    geometry: ScrollbarGeometry


@dataclass
class _ActiveSearch:
    component: AltScreenSearchComponent
    overlay: OverlayHandle | None = None
    query: str = ""
    matches: list[AltScreenSearchMatch] = field(default_factory=list)
    selected_index: int = -1
    selected_key: str | None = None
    anchor_row: int = 0
    selection_mode: SearchSelectionMode = "query"


@dataclass
class _SearchHighlightRange:
    start_col: int
    end_col: int
    current: bool


@dataclass
class TuiAltScreenOptions:
    wheel_scroll_lines: int = 1
    mouse: bool = True
    search_match_style: Callable[[str], str] | None = None
    search_current_match_style: Callable[[str], str] | None = None
    open_url: Callable[[str], None] | None = None
    on_right_click_paste: Callable[[], None] | None = None
    copy_selection: Callable[[str], bool] | None = None


class _ImplicitDocument:
    def __init__(self, tui: "TuiAltScreen") -> None:
        self._tui = tui

    def render(self, width: int) -> list[str]:
        return TUI.render(self._tui, width)

    def invalidate(self) -> None:
        for child in self._tui.children:
            if hasattr(child, "invalidate"):
                child.invalidate()

    def handle_input(self, _data: str) -> None:
        return


class TuiAltScreen(TUI):
    """Alternate-screen TUI with a scrollable, application-owned viewport."""

    is_viewport_tui = True

    def __init__(
        self,
        terminal: Terminal,
        show_hardware_cursor: bool | None = None,
        log_directory: str | None = None,
        options: TuiAltScreenOptions | dict | None = None,
    ) -> None:
        super().__init__(terminal, show_hardware_cursor)
        if isinstance(options, dict):
            options = TuiAltScreenOptions(**options)
        opts = options or TuiAltScreenOptions()
        self._log_directory = log_directory
        self._previous_screen: list[str] = []
        self._last_document: list[str] = []
        self._previous_screen_width = 0
        self._previous_screen_height = 0
        self._layout_root: object | None = None
        self._current_layout: LayoutFrame | None = None
        self._implicit_document = _ImplicitDocument(self)
        self._implicit_scroll_view = ScrollView(self._implicit_document, {"follow": "end", "primary": True})
        self._flashes = AltScreenFlashContainer(lambda: self.request_render())
        self._alt_screen_active = False
        self._image_protocol = None
        self._saved_capabilities = None
        self._uploaded_kitty_images: dict[int, _CachedKittyImage] = {}
        self._selection_anchor: _SelectionPoint | None = None
        self._selection_focus: _SelectionPoint | None = None
        self._selection_granularity: SelectionGranularity = "character"
        self._selection_initial_range: _SelectionRange | None = None
        self._last_click: _ClickTarget | None = None
        self._selection_drag_pointer: dict[str, int] | None = None
        self._selection_auto_scroll_direction: Literal[-1, 0, 1] = 0
        self._selection_auto_scroll_timer: threading.Timer | None = None
        self._selection_press_active = False
        self._scrollbar_drag: _ScrollbarDrag | None = None
        self._scrollbar_hover: ScrollView | None = None
        self._active_search: _ActiveSearch | None = None
        self._pressed_url: str | None = None
        self._selection_dragged = False
        self._wheel_scroll_lines = max(1, math.floor(opts.wheel_scroll_lines))
        self._mouse_enabled = opts.mouse
        self._search_match_style = opts.search_match_style or (lambda text: f"\x1b[4m{text}\x1b[24m")
        self._search_current_match_style = opts.search_current_match_style or (
            lambda text: f"\x1b[1;7m{text}\x1b[22;27m"
        )
        self._open_url = opts.open_url
        self._on_right_click_paste = opts.on_right_click_paste
        self._copy_selection = opts.copy_selection
        self.add_input_listener(self._handle_viewport_input)

    @property
    def mode(self) -> str:
        return "fullscreen"

    @property
    def viewport_top(self) -> int:
        return self._get_primary_scroll_view().scroll_top

    @property
    def is_following_output(self) -> bool:
        return self._get_primary_scroll_view().is_following_end

    def set_layout_root(self, component: object | None) -> None:
        if self._layout_root is component:
            return
        self._layout_root = component
        self._current_layout = None
        self.request_render()

    def render(self, width: int) -> list[str]:
        if self._layout_root is not None and hasattr(self._layout_root, "render"):
            return self._layout_root.render(width)
        return super().render(width)

    def _get_mounted_roots(self) -> list:
        return [self._layout_root] if self._layout_root is not None else list(self.children)

    def _get_primary_scroll_view(self) -> ScrollView:
        if self._current_layout and self._current_layout.primary_scroll_view is not None:
            return self._current_layout.primary_scroll_view
        return self._implicit_scroll_view

    def _before_terminal_start(self) -> None:
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._stop_scrollbar_hover()
        self._stop_scrollbar_drag()
        self._flashes.dispose()
        self._alt_screen_active = True
        capabilities = get_capabilities()
        self._image_protocol = capabilities.images
        self._uploaded_kitty_images.clear()
        if capabilities.images == "iterm2":
            self._saved_capabilities = capabilities
            set_capabilities(
                TerminalCapabilities(
                    images=None,
                    true_color=capabilities.true_color,
                    hyperlinks=capabilities.hyperlinks,
                )
            )
            self.invalidate()
        self._last_document = []
        self._selection_anchor = None
        self._selection_focus = None
        self._selection_granularity = "character"
        self._selection_initial_range = None
        self._last_click = None
        self._pressed_url = None
        self._selection_dragged = False
        self._reset_render_state()
        term = (os.environ.get("TERM") or "").lower()
        mouse_sequence = (
            _ENABLE_BUTTON_MOTION_MOUSE
            if (
                os.environ.get("TMUX") is not None
                or os.environ.get("ZELLIJ") is not None
                or os.environ.get("STY") is not None
                or term.startswith("tmux")
                or term.startswith("screen")
            )
            else _ENABLE_ALL_MOTION_MOUSE
        )
        self.terminal.write(
            f"{_ENTER_ALT_SCREEN}{_DISABLE_AUTOWRAP}{mouse_sequence if self._mouse_enabled else ''}\x1b[2J\x1b[H\x1b[?25l"
        )

    def _before_terminal_stop(self, _options: TuiStopOptions) -> None:
        self._close_search()
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._stop_scrollbar_hover()
        self._stop_scrollbar_drag()
        self._flashes.dispose()
        if not self._alt_screen_active:
            return
        self.terminal.write(
            f"{_BEGIN_SYNCHRONIZED_OUTPUT}{self._delete_kitty_images()}"
            f"{_DISABLE_MOUSE if self._mouse_enabled else ''}{_ENABLE_AUTOWRAP}{_END_SYNCHRONIZED_OUTPUT}"
        )
        self._uploaded_kitty_images.clear()

    def _after_terminal_stop(self, options: TuiStopOptions) -> None:
        if not self._alt_screen_active:
            return
        self._alt_screen_active = False
        if options.preserve_screen:
            self.terminal.write(f"{_BEGIN_SYNCHRONIZED_OUTPUT}{_EXIT_ALT_SCREEN}\x1b[?25h{_END_SYNCHRONIZED_OUTPUT}")
        else:
            width = max(1, self.terminal.columns)
            document_lines = [_OSC133_ZONE_PREFIX.sub("", line) for line in self.render(width)]
            reset = self._apply_line_resets([line.replace(CURSOR_MARKER, "") for line in document_lines])
            self._last_document = [
                line
                if is_image_line(line) or visible_width(line) <= width
                else slice_by_column(line, 0, width, True)
                for line in reset
            ]
            buf = f"{_BEGIN_SYNCHRONIZED_OUTPUT}{_EXIT_ALT_SCREEN}{_DISABLE_AUTOWRAP}"
            for row, line in enumerate(self._last_document):
                if row > 0:
                    buf += "\r\n"
                buf += f"\r\x1b[2K{line}"
            buf += f"\x1b[0m{_ENABLE_AUTOWRAP}\r\n\x1b[?25h{_END_SYNCHRONIZED_OUTPUT}"
            self.terminal.write(buf)
        if self._saved_capabilities is not None:
            set_capabilities(self._saved_capabilities)
            self._saved_capabilities = None

    def _delete_kitty_images(self) -> str:
        return delete_all_kitty_images() if self._image_protocol == "kitty" else ""

    def _prepare_kitty_screen(self, screen: list[str]) -> tuple[list[str], str]:
        visible_image_ids: set[int] = set()
        lines: list[str] = []
        for line in screen:
            placement = get_kitty_image_placement(line)
            if placement is None:
                lines.append(line)
                continue
            visible_image_ids.add(placement.image_id)
            cached = self._uploaded_kitty_images.pop(placement.image_id, None)
            self._uploaded_kitty_images[placement.image_id] = _CachedKittyImage(
                transmission_generation=placement.transmission_generation,
                transmission_bytes=placement.transmission_bytes,
                estimated_decoded_bytes=placement.estimated_decoded_bytes,
            )
            lines.append(
                placement.replacement_line
                if cached and cached.transmission_generation == placement.transmission_generation
                else line
            )

        cached_count = 0
        cached_transmission = 0
        cached_decoded = 0
        for image_id, cached in self._uploaded_kitty_images.items():
            if image_id in visible_image_ids:
                continue
            cached_count += 1
            cached_transmission += cached.transmission_bytes
            cached_decoded += cached.estimated_decoded_bytes

        evicted = ""
        for image_id in list(self._uploaded_kitty_images):
            if (
                cached_count <= _MAX_CACHED_OFFSCREEN_KITTY_IMAGES
                and cached_transmission <= _MAX_CACHED_OFFSCREEN_KITTY_TRANSMISSION_BYTES
                and cached_decoded <= _MAX_CACHED_OFFSCREEN_KITTY_DECODED_BYTES
            ):
                break
            if image_id in visible_image_ids:
                continue
            cached = self._uploaded_kitty_images.pop(image_id)
            evicted += delete_kitty_image(image_id)
            cached_count -= 1
            cached_transmission -= cached.transmission_bytes
            cached_decoded -= cached.estimated_decoded_bytes
        return lines, evicted

    def _reset_render_state(self) -> None:
        self._previous_screen = []
        self._previous_screen_width = 0
        self._previous_screen_height = 0
        self._current_layout = None

    def scroll_by(self, lines: int) -> None:
        self._get_primary_scroll_view().scroll_by(lines)
        self.request_render()

    def scroll_to_top(self) -> None:
        self._get_primary_scroll_view().scroll_to_start()
        self.request_render()

    def scroll_to_bottom(self) -> None:
        self._get_primary_scroll_view().scroll_to_end()
        self.request_render()

    def _scroll_to_prompt(self, direction: Literal[-1, 1]) -> None:
        if self._current_layout is None:
            return
        scroll_view = self._get_primary_scroll_view()
        box = get_scroll_view_box(self._current_layout, scroll_view)
        lines = box.scroll_content_lines if box else None
        if not lines:
            return
        row = scroll_view.scroll_top + direction
        while 0 <= row < len(lines):
            if _OSC133_PROMPT_START.match(lines[row] or ""):
                scroll_view.scroll_to(row)
                self.request_render()
                return
            row += direction

    def _open_search(self) -> None:
        if self._active_search is not None:
            if self._active_search.overlay:
                self._active_search.overlay.focus()
            return
        component = AltScreenSearchComponent(self._update_search_query)
        search = _ActiveSearch(
            component=component,
            query="",
            matches=[],
            selected_index=-1,
            anchor_row=self._get_primary_scroll_view().scroll_top,
            selection_mode="query",
        )
        self._active_search = search
        search.overlay = self.show_overlay(
            component,
            OverlayOptions(anchor="top-right", width="40%", min_width=24, margin=1),
        )

    def _close_search(self) -> None:
        search = self._active_search
        if search is None:
            return
        self._active_search = None
        if search.overlay:
            search.overlay.hide()
        self.request_render()

    def _update_search_query(self, query: str) -> None:
        search = self._active_search
        if search is None or query == search.query:
            return
        selected = search.matches[search.selected_index] if 0 <= search.selected_index < len(search.matches) else None
        search.anchor_row = (
            selected.segments[0].row if selected and selected.segments else self._get_primary_scroll_view().scroll_top
        )
        search.query = query
        search.selection_mode = "query"
        search.component.set_result(-1, 0)
        self.request_render()

    def _navigate_search(self, direction: Literal[-1, 1]) -> None:
        search = self._active_search
        if search is None or not search.query:
            return
        search.selection_mode = "previous" if direction < 0 else "next"
        self.request_render()

    def _refresh_search(self, layout: LayoutFrame) -> bool:
        search = self._active_search
        if search is None:
            return False
        scroll_view = layout.primary_scroll_view or self._implicit_scroll_view
        box = get_scroll_view_box(layout, scroll_view)
        lines = box.scroll_content_lines if box else None
        if not lines or not search.query.strip():
            search.matches = []
            search.selected_index = -1
            search.selected_key = None
            search.selection_mode = "retain"
            search.component.set_result(-1, 0)
            return False

        should_reveal = search.selection_mode != "retain"
        matches = find_alt_screen_search_matches(lines, search.query)
        exact_index = (
            next(
                (i for i, match in enumerate(matches) if get_alt_screen_search_match_key(match) == search.selected_key),
                -1,
            )
            if search.selected_key
            else -1
        )
        selected_index = -1
        if matches:
            if search.selection_mode == "query":
                selected_index = next(
                    (i for i, match in enumerate(matches) if (match.segments[0].row if match.segments else 0) >= search.anchor_row),
                    0,
                )
            elif search.selection_mode == "next":
                base = exact_index if exact_index >= 0 else min(search.selected_index, len(matches) - 1)
                selected_index = 0 if base < 0 else (base + 1) % len(matches)
            elif search.selection_mode == "previous":
                base = exact_index if exact_index >= 0 else min(search.selected_index, len(matches) - 1)
                selected_index = len(matches) - 1 if base < 0 else (base - 1 + len(matches)) % len(matches)
            else:
                selected_index = exact_index if exact_index >= 0 else min(max(0, search.selected_index), len(matches) - 1)

        search.matches = matches
        search.selected_index = selected_index
        search.selected_key = get_alt_screen_search_match_key(matches[selected_index]) if selected_index >= 0 else None
        search.selection_mode = "retain"
        search.component.set_result(selected_index, len(matches))
        if not should_reveal:
            return False

        selected = matches[selected_index] if 0 <= selected_index < len(matches) else None
        first_segment = selected.segments[0] if selected and selected.segments else None
        last_segment = selected.segments[-1] if selected and selected.segments else None
        if box is None or first_segment is None or last_segment is None or scroll_view.viewport_height <= 0:
            return False
        before = scroll_view.scroll_top
        visible_bottom = before + scroll_view.viewport_height - 1
        target = before
        if first_segment.row < before or last_segment.row > visible_bottom:
            target = first_segment.row - scroll_view.viewport_height // 3
        scroll_view.scroll_to(target, {"disable_follow": True})
        return scroll_view.scroll_top != before

    def flash(self, message: str, duration_ms: int | None = None) -> None:
        if duration_ms is None:
            self._flashes.flash(message)
        else:
            self._flashes.flash(message, duration_ms)

    def _should_defer_viewport_input_to_overlay(self) -> bool:
        return self._is_overlay_focused() and not (
            self._active_search and self._active_search.overlay and self._active_search.overlay.is_focused()
        )

    def _handle_viewport_input(self, data: str) -> dict | None:
        if data == _FOCUS_OUT:
            had_active = self._selection_press_active
            had_non_empty = had_active and self._get_selection_bounds() is not None
            self._selection_press_active = False
            self._stop_selection_auto_scroll()
            self._stop_scrollbar_hover()
            self._stop_scrollbar_drag()
            self._pressed_url = None
            self._selection_dragged = False
            if had_active:
                self._selection_anchor = None
                self._selection_focus = None
                self._selection_granularity = "character"
                self._selection_initial_range = None
                if had_non_empty:
                    self.request_render()
            self._last_click = None
            return {"consume": True}
        if data == _FOCUS_IN:
            return {"consume": True}

        wheel_event = self._parse_wheel_event(data)
        if wheel_event:
            if self._should_defer_viewport_input_to_overlay():
                return None
            self._route_wheel(wheel_event)
            return {"consume": True}
        mouse_event = self._parse_sgr_mouse_event(data)
        if mouse_event:
            if self._handle_right_click_paste(mouse_event):
                return {"consume": True}
            handled = self._handle_scrollbar_mouse_event(mouse_event)
            if self._scrollbar_drag is None:
                self._update_scrollbar_hover(mouse_event.x, mouse_event.y)
            if not handled:
                self._handle_selection_mouse_event(mouse_event)
            return {"consume": True}
        if self._is_mouse_sequence(data):
            return {"consume": True}

        keybindings = get_keybindings()
        is_release = is_key_release(data)
        if keybindings.matches(data, "tui.altScreen.search"):
            if not is_release:
                self._open_search()
            return {"consume": True}
        if self._active_search and self._active_search.overlay and self._active_search.overlay.is_focused():
            if keybindings.matches(data, "tui.altScreen.searchNext"):
                if not is_release:
                    self._navigate_search(1)
                return {"consume": True}
            if keybindings.matches(data, "tui.altScreen.searchPrevious"):
                if not is_release:
                    self._navigate_search(-1)
                return {"consume": True}
            if keybindings.matches(data, "tui.altScreen.searchClose"):
                if not is_release:
                    self._close_search()
                return {"consume": True}
        if self._should_defer_viewport_input_to_overlay():
            return None
        viewport_height = self._get_primary_scroll_view().viewport_height
        if keybindings.matches(data, "tui.altScreen.pageUp"):
            if not is_release:
                self.scroll_by(-max(1, viewport_height - _PAGE_SCROLL_OVERLAP))
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.pageDown"):
            if not is_release:
                self.scroll_by(max(1, viewport_height - _PAGE_SCROLL_OVERLAP))
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.halfPageUp"):
            if not is_release:
                self.scroll_by(-max(1, viewport_height // 2))
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.halfPageDown"):
            if not is_release:
                self.scroll_by(max(1, viewport_height // 2))
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.lineUp"):
            if not is_release:
                self.scroll_by(-1)
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.lineDown"):
            if not is_release:
                self.scroll_by(1)
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.previousPrompt"):
            if not is_release:
                self._scroll_to_prompt(-1)
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.nextPrompt"):
            if not is_release:
                self._scroll_to_prompt(1)
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.top"):
            if not is_release:
                self.scroll_to_top()
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.bottom"):
            if not is_release:
                self.scroll_to_bottom()
            return {"consume": True}
        return None

    def _parse_wheel_event(self, data: str) -> _WheelEvent | None:
        sgr = _SGR_WHEEL_RE.match(data)
        if sgr:
            button = int(sgr.group(1))
            if (button & 64) == 0:
                return None
            direction = button & 3
            if direction not in (0, 1):
                return None
            return _WheelEvent(
                direction=-1 if direction == 0 else 1,
                x=int(sgr.group(2)) - 1,
                y=int(sgr.group(3)) - 1,
            )
        if len(data) == 6 and data.startswith("\x1b[M"):
            button = ord(data[3]) - 32
            if (button & 64) == 0:
                return None
            direction = button & 3
            if direction not in (0, 1):
                return None
            return _WheelEvent(
                direction=-1 if direction == 0 else 1,
                x=ord(data[4]) - 33,
                y=ord(data[5]) - 33,
            )
        return None

    def _route_wheel(self, event: _WheelEvent) -> None:
        remaining = event.direction * self._wheel_scroll_lines
        seen: set[int] = set()
        for scroll_view in get_scroll_views_at(self._current_layout, event.x, event.y) if self._current_layout else []:
            seen.add(id(scroll_view))
            remaining = scroll_view.scroll_by(remaining)
            if remaining == 0 or scroll_view.overscroll == "contain":
                break
        primary = self._get_primary_scroll_view()
        if remaining != 0 and id(primary) not in seen:
            primary.scroll_by(remaining)
        self._update_scrollbar_hover(event.x, event.y)
        self.request_render()

    def _parse_sgr_mouse_event(self, data: str) -> _SgrMouseEvent | None:
        match = _SGR_MOUSE_RE.match(data)
        if not match:
            return None
        return _SgrMouseEvent(
            button=int(match.group(1)),
            x=int(match.group(2)) - 1,
            y=int(match.group(3)) - 1,
            release=match.group(4) == "m",
        )

    def _handle_right_click_paste(self, event: _SgrMouseEvent) -> bool:
        if (
            not self._on_right_click_paste
            or sys.platform != "win32"
            or (os.environ.get("TERM_PROGRAM") or "").lower() == "vscode"
            or event.release
            or event.button != 2
        ):
            return False
        try:
            self._on_right_click_paste()
        except Exception:
            pass
        return True

    def _get_scrollbar_target_at(self, x: int, y: int) -> _ScrollbarTarget | None:
        if self.has_overlay() or self._current_layout is None:
            return None
        for scroll_view in get_scroll_views_at(self._current_layout, x, y):
            box = get_scroll_view_box(self._current_layout, scroll_view)
            geometry = get_scrollbar_geometry(box) if box else None
            if geometry and x == geometry.column and geometry.thumb_top <= y < geometry.thumb_top + geometry.thumb_height:
                return _ScrollbarTarget(scroll_view=scroll_view, geometry=geometry)
        return None

    def _set_scrollbar_hover(self, scroll_view: ScrollView | None) -> None:
        if scroll_view is self._scrollbar_hover:
            return
        if self._scrollbar_hover is not None:
            self._scrollbar_hover.set_scrollbar_active(False)
        self._scrollbar_hover = scroll_view
        if self._scrollbar_hover is not None:
            self._scrollbar_hover.set_scrollbar_active(True)

    def _update_scrollbar_hover(self, x: int, y: int) -> None:
        target = self._get_scrollbar_target_at(x, y)
        self._set_scrollbar_hover(target.scroll_view if target else None)

    def _stop_scrollbar_hover(self) -> None:
        self._set_scrollbar_hover(None)

    def _handle_scrollbar_mouse_event(self, event: _SgrMouseEvent) -> bool:
        if self._scrollbar_drag:
            if event.release:
                self._stop_scrollbar_drag()
                return True
            box = (
                get_scroll_view_box(self._current_layout, self._scrollbar_drag.scroll_view)
                if self._current_layout
                else None
            )
            geometry = get_scrollbar_geometry(box) if box else None
            if geometry:
                max_thumb_offset = geometry.track_height - geometry.thumb_height
                thumb_offset = max(
                    0,
                    min(max_thumb_offset, event.y - geometry.track_top - self._scrollbar_drag.grab_offset),
                )
                scroll_top = 0 if max_thumb_offset == 0 else round((thumb_offset / max_thumb_offset) * geometry.max_scroll_top)
                self._scrollbar_drag.scroll_view.scroll_to(scroll_top)
            return True

        if event.release or (event.button & 32) != 0 or (event.button & 3) != 0:
            return False
        target = self._get_scrollbar_target_at(event.x, event.y)
        if target is None:
            return False
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._selection_anchor = None
        self._selection_focus = None
        self._selection_granularity = "character"
        self._selection_initial_range = None
        self._last_click = None
        self._pressed_url = None
        self._selection_dragged = False
        self._set_scrollbar_hover(target.scroll_view)
        self._scrollbar_drag = _ScrollbarDrag(
            scroll_view=target.scroll_view,
            grab_offset=event.y - target.geometry.thumb_top,
        )
        return True

    def _stop_scrollbar_drag(self) -> None:
        self._scrollbar_drag = None

    def _get_scroll_selection_point(self, scroll_view: ScrollView, x: int, y: int) -> _SelectionPoint | None:
        if self._current_layout is None:
            return None
        box = get_scroll_view_box(self._current_layout, scroll_view)
        if box is None or box.rect.height <= 0 or box.clip.height <= 0:
            return None
        visible_top = max(0, box.rect.y, box.clip.y)
        visible_bottom = min(
            self.terminal.rows - 1,
            box.rect.y + box.rect.height - 1,
            box.clip.y + box.clip.height - 1,
        )
        if visible_bottom < visible_top:
            return None
        pointer_row = max(visible_top, min(visible_bottom, y))
        max_content_row = max(0, (len(box.scroll_content_lines) if box.scroll_content_lines else 1) - 1)
        return _SelectionPoint(
            row=max(0, min(max_content_row, scroll_view.scroll_top + pointer_row - box.rect.y)),
            col=max(0, min(box.rect.width - 1, x - box.rect.x)),
            scroll_view=scroll_view,
        )

    def _get_selection_point(self, event: _SgrMouseEvent, scroll_view: ScrollView | None = None) -> _SelectionPoint:
        if scroll_view is not None:
            point = self._get_scroll_selection_point(scroll_view, event.x, event.y)
            if point is not None:
                return point
        return _SelectionPoint(
            row=max(0, min(self.terminal.rows - 1, event.y)),
            col=max(0, min(self.terminal.columns - 1, event.x)),
        )

    def _get_selection_source_line(self, point: _SelectionPoint) -> str:
        if point.scroll_view is not None and self._current_layout is not None:
            box = get_scroll_view_box(self._current_layout, point.scroll_view)
            if box and box.scroll_content_lines is not None:
                if 0 <= point.row < len(box.scroll_content_lines):
                    return box.scroll_content_lines[point.row]
                return ""
        if 0 <= point.row < len(self._previous_screen):
            return self._previous_screen[point.row]
        return ""

    def _get_word_selection(self, point: _SelectionPoint) -> _SelectionRange | None:
        line = strip_terminal_sequences(self._get_selection_source_line(point))
        start = 0
        for segment in _segment_words(line):
            end = start + visible_width(segment.segment)
            if start <= point.col < end:
                return _SelectionRange(
                    start=_SelectionPoint(row=point.row, col=start, scroll_view=point.scroll_view),
                    end=_SelectionPoint(row=point.row, col=end, scroll_view=point.scroll_view, boundary=True),
                )
            start = end
        return None

    def _get_line_selection(self, point: _SelectionPoint) -> _SelectionRange:
        return _SelectionRange(
            start=_SelectionPoint(row=point.row, col=0, scroll_view=point.scroll_view),
            end=_SelectionPoint(
                row=point.row,
                col=visible_width(self._get_selection_source_line(point)),
                scroll_view=point.scroll_view,
                boundary=True,
            ),
        )

    def _update_selection_focus(self, point: _SelectionPoint) -> None:
        if self._selection_granularity == "character" or self._selection_initial_range is None:
            self._selection_focus = point
            return
        rng = self._get_word_selection(point) if self._selection_granularity == "word" else self._get_line_selection(point)
        if rng is None:
            return
        initial = self._selection_initial_range
        target_before = rng.start.row < initial.start.row or (
            rng.start.row == initial.start.row and rng.start.col < initial.start.col
        )
        if target_before:
            self._selection_anchor = initial.end
            self._selection_focus = rng.start
        else:
            self._selection_anchor = initial.start
            self._selection_focus = rng.end

    def _get_click_count(self, point: _SelectionPoint, word: _SelectionRange | None) -> int:
        now = time.time() * 1000
        previous = self._last_click
        count = (
            (previous.count % 3) + 1
            if (
                word
                and previous
                and now - previous.timestamp <= _DOUBLE_CLICK_INTERVAL_MS
                and previous.row == point.row
                and previous.scroll_view is point.scroll_view
                and previous.word_start == word.start.col
                and previous.word_end == word.end.col
            )
            else 1
        )
        self._last_click = (
            _ClickTarget(
                timestamp=now,
                count=count,
                row=point.row,
                scroll_view=point.scroll_view,
                word_start=word.start.col,
                word_end=word.end.col,
            )
            if word
            else None
        )
        return count

    def _update_selection_auto_scroll(self, event: _SgrMouseEvent) -> None:
        scroll_view = self._selection_anchor.scroll_view if self._selection_anchor else None
        if scroll_view is None or self._current_layout is None:
            self._stop_selection_auto_scroll()
            return
        box = get_scroll_view_box(self._current_layout, scroll_view)
        if box is None or box.rect.height <= 0 or box.clip.height <= 0:
            self._stop_selection_auto_scroll()
            return
        visible_top = max(0, box.rect.y, box.clip.y)
        visible_bottom = min(
            self.terminal.rows - 1,
            box.rect.y + box.rect.height - 1,
            box.clip.y + box.clip.height - 1,
        )
        self._selection_drag_pointer = {"x": event.x, "y": event.y}
        self._selection_auto_scroll_direction = -1 if event.y <= visible_top else (1 if event.y >= visible_bottom else 0)
        if self._selection_auto_scroll_direction == 0:
            self._stop_selection_auto_scroll()
            return
        if self._selection_auto_scroll_timer is not None:
            return
        self._selection_auto_scroll_timer = threading.Timer(0.05, self._auto_scroll_selection_tick)
        self._selection_auto_scroll_timer.daemon = True
        self._selection_auto_scroll_timer.start()

    def _auto_scroll_selection_tick(self) -> None:
        self._selection_auto_scroll_timer = None
        self._auto_scroll_selection()
        if self._selection_auto_scroll_direction != 0:
            self._selection_auto_scroll_timer = threading.Timer(0.05, self._auto_scroll_selection_tick)
            self._selection_auto_scroll_timer.daemon = True
            self._selection_auto_scroll_timer.start()

    def _auto_scroll_selection(self) -> None:
        scroll_view = self._selection_anchor.scroll_view if self._selection_anchor else None
        pointer = self._selection_drag_pointer
        direction = self._selection_auto_scroll_direction
        if scroll_view is None or pointer is None or direction == 0:
            self._stop_selection_auto_scroll()
            return
        remaining = scroll_view.scroll_by(direction)
        if remaining == direction:
            self._stop_selection_auto_scroll()
            return
        point = self._get_scroll_selection_point(scroll_view, pointer["x"], pointer["y"])
        if point:
            self._update_selection_focus(point)
        self.request_render()

    def _stop_selection_auto_scroll(self) -> None:
        if self._selection_auto_scroll_timer is not None:
            self._selection_auto_scroll_timer.cancel()
            self._selection_auto_scroll_timer = None
        self._selection_auto_scroll_direction = 0
        self._selection_drag_pointer = None

    def _handle_selection_mouse_event(self, event: _SgrMouseEvent) -> None:
        button = event.button & 3
        if button != 0 and not (event.release and button == 3):
            return
        anchor_scroll_view = self._selection_anchor.scroll_view if self._selection_anchor else None
        point = self._get_selection_point(event, anchor_scroll_view)
        if event.release:
            if not self._selection_press_active:
                return
            self._selection_press_active = False
            self._stop_selection_auto_scroll()
            if self._selection_anchor is None:
                return
            self._update_selection_focus(point)
            clicked_url = (
                self._pressed_url
                if (
                    not self._selection_dragged
                    and self._selection_anchor.scroll_view is point.scroll_view
                    and self._selection_anchor.row == point.row
                    and self._selection_anchor.col == point.col
                )
                else None
            )
            self._pressed_url = None
            if clicked_url and self._open_url:
                self._selection_anchor = None
                self._selection_focus = None
                try:
                    self._open_url(clicked_url)
                except Exception:
                    pass
                self.request_render()
                return
            self._copy_selection_to_clipboard()
            self.request_render()
            return
        if (event.button & 32) != 0:
            if not self._selection_press_active or self._selection_anchor is None:
                return
            self._selection_dragged = True
            self._last_click = None
            self._pressed_url = None
            self._update_selection_focus(point)
            self._update_selection_auto_scroll(event)
            self.request_render()
            return
        self._stop_selection_auto_scroll()
        self._selection_press_active = True
        views = (
            get_scroll_views_at(self._current_layout, event.x, event.y)
            if not self.has_overlay() and self._current_layout
            else []
        )
        scroll_view = views[0] if views else None
        anchor = self._get_selection_point(event, scroll_view)
        word = self._get_word_selection(anchor)
        click_count = self._get_click_count(anchor, word)
        rng = word if click_count == 2 else (self._get_line_selection(anchor) if click_count == 3 else None)
        self._selection_granularity = "word" if rng and click_count == 2 else ("line" if rng else "character")
        self._selection_initial_range = rng
        self._selection_anchor = rng.start if rng else anchor
        self._selection_focus = rng.end if rng else anchor
        self._selection_dragged = False
        screen_row = max(0, min(self.terminal.rows - 1, event.y))
        screen_col = max(0, min(self.terminal.columns - 1, event.x))
        self._pressed_url = None if rng else get_osc8_link_at_column(
            self._previous_screen[screen_row] if screen_row < len(self._previous_screen) else "",
            screen_col,
        )
        self.request_render()

    def _get_selection_bounds(self) -> dict | None:
        if self._selection_anchor is None or self._selection_focus is None:
            return None
        if self._selection_anchor.scroll_view is not self._selection_focus.scroll_view:
            return None
        if (
            self._selection_anchor.row == self._selection_focus.row
            and self._selection_anchor.col == self._selection_focus.col
        ):
            return None
        anchor_before = self._selection_anchor.row < self._selection_focus.row or (
            self._selection_anchor.row == self._selection_focus.row
            and self._selection_anchor.col < self._selection_focus.col
        )
        return (
            {"start": self._selection_anchor, "end": self._selection_focus}
            if anchor_before
            else {"start": self._selection_focus, "end": self._selection_anchor}
        )

    def _get_selection_columns(
        self,
        line: str,
        row: int,
        selection: dict,
        min_column: int = 0,
        max_column: int | None = None,
    ) -> dict[str, int]:
        line_width = visible_width(line)
        if max_column is None:
            max_column = line_width
        start = max(0, min_column)
        end = min(line_width, max_column)
        if row == selection["start"].row:
            cell = get_grapheme_cell_range(line, selection["start"].col)
            start = cell[0] if cell else min(selection["start"].col, line_width)
        if row == selection["end"].row:
            if selection["end"].boundary:
                end = min(selection["end"].col, line_width)
            else:
                cell = get_grapheme_cell_range(line, selection["end"].col)
                end = cell[1] if cell else min(selection["end"].col + 1, line_width)
        return {"start": max(min_column, start), "end": min(max_column, end)}

    def _copy_selection_to_clipboard(self) -> None:
        selection = self._get_selection_bounds()
        if selection is None:
            return
        source_lines: list[str] = self._previous_screen
        if selection["start"].scroll_view is not None:
            if self._current_layout is None:
                return
            box = get_scroll_view_box(self._current_layout, selection["start"].scroll_view)
            if box is None or box.scroll_content_lines is None:
                return
            source_lines = list(box.scroll_content_lines)
        lines: list[str] = []
        for row in range(selection["start"].row, selection["end"].row + 1):
            line = source_lines[row] if row < len(source_lines) else ""
            columns = self._get_selection_columns(line, row, selection)
            lines.append(
                strip_terminal_sequences(
                    slice_by_column(line, columns["start"], max(0, columns["end"] - columns["start"]), True)
                ).rstrip()
            )
        text = "\n".join(lines)
        if not text:
            return
        if self._copy_selection:
            try:
                ok = self._copy_selection(text)
            except Exception:
                ok = False
            self.flash("Copied!" if ok else "Copy failed")
            return
        encoded = base64.b64encode(text.encode()).decode()
        self.terminal.write(f"\x1b]52;c;{encoded}\x07")
        self.flash("Copied!")

    def _apply_search_text_highlight(self, text: str, current: bool) -> str:
        style = self._search_current_match_style if current else self._search_match_style
        result = ""
        plain_start = 0
        index = 0
        while index < len(text):
            ansi = extract_ansi_code(text, index)
            if not ansi:
                index += 1
                continue
            if index > plain_start:
                result += style(text[plain_start:index])
            result += ansi.code
            index += ansi.length
            plain_start = index
        if plain_start < len(text):
            result += style(text[plain_start:])
        return result

    def _apply_search_highlights(self, screen: list[str], layout: LayoutFrame) -> list[str]:
        search = self._active_search
        if search is None or search.selected_index < 0 or not search.matches:
            return screen
        scroll_view = layout.primary_scroll_view or self._implicit_scroll_view
        box = get_scroll_view_box(layout, scroll_view)
        if box is None:
            return screen

        ranges_by_row: dict[int, list[_SearchHighlightRange]] = {}
        scrollbar_column = None
        geometry = get_scrollbar_geometry(box)
        if geometry:
            scrollbar_column = geometry.column
        min_row = max(0, box.rect.y, box.clip.y)
        max_row = min(len(screen), box.rect.y + box.rect.height, box.clip.y + box.clip.height)
        min_column = max(0, box.rect.x, box.clip.x)
        max_column = min(
            self.terminal.columns,
            box.rect.x + box.rect.width,
            box.clip.x + box.clip.width,
            scrollbar_column if scrollbar_column is not None else 10**9,
        )
        for match_index, match in enumerate(search.matches):
            for segment in match.segments:
                row = box.rect.y + segment.row - scroll_view.scroll_top
                if row < min_row or row >= max_row:
                    continue
                start_col = max(min_column, box.rect.x + segment.start_col)
                end_col = min(max_column, box.rect.x + segment.end_col)
                if end_col <= start_col:
                    continue
                ranges_by_row.setdefault(row, []).append(
                    _SearchHighlightRange(start_col=start_col, end_col=end_col, current=match_index == search.selected_index)
                )

        result = list(screen)
        for row, ranges in ranges_by_row.items():
            line = result[row] if row < len(result) else ""
            if is_image_line(line):
                continue
            line_width = visible_width(line)
            for rng in sorted(ranges, key=lambda item: item.start_col, reverse=True):
                start_col = min(rng.start_col, line_width)
                end_col = min(rng.end_col, line_width)
                if end_col <= start_col:
                    continue
                before = slice_by_column(line, 0, start_col, True)
                highlighted = slice_by_column(line, start_col, end_col - start_col, True)
                after = slice_by_column(line, end_col, max(0, line_width - end_col), True)
                line = f"{before}{self._apply_search_text_highlight(highlighted, rng.current)}{after}"
            result[row] = line
        return result

    def _apply_selection_highlight(self, text: str) -> str:
        result = "\x1b[7m"
        index = 0
        while index < len(text):
            ansi = extract_ansi_code(text, index)
            if not ansi:
                result += text[index]
                index += 1
                continue
            result += ansi.code
            if ansi.code.endswith("m"):
                result += "\x1b[7m"
            index += ansi.length
        return f"{result}\x1b[27m"

    def _apply_selection(self, screen: list[str], layout: LayoutFrame | None = None) -> list[str]:
        selection = self._get_selection_bounds()
        if selection is None:
            return screen
        layout = layout if layout is not None else self._current_layout
        screen_selection = selection
        min_row = 0
        max_row = len(screen) - 1
        min_column = 0
        max_column = self.terminal.columns
        if selection["start"].scroll_view is not None:
            if layout is None:
                return screen
            box = get_scroll_view_box(layout, selection["start"].scroll_view)
            if box is None:
                return screen
            min_row = max(0, box.rect.y, box.clip.y)
            max_row = min(len(screen) - 1, box.rect.y + box.rect.height - 1, box.clip.y + box.clip.height - 1)
            min_column = max(0, box.rect.x, box.clip.x)
            max_column = min(self.terminal.columns, box.rect.x + box.rect.width, box.clip.x + box.clip.width)
            screen_selection = {
                "start": _SelectionPoint(
                    row=box.rect.y + selection["start"].row - selection["start"].scroll_view.scroll_top,
                    col=box.rect.x + selection["start"].col,
                    scroll_view=selection["start"].scroll_view,
                    boundary=selection["start"].boundary,
                ),
                "end": _SelectionPoint(
                    row=box.rect.y + selection["end"].row - selection["start"].scroll_view.scroll_top,
                    col=box.rect.x + selection["end"].col,
                    scroll_view=selection["end"].scroll_view,
                    boundary=selection["end"].boundary,
                ),
            }
        result = []
        for row, line in enumerate(screen):
            if (
                row < min_row
                or row > max_row
                or row < screen_selection["start"].row
                or row > screen_selection["end"].row
                or is_image_line(line)
            ):
                result.append(line)
                continue
            line_width = visible_width(line)
            columns = self._get_selection_columns(line, row, screen_selection, min_column, max_column)
            if columns["end"] <= columns["start"]:
                result.append(line)
                continue
            before = slice_by_column(line, 0, columns["start"], True)
            selected = slice_by_column(line, columns["start"], columns["end"] - columns["start"], True)
            after = slice_by_column(line, columns["end"], max(0, line_width - columns["end"]), True)
            result.append(f"{before}{self._apply_selection_highlight(selected)}{after}")
        return result

    def _is_mouse_sequence(self, data: str) -> bool:
        return bool(_SGR_MOUSE_RE.match(data)) or (len(data) == 6 and data.startswith("\x1b[M"))

    def _composite_flashes(self, screen: list[str], width: int, height: int) -> list[str]:
        flash_lines = self._flashes.render(width)[-height:]
        if not flash_lines:
            return screen
        result = list(screen)
        while len(result) < height:
            result.append("")
        for row, line in enumerate(flash_lines):
            flash_width = visible_width(line)
            if flash_width == 0:
                continue
            result[row] = composite_tui_line(result[row] if row < len(result) else "", line, width - flash_width, flash_width, width)
        return result

    def _do_render(self) -> None:
        if self._stopped or not self._alt_screen_active:
            return
        width = max(1, self.terminal.columns)
        height = max(1, self.terminal.rows)
        root = self._layout_root if self._layout_root is not None else self._implicit_scroll_view
        next_layout = render_layout_frame(root, width, height, lambda: self.request_render())
        if self._refresh_search(next_layout):
            next_layout = render_layout_frame(root, width, height, lambda: self.request_render())
        screen = [_OSC133_ZONE_PREFIX.sub("", line) for line in next_layout.lines]
        screen = self._apply_search_highlights(screen, next_layout)
        screen = self._composite_overlays(screen, width, height)
        if len(screen) > height:
            screen = screen[len(screen) - height :]
        screen = self._apply_selection(screen, next_layout)
        screen = self._composite_flashes(screen, width, height)

        cursor_pos = self._extract_cursor_position(screen, height)
        screen = [
            line if is_image_line(line) or visible_width(line) <= width else slice_by_column(line, 0, width, True)
            for line in self._apply_line_resets(screen)
        ]

        full_redraw = (
            not self._previous_screen
            or self._previous_screen_width != width
            or self._previous_screen_height != height
        )
        images_need_redraw = any(
            line != (self._previous_screen[row] if row < len(self._previous_screen) else "")
            and (
                is_image_line(line)
                or is_image_line(self._previous_screen[row] if row < len(self._previous_screen) else "")
            )
            for row, line in enumerate(screen)
        )
        redraw_images = full_redraw or images_need_redraw
        had_uploaded = bool(self._uploaded_kitty_images)
        if redraw_images and self._image_protocol == "kitty":
            prepared_lines, evicted = self._prepare_kitty_screen(screen)
        else:
            prepared_lines, evicted = screen, ""

        buf = _BEGIN_SYNCHRONIZED_OUTPUT
        if full_redraw:
            self._full_redraw_count += 1
            clear_images = (
                delete_all_kitty_placements()
                if self._image_protocol == "kitty" and had_uploaded
                else self._delete_kitty_images()
            )
            buf += f"{clear_images}\x1b[2J"
        elif images_need_redraw:
            if self._image_protocol == "iterm2":
                buf += "\x1b[2J"
            elif self._image_protocol == "kitty":
                buf += delete_all_kitty_placements()
        buf += evicted

        for row in range(height):
            if not full_redraw and not images_need_redraw and row < len(self._previous_screen) and screen[row] == self._previous_screen[row]:
                continue
            line = prepared_lines[row] if row < len(prepared_lines) else ""
            buf += f"\x1b[{row + 1};1H\x1b[2K{line}"

        if cursor_pos:
            buf += f"\x1b[{cursor_pos[0] + 1};{min(width, cursor_pos[1]) + 1}H"
            buf += "\x1b[?25h" if self.get_show_hardware_cursor() else "\x1b[?25l"
        else:
            buf += "\x1b[?25l"
        buf += _END_SYNCHRONIZED_OUTPUT
        self.terminal.write(buf)

        self._previous_screen = screen
        self._previous_screen_width = width
        self._previous_screen_height = height
        self._current_layout = next_layout
