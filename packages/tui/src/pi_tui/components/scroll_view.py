"""ScrollView — mirrors components/scroll-view.ts"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Callable, Literal

from ..layout_node import ScrollLayoutNode
from ..tui import Container

ScrollViewScrollbar = Literal["hidden", "auto", "always"]


@dataclass
class ScrollViewOptions:
    axis: Literal["vertical"] | None = None
    follow: Literal["none", "end"] = "none"
    primary: bool = False
    overscroll: Literal["chain", "contain"] = "chain"
    scrollbar: ScrollViewScrollbar = "hidden"
    scrollbar_style: Callable[[str], str] | None = None
    scrollbar_hide_delay_ms: int = 1000


@dataclass
class ScrollViewScrollToOptions:
    disable_follow: bool = False


def _default_scrollbar_style(text: str) -> str:
    return f"\x1b[100m{text}\x1b[49m"


class ScrollView(Container):
    def __init__(
        self,
        component: object,
        options: ScrollViewOptions | dict | None = None,
    ) -> None:
        super().__init__()
        if isinstance(options, dict):
            options = ScrollViewOptions(**options)
        opts = options or ScrollViewOptions()
        if opts.axis is not None and opts.axis != "vertical":
            raise ValueError(f"Unsupported ScrollView axis: {opts.axis}")

        self._child = component
        self.children.append(component)
        self._follow_end = (opts.follow or "none") == "end"
        self._following_end = self._follow_end
        self.primary = opts.primary
        self.overscroll = opts.overscroll
        self.scrollbar_style = opts.scrollbar_style or _default_scrollbar_style
        self._current_scrollbar: ScrollViewScrollbar = opts.scrollbar
        self._scrollbar_hide_delay_ms = max(0, math.floor(opts.scrollbar_hide_delay_ms))
        self._current_scroll_top = 0
        self._content_height = 0
        self._current_viewport_height = 0
        self._follow_suppressed_at_end = False
        self._request_render: Callable[[], None] | None = None
        self._transient_scrollbar_visible = False
        self._scrollbar_active = False
        self._scrollbar_hide_timer: threading.Timer | None = None

    @property
    def scroll_top(self) -> int:
        return self._current_scroll_top

    @property
    def is_following_end(self) -> bool:
        return self._following_end

    @property
    def viewport_height(self) -> int:
        return self._current_viewport_height

    @property
    def scrollbar(self) -> ScrollViewScrollbar:
        return self._current_scrollbar

    @property
    def is_scrollbar_visible(self) -> bool:
        if self.scrollbar == "always":
            return self._current_viewport_height > 0
        return (
            self.scrollbar == "auto"
            and self._content_height > self._current_viewport_height
            and self._transient_scrollbar_visible
        )

    def set_scrollbar(self, scrollbar: ScrollViewScrollbar) -> None:
        if scrollbar == self._current_scrollbar:
            return
        self._current_scrollbar = scrollbar
        if scrollbar != "auto":
            self._hide_transient_scrollbar()
        elif self._scrollbar_active:
            self._mark_scrollbar_activity()
        if self._request_render:
            self._request_render()

    def get_content_width(self, width: int) -> int:
        return width - 1 if self.scrollbar == "always" and width > 1 else width

    def _mark_scrollbar_activity(self) -> None:
        if self.scrollbar != "auto" or self._content_height <= self._current_viewport_height:
            return
        self._transient_scrollbar_visible = True
        if self._scrollbar_hide_timer is not None:
            self._scrollbar_hide_timer.cancel()
            self._scrollbar_hide_timer = None
        if self._scrollbar_active:
            return

        def _hide() -> None:
            self._scrollbar_hide_timer = None
            self._transient_scrollbar_visible = False
            if self._request_render:
                self._request_render()

        self._scrollbar_hide_timer = threading.Timer(self._scrollbar_hide_delay_ms / 1000.0, _hide)
        self._scrollbar_hide_timer.daemon = True
        self._scrollbar_hide_timer.start()

    def _hide_transient_scrollbar(self) -> None:
        self._transient_scrollbar_visible = False
        if self._scrollbar_hide_timer is None:
            return
        self._scrollbar_hide_timer.cancel()
        self._scrollbar_hide_timer = None

    def set_scrollbar_active(self, active: bool) -> None:
        if active == self._scrollbar_active:
            return
        self._scrollbar_active = active
        self._mark_scrollbar_activity()

    def scroll_to(
        self,
        scroll_top: float,
        options: ScrollViewScrollToOptions | dict | None = None,
    ) -> None:
        if isinstance(options, dict):
            options = ScrollViewScrollToOptions(**options)
        opts = options or ScrollViewScrollToOptions()
        requested = math.trunc(scroll_top) if math.isfinite(scroll_top) else self._current_scroll_top
        max_scroll_top = max(0, self._content_height - self._current_viewport_height)
        nxt = max(0, min(max_scroll_top, requested))
        next_follow_suppressed = opts.disable_follow is True and nxt == max_scroll_top
        next_following_end = not next_follow_suppressed and self._follow_end and nxt == max_scroll_top
        if (
            nxt == self._current_scroll_top
            and next_following_end == self._following_end
            and next_follow_suppressed == self._follow_suppressed_at_end
        ):
            return
        moved = nxt != self._current_scroll_top
        self._current_scroll_top = nxt
        self._following_end = next_following_end
        self._follow_suppressed_at_end = next_follow_suppressed
        if moved:
            self._mark_scrollbar_activity()
        if self._request_render:
            self._request_render()

    def scroll_by(self, lines: float) -> int:
        requested = math.trunc(lines) if math.isfinite(lines) else 0
        if requested == 0:
            return 0
        max_scroll_top = max(0, self._content_height - self._current_viewport_height)
        start = max_scroll_top if self._following_end else self._current_scroll_top
        nxt = max(0, min(max_scroll_top, start + requested))
        moved = nxt - start
        was_following_end = self._following_end
        self._current_scroll_top = nxt
        self._following_end = self._follow_end and nxt == max_scroll_top
        self._follow_suppressed_at_end = False
        if moved != 0:
            self._mark_scrollbar_activity()
        if moved != 0 or self._following_end != was_following_end:
            if self._request_render:
                self._request_render()
        return requested - moved

    def scroll_to_start(self) -> None:
        changed = self._current_scroll_top != 0 or self._following_end != (
            self._follow_end and self._content_height <= self._current_viewport_height
        )
        self._current_scroll_top = 0
        self._following_end = self._follow_end and self._content_height <= self._current_viewport_height
        self._follow_suppressed_at_end = False
        if changed:
            self._mark_scrollbar_activity()
            if self._request_render:
                self._request_render()

    def scroll_to_end(self) -> None:
        nxt = max(0, self._content_height - self._current_viewport_height)
        changed = self._current_scroll_top != nxt or self._following_end != self._follow_end
        self._current_scroll_top = nxt
        self._following_end = self._follow_end
        self._follow_suppressed_at_end = False
        if changed:
            self._mark_scrollbar_activity()
            if self._request_render:
                self._request_render()

    def update_layout(
        self,
        content_height: int,
        viewport_height: int,
        request_render: Callable[[], None],
    ) -> None:
        self._content_height = max(0, math.floor(content_height))
        self._current_viewport_height = max(0, math.floor(viewport_height))
        self._request_render = request_render
        max_scroll_top = max(0, self._content_height - self._current_viewport_height)
        if self._following_end:
            self._current_scroll_top = max_scroll_top
        else:
            self._current_scroll_top = max(0, min(self._current_scroll_top, max_scroll_top))
        if self._current_scroll_top < max_scroll_top:
            self._follow_suppressed_at_end = False
        if self._follow_end and self._current_scroll_top == max_scroll_top and not self._follow_suppressed_at_end:
            self._following_end = True
        if self._content_height <= self._current_viewport_height:
            self._hide_transient_scrollbar()

    def add_child(self, _component: object) -> None:
        raise RuntimeError("ScrollView has exactly one child")

    def remove_child(self, _component: object) -> None:
        raise RuntimeError("ScrollView child cannot be removed")

    def clear(self) -> None:
        raise RuntimeError("ScrollView child cannot be cleared")

    def render(self, width: int) -> list[str]:
        content_width = self.get_content_width(width)
        lines = self._child.render(content_width) if hasattr(self._child, "render") else []
        if content_width == width:
            return lines
        return [f"{line} " for line in lines]

    def layout_node(self) -> ScrollLayoutNode:
        return ScrollLayoutNode(type="scroll", component=self._child, state=self)
