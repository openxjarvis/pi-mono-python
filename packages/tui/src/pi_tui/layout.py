"""Viewport layout engine — mirrors packages/tui/src/layout.ts"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

from .components.stack import allocate_stack_sizes, visible_stack_entries
from .layout_node import get_layout_node
from .terminal_image import crop_kitty_image_line, get_kitty_image_metadata, is_image_line
from .tui import CURSOR_MARKER, composite_tui_line
from .utils import extract_ansi_code, get_grapheme_cell_range, slice_by_column, visible_width

_OSC133_ZONE_PREFIX = re.compile(r"^(?:\x1b\]133;[ABC](?:\x07|\x1b\\))+")


@dataclass
class LayoutRect:
    x: int
    y: int
    width: int
    height: int


@dataclass
class LayoutBox:
    component: object
    rect: LayoutRect
    clip: LayoutRect
    children: list[LayoutBox] = field(default_factory=list)
    parent: LayoutBox | None = None
    lines: list[str] | None = None
    line_offset: int | None = None
    scroll_view: object | None = None
    scroll_content_lines: list[str] | None = None
    layer: int = 0


@dataclass
class LayoutFrame:
    root: LayoutBox
    width: int
    height: int
    lines: list[str]
    primary_scroll_view: object | None = None


@dataclass
class ScrollbarGeometry:
    column: int
    track_top: int
    track_height: int
    thumb_top: int
    thumb_height: int
    max_scroll_top: int


@dataclass
class _LayoutContext:
    viewport: dict
    render_cache: dict
    request_render: Callable[[], None]
    primary_scroll_view: object | None = None


def _intersect(a: LayoutRect, b: LayoutRect) -> LayoutRect:
    x = max(a.x, b.x)
    y = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)
    return LayoutRect(x=x, y=y, width=max(0, right - x), height=max(0, bottom - y))


def _render_cached(context: _LayoutContext, component: object, width: int) -> list[str]:
    safe_width = max(1, math.floor(width))
    widths = context.render_cache.get(component)
    if widths is None:
        widths = {}
        context.render_cache[component] = widths
    lines = widths.get(safe_width)
    if lines is None:
        lines = component.render(safe_width) if hasattr(component, "render") else []
        widths[safe_width] = lines
    return lines


def _measure_height(context: _LayoutContext, component: object, width: int) -> int:
    return len(_render_cached(context, component, width))


def _measure_width(context: _LayoutContext, component: object, width: int) -> int:
    return max((visible_width(line) for line in _render_cached(context, component, width)), default=0)


def _with_parent(box: LayoutBox, parent: LayoutBox) -> LayoutBox:
    box.parent = parent
    return box


def _translate_box(box: LayoutBox, delta_y: int) -> None:
    box.rect.y += delta_y
    for child in box.children:
        _translate_box(child, delta_y)


def _update_clips(box: LayoutBox, parent_clip: LayoutRect) -> None:
    box.clip = _intersect(parent_clip, box.rect)
    for child in box.children:
        _update_clips(child, box.clip)


def _layout_component(
    context: _LayoutContext,
    component: object,
    x: int,
    y: int,
    width: int,
    height: int | None,
    clip: LayoutRect,
) -> LayoutBox:
    safe_width = max(1, math.floor(width))
    node = get_layout_node(component)
    if node is None:
        lines = _render_cached(context, component, safe_width)
        allocated_height = len(lines) if height is None else max(0, math.floor(height))
        line_offset = 0
        if len(lines) > allocated_height and allocated_height > 0:
            cursor_line = next((i for i, line in enumerate(lines) if CURSOR_MARKER in line), -1)
            if cursor_line >= allocated_height:
                line_offset = cursor_line - allocated_height + 1
        rect = LayoutRect(x=x, y=y, width=safe_width, height=allocated_height)
        return LayoutBox(
            component=component,
            rect=rect,
            clip=_intersect(clip, rect),
            children=[],
            lines=lines,
            line_offset=line_offset,
            layer=0,
        )

    if node.type == "scroll":
        previous_scroll_top = node.state.scroll_top
        content_width = node.state.get_content_width(safe_width)
        child_box = _layout_component(
            context,
            node.component,
            x,
            y - previous_scroll_top,
            content_width,
            None,
            clip,
        )
        content_height = child_box.rect.height
        viewport_height = content_height if height is None else max(0, math.floor(height))
        node.state.update_layout(content_height, viewport_height, context.request_render)
        _translate_box(child_box, previous_scroll_top - node.state.scroll_top)
        scroll_view = node.state
        if node.state.primary or context.primary_scroll_view is None:
            context.primary_scroll_view = scroll_view
        rect = LayoutRect(x=x, y=y, width=safe_width, height=viewport_height)
        child_clip = _intersect(clip, rect)
        box = LayoutBox(
            component=component,
            rect=rect,
            clip=child_clip,
            children=[child_box],
            scroll_view=scroll_view,
            scroll_content_lines=_render_cached(context, node.component, content_width),
            layer=0,
        )
        child_box.parent = box
        _update_clips(child_box, child_clip)
        return box

    entries = visible_stack_entries(list(node.entries), context.viewport)
    gap_total = max(0, len(entries) - 1) * node.gap
    if node.type == "vstack":
        intrinsic_heights = [
            entry.basis
            if isinstance(entry.basis, int)
            else _measure_height(context, entry.component, safe_width)
            for entry in entries
        ]
        sizes = allocate_stack_sizes(entries, intrinsic_heights, height, node.gap)
        natural_height = sum(sizes) + gap_total
        allocated_height = natural_height if height is None else max(0, math.floor(height))
        rect = LayoutRect(x=x, y=y, width=safe_width, height=allocated_height)
        box = LayoutBox(component=component, rect=rect, clip=_intersect(clip, rect), children=[], layer=0)
        child_y = y
        for index, entry in enumerate(entries):
            box.children.append(
                _with_parent(
                    _layout_component(
                        context,
                        entry.component,
                        x,
                        child_y,
                        safe_width,
                        sizes[index],
                        box.clip,
                    ),
                    box,
                )
            )
            child_y += sizes[index] + node.gap
        return box

    intrinsic_widths = [
        entry.basis if isinstance(entry.basis, int) else _measure_width(context, entry.component, safe_width)
        for entry in entries
    ]
    widths = allocate_stack_sizes(entries, intrinsic_widths, safe_width, node.gap)
    intrinsic_heights = [
        _measure_height(context, entry.component, max(1, widths[index]))
        for index, entry in enumerate(entries)
    ]
    allocated_height = (
        max(intrinsic_heights, default=0) if height is None else max(0, height)
    )
    rect = LayoutRect(x=x, y=y, width=safe_width, height=allocated_height)
    box = LayoutBox(component=component, rect=rect, clip=_intersect(clip, rect), children=[], layer=0)
    child_x = x
    for index, entry in enumerate(entries):
        natural_child_height = intrinsic_heights[index]
        child_height = allocated_height if node.align == "stretch" else min(allocated_height, natural_child_height)
        child_y = y
        if node.align == "center":
            child_y += (allocated_height - child_height) // 2
        elif node.align == "end":
            child_y += allocated_height - child_height
        child_width = widths[index]
        if child_width == 0:
            box.children.append(
                LayoutBox(
                    component=entry.component,
                    rect=LayoutRect(x=child_x, y=child_y, width=0, height=child_height),
                    clip=LayoutRect(x=child_x, y=child_y, width=0, height=0),
                    children=[],
                    parent=box,
                    layer=0,
                )
            )
        else:
            box.children.append(
                _with_parent(
                    _layout_component(
                        context,
                        entry.component,
                        child_x,
                        child_y,
                        child_width,
                        child_height,
                        box.clip,
                    ),
                    box,
                )
            )
        child_x += child_width + node.gap
    return box


def _style_scrollbar_cell(line: str, column: int, total_width: int, style: Callable[[str], str]) -> str:
    if is_image_line(line):
        return line

    grapheme_range = get_grapheme_cell_range(line, column)
    start = grapheme_range[0] if grapheme_range else column
    end = grapheme_range[1] if grapheme_range else column + 1
    before = slice_by_column(line, 0, start, True)
    target = slice_by_column(line, start, end - start, True)
    after = slice_by_column(line, end, max(0, total_width - end), True)

    target_prefix = ""
    target_index = 0
    while target_index < len(target):
        ansi = extract_ansi_code(target, target_index)
        if not ansi:
            break
        target_prefix += ansi.code
        target_index += ansi.length
    target_text = target[target_index:] or (" " * (end - start))
    before_padding = " " * max(0, start - visible_width(before))
    return f"{before}{before_padding}{target_prefix}{style(target_text)}{after}"


def get_scrollbar_geometry(box: LayoutBox) -> ScrollbarGeometry | None:
    scroll_view = box.scroll_view
    if scroll_view is None or not getattr(scroll_view, "is_scrollbar_visible", False):
        return None
    if box.rect.width <= 0 or box.rect.height <= 0:
        return None

    content_height = box.children[0].rect.height if box.children else (len(box.scroll_content_lines or []))
    track_height = box.rect.height
    min_thumb_height = min(2, track_height)
    thumb_height = max(
        min_thumb_height,
        min(track_height, round((track_height * track_height) / content_height) if content_height else track_height),
    )
    max_scroll_top = max(0, content_height - track_height)
    max_thumb_top = track_height - thumb_height
    thumb_offset = 0 if max_scroll_top == 0 else round((scroll_view.scroll_top / max_scroll_top) * max_thumb_top)
    column = box.rect.x + box.rect.width - 1
    if column < box.clip.x or column >= box.clip.x + box.clip.width:
        return None

    return ScrollbarGeometry(
        column=column,
        track_top=box.rect.y,
        track_height=track_height,
        thumb_top=box.rect.y + thumb_offset,
        thumb_height=thumb_height,
        max_scroll_top=max_scroll_top,
    )


def _paint_scrollbar(box: LayoutBox, screen: list[str], total_width: int) -> None:
    geometry = get_scrollbar_geometry(box)
    if geometry is None or box.scroll_view is None:
        return
    style = box.scroll_view.scrollbar_style
    for offset in range(geometry.thumb_height):
        row = geometry.thumb_top + offset
        if row < box.clip.y or row >= box.clip.y + box.clip.height or row < 0 or row >= len(screen):
            continue
        screen[row] = _style_scrollbar_cell(screen[row] if row < len(screen) else "", geometry.column, total_width, style)


def _paint_box(box: LayoutBox, screen: list[str], total_width: int) -> None:
    if box.lines is not None:
        offset = box.line_offset or 0
        first_row = max(box.rect.y, box.clip.y, 0)
        last_row = min(box.rect.y + box.rect.height, box.clip.y + box.clip.height, len(screen))
        for row in range(first_row, last_row):
            source_index = offset + row - box.rect.y
            if source_index < 0 or source_index >= len(box.lines):
                continue
            source_line = box.lines[source_index]
            line = _OSC133_ZONE_PREFIX.sub("", source_line)
            image_metadata = get_kitty_image_metadata(line)
            if image_metadata:
                clip_bottom = min(len(screen), box.clip.y + box.clip.height)
                visible_rows = min(image_metadata.rows, clip_bottom - row)
                if visible_rows < image_metadata.rows:
                    line = crop_kitty_image_line(line, 0, visible_rows)
            if box.rect.x == 0 and box.rect.width >= total_width and (is_image_line(line) or not screen[row]):
                screen[row] = line
            else:
                screen[row] = composite_tui_line(screen[row] or "", line, box.rect.x, box.rect.width, total_width)

    for child in box.children:
        _paint_box(child, screen, total_width)

    if (
        box.scroll_view is not None
        and box.scroll_content_lines is not None
        and box.scroll_view.scroll_top > 0
        and box.rect.height > 0
    ):
        for image_row in range(box.scroll_view.scroll_top - 1, -1, -1):
            image_line = box.scroll_content_lines[image_row] if image_row < len(box.scroll_content_lines) else ""
            metadata = get_kitty_image_metadata(image_line)
            if metadata:
                hidden_rows = box.scroll_view.scroll_top - image_row
                if hidden_rows < metadata.rows:
                    visible_rows = min(box.rect.height, metadata.rows - hidden_rows)
                    cropped = crop_kitty_image_line(image_line, hidden_rows, visible_rows)
                    if box.rect.x == 0 and box.rect.width >= total_width:
                        screen[box.rect.y] = cropped
                break
            if image_line != "":
                break

    _paint_scrollbar(box, screen, total_width)


def render_layout_frame(
    root: object,
    width: int,
    height: int,
    request_render: Callable[[], None],
) -> LayoutFrame:
    safe_width = max(1, math.floor(width))
    safe_height = max(1, math.floor(height))
    context = _LayoutContext(
        viewport={"width": safe_width, "height": safe_height},
        render_cache={},
        request_render=request_render,
        primary_scroll_view=None,
    )
    root_box = _layout_component(
        context,
        root,
        0,
        0,
        safe_width,
        safe_height,
        LayoutRect(x=0, y=0, width=safe_width, height=safe_height),
    )
    lines = [""] * safe_height
    _paint_box(root_box, lines, safe_width)
    return LayoutFrame(
        root=root_box,
        width=safe_width,
        height=safe_height,
        lines=lines,
        primary_scroll_view=context.primary_scroll_view,
    )


def _contains_point(rect: LayoutRect, x: int, y: int) -> bool:
    return rect.x <= x < rect.x + rect.width and rect.y <= y < rect.y + rect.height


def get_scroll_view_box(frame: LayoutFrame, scroll_view: object) -> LayoutBox | None:
    def visit(box: LayoutBox) -> LayoutBox | None:
        if box.scroll_view is scroll_view:
            return box
        for child in box.children:
            match = visit(child)
            if match is not None:
                return match
        return None

    return visit(frame.root)


def get_scroll_views_at(frame: LayoutFrame, x: int, y: int) -> list:
    result: list[tuple[object, int]] = []

    def visit(box: LayoutBox, depth: int) -> None:
        if not _contains_point(box.clip, x, y):
            return
        if box.scroll_view is not None and _contains_point(box.rect, x, y):
            result.append((box.scroll_view, depth))
        for child in box.children:
            visit(child, depth + 1)

    visit(frame.root, 0)
    result.sort(key=lambda entry: entry[1], reverse=True)
    return [entry[0] for entry in result]
