"""Stack layout helpers — mirrors components/stack.ts"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from ..layout_node import StackLayoutNode
from ..tui import Container

Align = Literal["stretch", "start", "center", "end"]


@dataclass
class StackEntryOptions:
    basis: int | Literal["auto"] | None = None
    grow: int | None = None
    shrink: int | None = None
    min_size: int | None = None
    max_size: int | None = None
    visible: Callable[[dict], bool] | None = None


@dataclass
class StackEntry(StackEntryOptions):
    component: object = None


StackChild = object  # Component | StackEntry


def _normalize_size(value: int | None, fallback: int) -> int:
    if value is None:
        return fallback
    return max(0, int(value))


def _is_stack_entry(child: object) -> bool:
    return isinstance(child, StackEntry) or (
        hasattr(child, "component") and not hasattr(child, "render")
    )


class Stack(Container):
    layout_type: Literal["vstack", "hstack"] = "vstack"

    def __init__(self, children: list | None = None, options: dict | None = None) -> None:
        super().__init__()
        opts = options or {}
        self.gap = _normalize_size(opts.get("gap"), 0)
        self.align: Align = opts.get("align", "stretch")
        self.entries: list[StackEntry] = []
        for child in children or []:
            if _is_stack_entry(child):
                self.add_child(child.component, child)
            else:
                self.add_child(child)

    def layout_node(self) -> StackLayoutNode:
        return StackLayoutNode(
            type=self.layout_type,
            entries=self.entries,
            gap=self.gap,
            align=self.align,
        )

    def add_child(self, component: object, options: StackEntryOptions | None = None) -> None:  # type: ignore[override]
        super().add_child(component)
        opts = options or StackEntryOptions()
        self.entries.append(
            StackEntry(
                component=component,
                basis=opts.basis,
                grow=_normalize_size(opts.grow, 0) if opts.grow is not None else None,
                shrink=_normalize_size(opts.shrink, 1) if opts.shrink is not None else None,
                min_size=_normalize_size(opts.min_size, 0) if opts.min_size is not None else None,
                max_size=_normalize_size(opts.max_size, 10**9) if opts.max_size is not None else None,
                visible=opts.visible,
            )
        )

    def remove_child(self, component: object) -> None:
        super().remove_child(component)
        self.entries = [e for e in self.entries if e.component is not component]

    def clear(self) -> None:
        super().clear()
        self.entries = []


def visible_stack_entries(entries: list[StackEntry], viewport: dict) -> list[StackEntry]:
    return [entry for entry in entries if entry.visible is None or entry.visible(viewport)]


def _clamp_size(size: int, entry: StackEntry) -> int:
    minimum = max(0, int(entry.min_size or 0))
    maximum = max(minimum, int(entry.max_size if entry.max_size is not None else 10**9))
    return max(minimum, min(maximum, max(0, int(size))))


def _distribute(sizes: list[int], entries: list[StackEntry], amount: int, mode: str) -> None:
    remaining = amount
    while remaining > 0:
        candidates = []
        for index, entry in enumerate(entries):
            if mode == "grow":
                if (entry.grow or 0) > 0 and sizes[index] < (entry.max_size if entry.max_size is not None else 10**9):
                    candidates.append((entry, index))
            elif (entry.shrink if entry.shrink is not None else 1) > 0 and sizes[index] > (entry.min_size or 0):
                candidates.append((entry, index))
        if not candidates:
            return

        total_weight = 0
        for entry, index in candidates:
            if mode == "grow":
                total_weight += entry.grow or 0
            else:
                total_weight += (entry.shrink if entry.shrink is not None else 1) * max(1, sizes[index])
        distributed = 0
        for entry, index in candidates:
            if remaining <= 0:
                break
            weight = (entry.grow or 0) if mode == "grow" else (entry.shrink if entry.shrink is not None else 1) * max(1, sizes[index])
            proposed = max(1, (remaining * weight) // total_weight) if total_weight else 0
            capacity = (
                (entry.max_size if entry.max_size is not None else 10**9) - sizes[index]
                if mode == "grow"
                else sizes[index] - (entry.min_size or 0)
            )
            delta = min(remaining, proposed, capacity)
            if delta <= 0:
                continue
            sizes[index] += delta if mode == "grow" else -delta
            remaining -= delta
            distributed += delta
        if distributed == 0:
            return


def allocate_stack_sizes(
    entries: list[StackEntry],
    intrinsic_sizes: list[int],
    available_size: int | None,
    gap: int,
) -> list[int]:
    sizes = [
        _clamp_size(
            intrinsic_sizes[index] if entry.basis is None or entry.basis == "auto" else int(entry.basis),
            entry,
        )
        for index, entry in enumerate(entries)
    ]
    if available_size is None:
        return sizes

    content_size = max(0, int(available_size) - max(0, len(entries) - 1) * gap)
    total = sum(sizes)
    if total < content_size:
        _distribute(sizes, entries, content_size - total, "grow")
    elif total > content_size:
        _distribute(sizes, entries, total - content_size, "shrink")
    return sizes
