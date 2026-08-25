"""Layout node protocol — mirrors packages/tui/src/layout-node.ts"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence

LayoutAlign = Literal["stretch", "start", "center", "end"]
StackLayoutType = Literal["vstack", "hstack"]
ScrollOverscroll = Literal["chain", "contain"]


@dataclass
class LayoutViewport:
    width: int
    height: int

    def __getitem__(self, key: str) -> int:
        if key == "width":
            return self.width
        if key == "height":
            return self.height
        raise KeyError(key)


@dataclass
class StackLayoutEntry:
    component: object
    basis: int | Literal["auto"] | None = None
    grow: int | None = None
    shrink: int | None = None
    min_size: int | None = None
    max_size: int | None = None
    visible: Callable[[LayoutViewport], bool] | None = None


@dataclass
class StackLayoutNode:
    type: StackLayoutType
    entries: Sequence[object]
    gap: int
    align: LayoutAlign


class ScrollLayoutState(Protocol):
    scroll_top: int
    primary: bool
    overscroll: ScrollOverscroll
    viewport_height: int

    def get_content_width(self, width: int) -> int: ...

    def update_layout(
        self,
        content_height: int,
        viewport_height: int,
        request_render: Callable[[], None],
    ) -> None: ...


@dataclass
class ScrollLayoutNode:
    type: Literal["scroll"]
    component: object
    state: ScrollLayoutState


LayoutNode = StackLayoutNode | ScrollLayoutNode


class LayoutComponent(Protocol):
    def layout_node(self) -> LayoutNode: ...


def get_layout_node(component: object) -> LayoutNode | None:
    candidate = getattr(component, "layout_node", None)
    if callable(candidate):
        node = candidate()
        if isinstance(node, (StackLayoutNode, ScrollLayoutNode)):
            return node
    return None
