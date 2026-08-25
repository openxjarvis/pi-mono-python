"""Session tree selector. Mirrors tree-selector.ts."""
from __future__ import annotations

from typing import Any, Callable, Literal

from .component import Component

FilterMode = Literal["default", "no-tools", "user-only", "labeled-only", "all"]
FILTER_CYCLE: tuple[FilterMode, ...] = ("default", "no-tools", "user-only", "labeled-only", "all")
TREE_GUTTER_WIDTH = 2
MIN_VISIBLE_ANCHOR_CONTENT_WIDTH = 4
MAX_VISIBLE_ANCHOR_CONTENT_WIDTH = 20
MIN_ANCHOR_CONTEXT_WIDTH = 2
MAX_ANCHOR_CONTEXT_WIDTH = 12


def _node_entry(node: Any) -> Any:
    return getattr(node, "entry", node)


def _entry_field(entry: Any, snake: str, camel: str, default: Any = None) -> Any:
    if isinstance(entry, dict):
        return entry.get(snake, entry.get(camel, default))
    return getattr(entry, snake, None) or getattr(entry, camel, default)


def entry_id(node: Any) -> str:
    entry = _node_entry(node)
    return str(_entry_field(entry, "id", "id", "") or getattr(node, "id", "") or "")


def entry_parent_id(node: Any) -> str | None:
    entry = _node_entry(node)
    value = _entry_field(entry, "parent_id", "parentId")
    return None if value in (None, "") else str(value)


def entry_type(node: Any) -> str:
    entry = _node_entry(node)
    return str(_entry_field(entry, "type", "type", "") or "")


def entry_message(node: Any) -> Any:
    entry = _node_entry(node)
    if isinstance(entry, dict):
        return entry.get("message") or (entry.get("data") or {}).get("message")
    data = getattr(entry, "data", None)
    if isinstance(data, dict) and data.get("message") is not None:
        return data.get("message")
    return getattr(entry, "message", None)


def node_label(node: Any) -> str | None:
    label = getattr(node, "label", None)
    if label:
        return str(label)
    entry = _node_entry(node)
    if isinstance(entry, dict):
        return entry.get("label")
    return getattr(entry, "label", None)


def node_children(node: Any) -> list[Any]:
    kids = getattr(node, "children", None)
    if kids is None and isinstance(node, dict):
        kids = node.get("children")
    return list(kids or [])


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "")
    return str(getattr(message, "role", "") or "")


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _has_text_content(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and str(block.get("text") or "").strip():
            return True
        if getattr(block, "type", None) == "text" and str(getattr(block, "text", "") or "").strip():
            return True
    return False


def _searchable_text(node: Any) -> str:
    parts = [entry_id(node), node_label(node) or "", entry_type(node)]
    message = entry_message(node)
    if message is not None:
        parts.append(_message_role(message))
        content = _message_content(message)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    parts.append(str(block.get("text") or block.get("thinking") or block.get("name") or ""))
                else:
                    parts.append(str(getattr(block, "text", "") or getattr(block, "name", "")))
    return " ".join(part for part in parts if part)


def render_horizontal_viewport(rows: list[dict[str, Any]], width: int) -> list[str]:
    viewport_width = max(0, width - TREE_GUTTER_WIDTH)
    max_body_width = max((row["body_width"] for row in rows), default=0)
    max_scroll = max(0, max_body_width - viewport_width)
    selected = next((row for row in rows if row["is_selected"]), None)
    horizontal_scroll = 0
    if selected and max_scroll > 0:
        min_visible = min(
            MAX_VISIBLE_ANCHOR_CONTENT_WIDTH,
            max(MIN_VISIBLE_ANCHOR_CONTENT_WIDTH, viewport_width // 3),
        )
        if selected["anchor_col"] > viewport_width - min_visible:
            context = min(MAX_ANCHOR_CONTEXT_WIDTH, max(MIN_ANCHOR_CONTEXT_WIDTH, viewport_width // 4))
            horizontal_scroll = min(max_scroll, selected["anchor_col"] - context)
    lines: list[str] = []
    for row in rows:
        body = row["body"]
        if horizontal_scroll > 0:
            body = body[horizontal_scroll : horizontal_scroll + viewport_width]
        else:
            body = body[:viewport_width]
        line = f"{row['gutter']}{body}"
        lines.append(line[:width])
    return lines


class TreeSelectorComponent(Component):
    name = "tree_selector"

    def __init__(
        self,
        tree: list[Any] | None = None,
        current_leaf_id: str | None = None,
        initial_selected_id: str | None = None,
        initial_filter_mode: FilterMode = "default",
        on_select: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        on_copy: Callable[[str | None], None] | None = None,
        on_label_edit: Callable[[str, str | None], None] | None = None,
        items: list[dict[str, Any]] | None = None,
        selected_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tree = list(tree or items or [])
        self.current_leaf_id = current_leaf_id
        self.filter_mode: FilterMode = initial_filter_mode  # type: ignore[assignment]
        self.search_query = ""
        self.on_select = on_select
        self.on_cancel = on_cancel
        self.on_copy = on_copy
        self.on_label_edit = on_label_edit
        self.folded: set[str] = set()
        self.show_label_timestamps = False
        self.multiple_roots = len(self.tree) > 1
        self.tool_call_map: dict[str, dict[str, Any]] = {}
        self.flat = self._flatten_tree(self.tree)
        self.active_path_ids = self._build_active_path()
        self.filtered: list[dict[str, Any]] = []
        self.last_selected_id = selected_id or initial_selected_id or current_leaf_id
        self.selected_index = 0
        self._apply_filter()
        self.selected_index = self._find_nearest_visible_index(self.last_selected_id)

    def set_filter_mode(self, mode: FilterMode) -> None:
        self.filter_mode = mode
        self._apply_filter()
        self.invalidate()

    def cycle_filter(self, delta: int = 1) -> None:
        index = FILTER_CYCLE.index(self.filter_mode) if self.filter_mode in FILTER_CYCLE else 0
        self.set_filter_mode(FILTER_CYCLE[(index + delta) % len(FILTER_CYCLE)])

    def set_search(self, query: str) -> None:
        self.search_query = query
        self._apply_filter()
        self.invalidate()

    def toggle_fold(self) -> None:
        node = self.get_selected_node()
        if node is None or not self._is_foldable(node):
            return
        identifier = entry_id(node)
        if identifier in self.folded:
            self.folded.discard(identifier)
        else:
            self.folded.add(identifier)
        self._apply_filter()
        self.invalidate()

    def toggle_label_timestamps(self) -> None:
        self.show_label_timestamps = not self.show_label_timestamps
        self.invalidate()

    def move(self, delta: int) -> None:
        if not self.filtered:
            return
        self.selected_index = max(0, min(len(self.filtered) - 1, self.selected_index + delta))
        self.last_selected_id = entry_id(self.filtered[self.selected_index]["node"])
        self.invalidate()

    def page(self, delta: int, page_size: int = 8) -> None:
        self.move(delta * page_size)

    def select_current(self) -> str | None:
        if not self.filtered:
            if self.on_cancel:
                self.on_cancel()
            return None
        identifier = entry_id(self.filtered[self.selected_index]["node"])
        if self.on_select:
            self.on_select(identifier)
        return identifier

    def cancel(self) -> None:
        if self.on_cancel:
            self.on_cancel()

    def get_selected_node(self) -> Any | None:
        if not self.filtered:
            return None
        return self.filtered[self.selected_index]["node"]

    def get_search_query(self) -> str:
        return self.search_query

    def copy_selected(self) -> str | None:
        node = self.get_selected_node()
        if node is None:
            return None
        text = _searchable_text(node).strip() or None
        if self.on_copy:
            self.on_copy(text)
        return text

    def edit_selected_label(self) -> None:
        node = self.get_selected_node()
        if node is None or not self.on_label_edit:
            return
        self.on_label_edit(entry_id(node), node_label(node))

    def handle_input(self, action: str) -> bool:
        handlers = {
            "tui.select.up": lambda: self.move(-1),
            "tui.select.down": lambda: self.move(1),
            "tui.select.pageUp": lambda: self.page(-1),
            "tui.select.pageDown": lambda: self.page(1),
            "tui.select.confirm": self.select_current,
            "tui.select.cancel": self.cancel,
            "app.tree.foldOrUp": lambda: self.toggle_fold() if self._is_foldable(self.get_selected_node()) else self.move(-1),
            "app.tree.unfoldOrDown": lambda: self.toggle_fold() if self.get_selected_node() and entry_id(self.get_selected_node()) in self.folded else self.move(1),
            "app.tree.editLabel": self.edit_selected_label,
            "app.tree.toggleLabelTimestamp": self.toggle_label_timestamps,
            "app.tree.filter.cycleForward": lambda: self.cycle_filter(1),
            "app.tree.filter.cycleBack": lambda: self.cycle_filter(-1),
            "app.tree.filter.default": lambda: self.set_filter_mode("default"),
            "app.tree.filter.noTools": lambda: self.set_filter_mode("no-tools"),
            "app.tree.filter.userOnly": lambda: self.set_filter_mode("user-only"),
            "app.tree.filter.labeledOnly": lambda: self.set_filter_mode("labeled-only"),
            "app.tree.filter.all": lambda: self.set_filter_mode("all"),
            "app.message.copy": self.copy_selected,
        }
        handler = handlers.get(action)
        if handler is None:
            return False
        handler()
        return True

    def _flatten_tree(self, roots: list[Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        self.tool_call_map.clear()
        contains_active: dict[int, bool] = {}
        all_nodes: list[Any] = []
        stack = list(reversed(roots))
        while stack:
            node = stack.pop()
            all_nodes.append(node)
            stack.extend(reversed(node_children(node)))
        for node in reversed(all_nodes):
            has = self.current_leaf_id is not None and entry_id(node) == self.current_leaf_id
            for child in node_children(node):
                if contains_active.get(id(child)):
                    has = True
            contains_active[id(node)] = has

        multiple_roots = len(roots) > 1
        ordered_roots = sorted(roots, key=lambda node: int(not contains_active.get(id(node), False)))
        walk: list[tuple[Any, int, bool, bool, bool, list[dict[str, Any]], bool]] = []
        for index, root in enumerate(reversed(ordered_roots)):
            is_last = index == 0
            walk.append((root, 1 if multiple_roots else 0, multiple_roots, multiple_roots, is_last, [], multiple_roots))

        while walk:
            node, indent, just_branched, show_connector, is_last, gutters, is_virtual_root_child = walk.pop()
            message = entry_message(node)
            if entry_type(node) == "message" and _message_role(message) == "assistant":
                content = _message_content(message)
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "toolCall":
                            self.tool_call_map[str(block.get("id"))] = {
                                "name": block.get("name"),
                                "arguments": block.get("arguments") or {},
                            }
            result.append(
                {
                    "node": node,
                    "indent": indent,
                    "show_connector": show_connector,
                    "is_last": is_last,
                    "gutters": gutters,
                    "is_virtual_root_child": is_virtual_root_child,
                }
            )
            children = node_children(node)
            multiple_children = len(children) > 1
            prioritized = [child for child in children if contains_active.get(id(child))]
            rest = [child for child in children if not contains_active.get(id(child))]
            ordered_children = [*prioritized, *rest]
            if multiple_children:
                child_indent = indent + 1
            elif just_branched and indent > 0:
                child_indent = indent + 1
            else:
                child_indent = indent
            connector_displayed = show_connector and not is_virtual_root_child
            current_display_indent = max(0, indent - 1) if self.multiple_roots else indent
            connector_position = max(0, current_display_indent - 1)
            child_gutters = [*gutters, {"position": connector_position, "show": not is_last}] if connector_displayed else gutters
            for child_index, child in enumerate(reversed(ordered_children)):
                child_is_last = child_index == 0
                walk.append((child, child_indent, multiple_children, multiple_children, child_is_last, child_gutters, False))
        return result

    def _build_active_path(self) -> set[str]:
        active: set[str] = set()
        if not self.current_leaf_id:
            return active
        entry_map = {entry_id(item["node"]): item for item in self.flat}
        current = self.current_leaf_id
        while current:
            active.add(current)
            node = entry_map.get(current)
            if node is None:
                break
            current = entry_parent_id(node["node"])
        return active

    def _find_nearest_visible_index(self, identifier: str | None) -> int:
        if not self.filtered:
            return 0
        entry_map = {entry_id(item["node"]): item for item in self.flat}
        visible = {entry_id(item["node"]): index for index, item in enumerate(self.filtered)}
        current = identifier
        while current:
            if current in visible:
                return visible[current]
            node = entry_map.get(current)
            if node is None:
                break
            current = entry_parent_id(node["node"])
        return max(0, len(self.filtered) - 1)

    def _is_foldable(self, node: Any | None) -> bool:
        return bool(node is not None and node_children(node))

    def _passes_filter(self, item: dict[str, Any]) -> bool:
        node = item["node"]
        identifier = entry_id(node)
        kind = entry_type(node)
        message = entry_message(node)
        is_current = identifier == self.current_leaf_id
        if kind == "message" and _message_role(message) == "assistant" and not is_current:
            stop = ""
            if isinstance(message, dict):
                stop = str(message.get("stopReason") or message.get("stop_reason") or "")
            else:
                stop = str(getattr(message, "stop_reason", "") or getattr(message, "stopReason", "") or "")
            if not _has_text_content(_message_content(message)) and stop in {"", "stop", "toolUse"}:
                return False
        settings_entry = kind in {"label", "custom", "model_change", "thinking_level_change", "session_info"}
        if self.filter_mode == "user-only":
            passes = kind == "message" and _message_role(message) == "user"
        elif self.filter_mode == "no-tools":
            passes = not settings_entry and not (kind == "message" and _message_role(message) == "toolResult")
        elif self.filter_mode == "labeled-only":
            passes = node_label(node) is not None
        elif self.filter_mode == "all":
            passes = True
        else:
            passes = not settings_entry
        if not passes:
            return False
        tokens = [token for token in self.search_query.lower().split() if token]
        if tokens:
            text = _searchable_text(node).lower()
            return all(token in text for token in tokens)
        return True

    def _apply_filter(self) -> None:
        if self.filtered:
            self.last_selected_id = entry_id(self.filtered[self.selected_index]["node"])
        self.filtered = [item for item in self.flat if self._passes_filter(item)]
        if self.folded:
            skip: set[str] = set()
            for item in self.flat:
                identifier = entry_id(item["node"])
                parent = entry_parent_id(item["node"])
                if parent is not None and (parent in self.folded or parent in skip):
                    skip.add(identifier)
            self.filtered = [item for item in self.filtered if entry_id(item["node"]) not in skip]
        self._recalculate_visual_structure()
        if self.last_selected_id:
            self.selected_index = self._find_nearest_visible_index(self.last_selected_id)
        elif self.selected_index >= len(self.filtered):
            self.selected_index = max(0, len(self.filtered) - 1)
        if self.filtered:
            self.last_selected_id = entry_id(self.filtered[self.selected_index]["node"])

    def _recalculate_visual_structure(self) -> None:
        if not self.filtered:
            return
        visible_ids = {entry_id(item["node"]) for item in self.filtered}
        entry_map = {entry_id(item["node"]): item for item in self.flat}

        def visible_ancestor(identifier: str) -> str | None:
            current = entry_parent_id(entry_map[identifier]["node"]) if identifier in entry_map else None
            while current:
                if current in visible_ids:
                    return current
                current = entry_parent_id(entry_map[current]["node"]) if current in entry_map else None
            return None

        children_map: dict[str | None, list[str]] = {}
        for item in self.filtered:
            identifier = entry_id(item["node"])
            parent = visible_ancestor(identifier)
            children_map.setdefault(parent, []).append(identifier)
            item["visible_parent"] = parent
        for item in self.filtered:
            identifier = entry_id(item["node"])
            parent = item.get("visible_parent")
            siblings = children_map.get(parent) or [identifier]
            item["show_connector"] = len(siblings) > 1
            item["is_last"] = siblings[-1] == identifier
            depth = 0
            current = parent
            while current:
                depth += 1
                current = next((row.get("visible_parent") for row in self.filtered if entry_id(row["node"]) == current), None)
            item["indent"] = depth

    def _row_label(self, node: Any) -> str:
        label = node_label(node)
        if label:
            suffix = ""
            if self.show_label_timestamps:
                entry = _node_entry(node)
                timestamp = _entry_field(entry, "timestamp", "timestamp")
                if timestamp:
                    suffix = f"  {timestamp}"
            return f"{label}{suffix}"
        message = entry_message(node)
        role = _message_role(message) or entry_type(node) or "entry"
        text = ""
        content = _message_content(message)
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = str(block.get("text") or "")
                    break
        preview = " ".join(text.split())
        if len(preview) > 60:
            preview = f"{preview[:57]}..."
        return f"{role}: {preview}" if preview else role

    def _render_body(self, width: int) -> str:
        lines = [f"Session tree  filter:{self.filter_mode}"]
        if self.search_query:
            lines.append(f"  search: {self.search_query}")
        if not self.filtered:
            lines.append("  (empty)")
            return "\n".join(lines)
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(self.filtered):
            node = item["node"]
            identifier = entry_id(node)
            depth = int(item.get("indent") or 0)
            connector = ""
            if item.get("show_connector"):
                connector = "└─ " if item.get("is_last") else "├─ "
            fold = "-" if identifier in self.folded else ("+" if self._is_foldable(node) else " ")
            leaf = "*" if identifier == self.current_leaf_id else " "
            marker = ">" if index == self.selected_index else " "
            indent = "   " * max(0, depth - (1 if connector else 0))
            body = f"{indent}{connector}{fold}{leaf}{self._row_label(node)}"
            gutter = f"{marker} "
            rows.append(
                {
                    "gutter": gutter,
                    "body": body,
                    "body_width": len(body),
                    "anchor_col": len(indent) + len(connector),
                    "is_selected": index == self.selected_index,
                }
            )
        lines.extend(render_horizontal_viewport(rows, max(width, 20)))
        return "\n".join(lines)
