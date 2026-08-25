"""Session selector — mirrors session-selector.ts"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable, Literal

from .component import Component
from .session_selector_search import (
    NameFilter,
    SortMode,
    filter_and_sort_sessions,
    has_session_name,
)

SessionScope = Literal["current", "all"]


def _shorten_path(path: str) -> str:
    home = os.path.expanduser("~")
    if path.startswith(home):
        return f"~{path[len(home):]}"
    return path


def _format_session_date(value: Any) -> str:
    if isinstance(value, datetime):
        date = value
    else:
        try:
            date = datetime.fromtimestamp(float(value))
        except Exception:
            return ""
    diff_mins = int((datetime.now() - date).total_seconds() // 60)
    if diff_mins < 1:
        return "now"
    if diff_mins < 60:
        return f"{diff_mins}m"
    if diff_mins < 1440:
        return f"{diff_mins // 60}h"
    if diff_mins < 10080:
        return f"{diff_mins // 1440}d"
    return f"{diff_mins // 10080}w"


class SessionSelectorComponent(Component):
    name = "session_selector"

    def __init__(
        self,
        sessions: list[Any] | None = None,
        on_select: Callable[[Any], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        cwd: str | None = None,
        scope: SessionScope = "current",
        sort_mode: SortMode = "recent",
        name_filter: NameFilter = "all",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sessions = list(sessions or [])
        self.on_select = on_select
        self.on_cancel = on_cancel
        self.cwd = cwd
        self.scope = scope
        self.sort_mode = sort_mode
        self.name_filter = name_filter
        self.selected_index = 0
        self.query = ""
        self.loading = False
        self.show_path = True
        self.mode: Literal["list", "rename"] = "list"
        self.rename_value = ""
        self.confirming_delete: Any | None = None
        self.on_delete = kwargs.get("on_delete") or kwargs.get("onDeleteSession")
        self.on_rename = kwargs.get("on_rename") or kwargs.get("onRenameSession")

    def set_scope(self, scope: SessionScope) -> None:
        self.scope = scope
        self.selected_index = 0
        self.invalidate()

    def set_sort_mode(self, sort_mode: SortMode) -> None:
        self.sort_mode = sort_mode
        self.selected_index = 0
        self.invalidate()

    def set_name_filter(self, name_filter: NameFilter) -> None:
        self.name_filter = name_filter
        self.selected_index = 0
        self.invalidate()

    def filter(self, query: str) -> None:
        self.query = query
        self.selected_index = 0
        self.invalidate()

    def _scoped(self) -> list[Any]:
        if self.scope == "all" or not self.cwd:
            return self.sessions
        cwd = os.path.abspath(self.cwd)
        scoped = []
        for session in self.sessions:
            session_cwd = getattr(session, "cwd", None)
            if session_cwd and os.path.abspath(session_cwd) == cwd:
                scoped.append(session)
        return scoped or self.sessions

    def _visible(self) -> list[Any]:
        return filter_and_sort_sessions(self._scoped(), self.query, self.sort_mode, self.name_filter)

    def select_current(self) -> Any | None:
        visible = self._visible()
        if not visible:
            if self.on_cancel:
                self.on_cancel()
            return None
        session = visible[min(self.selected_index, len(visible) - 1)]
        if self.on_select:
            self.on_select(session)
        return session

    def move(self, delta: int) -> None:
        visible = self._visible()
        if not visible:
            return
        self.selected_index = max(0, min(len(visible) - 1, self.selected_index + delta))
        self.invalidate()

    def toggle_scope(self) -> None:
        self.set_scope("all" if self.scope == "current" else "current")

    def toggle_sort_mode(self) -> None:
        self.set_sort_mode("threaded" if self.sort_mode == "recent" else "recent")

    def toggle_name_filter(self) -> None:
        self.set_name_filter("named" if self.name_filter == "all" else "all")

    def toggle_path(self) -> None:
        self.show_path = not self.show_path
        self.invalidate()

    def get_session_list(self) -> list[Any]:
        return self._visible()

    def get_selected_session(self) -> Any | None:
        visible = self._visible()
        if not visible:
            return None
        return visible[min(self.selected_index, len(visible) - 1)]

    def get_selected_session_path(self) -> str | None:
        session = self.get_selected_session()
        if session is None:
            return None
        return getattr(session, "file_path", None) or getattr(session, "path", None)

    def enter_rename_mode(self) -> None:
        session = self.get_selected_session()
        if session is None:
            return
        self.mode = "rename"
        self.rename_value = str(getattr(session, "name", "") or "")
        self.invalidate()

    def exit_rename_mode(self, apply: bool = False) -> None:
        session = self.get_selected_session()
        if apply and session is not None and self.on_rename:
            self.on_rename(session, self.rename_value.strip() or None)
        self.mode = "list"
        self.rename_value = ""
        self.invalidate()

    def start_delete_confirmation(self) -> None:
        self.confirming_delete = self.get_selected_session()
        self.invalidate()

    def confirm_delete(self) -> None:
        session = self.confirming_delete
        self.confirming_delete = None
        if session is not None and self.on_delete:
            self.on_delete(session)
        self.invalidate()

    def cancel_delete(self) -> None:
        self.confirming_delete = None
        self.invalidate()

    def handle_input(self, action: str) -> bool:
        if self.mode == "rename":
            if action == "tui.select.confirm":
                self.exit_rename_mode(True)
                return True
            if action == "tui.select.cancel":
                self.exit_rename_mode(False)
                return True
            return False
        if self.confirming_delete is not None:
            if action == "tui.select.confirm":
                self.confirm_delete()
                return True
            if action == "tui.select.cancel":
                self.cancel_delete()
                return True
            return False
        handlers = {
            "tui.select.up": lambda: self.move(-1),
            "tui.select.down": lambda: self.move(1),
            "tui.select.pageUp": lambda: self.move(-8),
            "tui.select.pageDown": lambda: self.move(8),
            "tui.select.confirm": self.select_current,
            "tui.select.cancel": lambda: self.on_cancel() if self.on_cancel else None,
            "tui.input.tab": self.toggle_scope,
            "app.session.toggleSort": self.toggle_sort_mode,
            "app.session.toggleNamedFilter": self.toggle_name_filter,
            "app.session.togglePath": self.toggle_path,
            "app.session.rename": self.enter_rename_mode,
            "app.session.delete": self.start_delete_confirmation,
            "app.session.deleteNoninvasive": self.start_delete_confirmation,
        }
        handler = handlers.get(action)
        if handler is None:
            return False
        handler()
        return True

    def _tree_prefix(self, session: Any, index: int, visible: list[Any]) -> str:
        if self.sort_mode != "threaded":
            return ""
        parent = getattr(session, "parent_session_path", None) or getattr(session, "parent_session", None)
        if not parent:
            return ""
        last = index == len(visible) - 1 or not (
            getattr(visible[index + 1], "parent_session_path", None)
            or getattr(visible[index + 1], "parent_session", None)
        )
        return "└─ " if last else "├─ "

    def _render_body(self, width: int) -> str:
        visible = self._visible()
        header = f"Resume session  [{self.scope}] sort:{self.sort_mode} names:{self.name_filter}"
        lines = [header]
        if self.query:
            lines.append(f"  filter: {self.query}")
        if self.loading:
            lines.append("  loading...")
        if self.mode == "rename":
            lines.append(f"  rename: {self.rename_value}")
        if self.confirming_delete is not None:
            label = getattr(self.confirming_delete, "name", None) or getattr(self.confirming_delete, "session_id", "")
            lines.append(f"  delete {label}? confirm/cancel")
        if not visible:
            lines.append("  (no sessions)")
            return "\n".join(lines)
        for index, session in enumerate(visible):
            marker = ">" if index == self.selected_index else " "
            label = getattr(session, "name", None) or getattr(session, "session_id", None) or getattr(session, "id", session)
            named = "*" if has_session_name(session) else " "
            when = _format_session_date(getattr(session, "modified", None) or getattr(session, "mtime", None))
            path = _shorten_path(str(getattr(session, "cwd", "") or getattr(session, "file_path", "") or ""))
            extra = f"  {when}" if when else ""
            prefix = self._tree_prefix(session, index, visible)
            lines.append(f"  {marker}{named} {prefix}{label}{extra}")
            if self.show_path and path and width > 40:
                lines.append(f"      {path}")
        return "\n".join(lines)
