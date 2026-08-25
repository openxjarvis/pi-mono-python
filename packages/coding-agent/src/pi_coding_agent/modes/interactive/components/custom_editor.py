"""Custom editor — mirrors components/custom-editor.ts"""
from __future__ import annotations

from typing import Any, Callable

from pi_tui.components.editor import Editor, EditorOptions, EditorTheme
from pi_tui.tui import TUI

from pi_coding_agent.core.keybindings import KeybindingsManager


class CustomEditor(Editor):
    """Editor that intercepts app-level keybindings before editor handling."""

    def __init__(
        self,
        tui: TUI,
        theme: EditorTheme,
        keybindings: KeybindingsManager,
        options: EditorOptions | dict[str, Any] | None = None,
    ) -> None:
        if isinstance(options, dict):
            options = EditorOptions(
                padding_x=int(options.get("padding_x") or options.get("paddingX") or 0),
                autocomplete_max_visible=int(
                    options.get("autocomplete_max_visible") or options.get("autocompleteMaxVisible") or 5
                ),
            )
        super().__init__(tui, theme, options)
        self.keybindings = keybindings
        self.action_handlers: dict[str, Callable[[], None]] = {}
        self.on_escape: Callable[[], None] | None = None
        self.on_ctrl_d: Callable[[], None] | None = None
        self.on_paste_image: Callable[[], None] | None = None
        self.on_extension_shortcut: Callable[[str], bool] | None = None

    def on_action(self, action: str, handler: Callable[[], None]) -> None:
        self.action_handlers[action] = handler

    def is_showing_autocomplete(self) -> bool:
        getter = getattr(self, "_is_showing_autocomplete", None) or getattr(self, "is_showing_autocomplete", None)
        if callable(getter) and getter is not CustomEditor.is_showing_autocomplete:
            return bool(getter())
        return bool(getattr(self, "_autocomplete_visible", False) or getattr(self, "_showing_autocomplete", False))

    def handle_input(self, data: str) -> None:
        if self.on_extension_shortcut and self.on_extension_shortcut(data):
            return
        if self.keybindings.matches("app.clipboard.pasteImage", data) or self.keybindings.matches("pasteImage", data):
            if self.on_paste_image:
                self.on_paste_image()
                return
        if self.keybindings.matches("app.interrupt", data) or self.keybindings.matches("interrupt", data):
            if not self.is_showing_autocomplete():
                handler = self.on_escape or self.action_handlers.get("app.interrupt") or self.action_handlers.get("interrupt")
                if handler:
                    handler()
                    return
            super().handle_input(data)
            return
        if self.keybindings.matches("app.exit", data) or self.keybindings.matches("exit", data):
            if len(self.get_text()) == 0:
                handler = self.on_ctrl_d or self.action_handlers.get("app.exit") or self.action_handlers.get("exit")
                if handler:
                    handler()
                    return
        if self.keybindings.matches("tui.editor.historyPrevious", data) or self.keybindings.matches(
            "tui.editor.historyNext", data
        ):
            super().handle_input(data)
            return
        for action, handler in self.action_handlers.items():
            if self.keybindings.matches(action, data):
                handler()
                return
        super().handle_input(data)


CustomEditorComponent = CustomEditor
