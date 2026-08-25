"""
Global keybinding registry — mirrors packages/tui/src/keybindings.ts

TUI ids are namespaced (``tui.editor.cursorUp``). The older editor-only
``EditorKeybindingsManager`` remains as a compatibility wrapper that maps
legacy action names onto the shared registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .keys import KeyId, matches_key

Keybinding = str
KeybindingsConfig = dict[str, "KeyId | list[KeyId] | None"]


@dataclass
class KeybindingDefinition:
    default_keys: KeyId | list[KeyId]
    description: str = ""


KeybindingDefinitions = dict[str, KeybindingDefinition]


@dataclass
class KeybindingConflict:
    key: KeyId
    keybindings: list[str]


TUI_KEYBINDINGS: KeybindingDefinitions = {
    "tui.editor.cursorUp": KeybindingDefinition("up", "Move cursor up"),
    "tui.editor.cursorDown": KeybindingDefinition("down", "Move cursor down"),
    "tui.editor.historyPrevious": KeybindingDefinition([], "Select previous prompt history entry"),
    "tui.editor.historyNext": KeybindingDefinition([], "Select next prompt history entry"),
    "tui.editor.cursorLeft": KeybindingDefinition(["left", "ctrl+b"], "Move cursor left"),
    "tui.editor.cursorRight": KeybindingDefinition(["right", "ctrl+f"], "Move cursor right"),
    "tui.editor.cursorWordLeft": KeybindingDefinition(
        ["alt+left", "ctrl+left", "alt+b"], "Move cursor word left"
    ),
    "tui.editor.cursorWordRight": KeybindingDefinition(
        ["alt+right", "ctrl+right", "alt+f"], "Move cursor word right"
    ),
    "tui.editor.cursorLineStart": KeybindingDefinition(
        ["home", "ctrl+home", "ctrl+a"], "Move to line start"
    ),
    "tui.editor.cursorLineEnd": KeybindingDefinition(
        ["end", "ctrl+end", "ctrl+e"], "Move to line end"
    ),
    "tui.editor.jumpForward": KeybindingDefinition("ctrl+]", "Jump forward to character"),
    "tui.editor.jumpBackward": KeybindingDefinition("ctrl+alt+]", "Jump backward to character"),
    "tui.editor.pageUp": KeybindingDefinition(["pageUp", "ctrl+pageUp"], "Page up"),
    "tui.editor.pageDown": KeybindingDefinition(["pageDown", "ctrl+pageDown"], "Page down"),
    "tui.editor.deleteCharBackward": KeybindingDefinition("backspace", "Delete character backward"),
    "tui.editor.deleteCharForward": KeybindingDefinition(
        ["delete", "ctrl+d"], "Delete character forward"
    ),
    "tui.editor.deleteWordBackward": KeybindingDefinition(
        ["ctrl+w", "alt+backspace"], "Delete word backward"
    ),
    "tui.editor.deleteWordForward": KeybindingDefinition(
        ["alt+d", "alt+delete"], "Delete word forward"
    ),
    "tui.editor.deleteToLineStart": KeybindingDefinition("ctrl+u", "Delete to line start"),
    "tui.editor.deleteToLineEnd": KeybindingDefinition("ctrl+k", "Delete to line end"),
    "tui.editor.yank": KeybindingDefinition("ctrl+y", "Yank"),
    "tui.editor.yankPop": KeybindingDefinition("alt+y", "Yank pop"),
    "tui.editor.undo": KeybindingDefinition("ctrl+-", "Undo"),
    "tui.input.newLine": KeybindingDefinition(["shift+enter", "ctrl+j"], "Insert newline"),
    "tui.input.submit": KeybindingDefinition("enter", "Submit input"),
    "tui.input.tab": KeybindingDefinition("tab", "Tab / autocomplete"),
    "tui.input.copy": KeybindingDefinition("ctrl+c", "Copy selection"),
    "tui.select.up": KeybindingDefinition("up", "Move selection up"),
    "tui.select.down": KeybindingDefinition("down", "Move selection down"),
    "tui.select.pageUp": KeybindingDefinition("pageUp", "Selection page up"),
    "tui.select.pageDown": KeybindingDefinition("pageDown", "Selection page down"),
    "tui.select.confirm": KeybindingDefinition("enter", "Confirm selection"),
    "tui.select.cancel": KeybindingDefinition(["escape", "ctrl+c"], "Cancel selection"),
    "tui.altScreen.pageUp": KeybindingDefinition("pageUp", "Scroll viewport up one page"),
    "tui.altScreen.pageDown": KeybindingDefinition("pageDown", "Scroll viewport down one page"),
    "tui.altScreen.halfPageUp": KeybindingDefinition([], "Scroll viewport up half a page"),
    "tui.altScreen.halfPageDown": KeybindingDefinition([], "Scroll viewport down half a page"),
    "tui.altScreen.lineUp": KeybindingDefinition([], "Scroll viewport up one line"),
    "tui.altScreen.lineDown": KeybindingDefinition([], "Scroll viewport down one line"),
    "tui.altScreen.previousPrompt": KeybindingDefinition(
        ["ctrl+shift+up", "ctrl+up"], "Jump to previous semantic prompt"
    ),
    "tui.altScreen.nextPrompt": KeybindingDefinition(
        ["ctrl+shift+down", "ctrl+down"], "Jump to next semantic prompt"
    ),
    "tui.altScreen.search": KeybindingDefinition("ctrl+shift+f", "Search the primary scroll view"),
    "tui.altScreen.searchNext": KeybindingDefinition(
        ["enter", "ctrl+g"], "Select the next search match"
    ),
    "tui.altScreen.searchPrevious": KeybindingDefinition(
        ["shift+enter", "ctrl+shift+g"], "Select the previous search match"
    ),
    "tui.altScreen.searchClose": KeybindingDefinition("escape", "Close transcript search"),
    "tui.altScreen.top": KeybindingDefinition("home", "Scroll viewport to top"),
    "tui.altScreen.bottom": KeybindingDefinition("end", "Scroll viewport to bottom"),
}


def _normalize_keys(keys: KeyId | list[KeyId] | None) -> list[KeyId]:
    if keys is None:
        return []
    key_list = keys if isinstance(keys, list) else [keys]
    seen: set[KeyId] = set()
    result: list[KeyId] = []
    for key in key_list:
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


class KeybindingsManager:
    """Shared namespaced keybinding registry."""

    def __init__(
        self,
        definitions: KeybindingDefinitions,
        user_bindings: KeybindingsConfig | None = None,
    ) -> None:
        self._definitions = definitions
        self._user_bindings = dict(user_bindings or {})
        self._keys_by_id: dict[str, list[KeyId]] = {}
        self._conflicts: list[KeybindingConflict] = []
        self._rebuild()

    def _rebuild(self) -> None:
        self._keys_by_id.clear()
        self._conflicts = []

        user_claims: dict[KeyId, set[str]] = {}
        for keybinding, keys in self._user_bindings.items():
            if keybinding not in self._definitions:
                continue
            for key in _normalize_keys(keys):
                user_claims.setdefault(key, set()).add(keybinding)

        for key, keybindings in user_claims.items():
            if len(keybindings) > 1:
                self._conflicts.append(KeybindingConflict(key=key, keybindings=list(keybindings)))

        for binding_id, definition in self._definitions.items():
            user_keys = self._user_bindings.get(binding_id, None)
            keys = (
                _normalize_keys(definition.default_keys)
                if user_keys is None
                else _normalize_keys(user_keys)
            )
            self._keys_by_id[binding_id] = keys

    def matches(self, data: str, keybinding: Keybinding) -> bool:
        return any(matches_key(data, key) for key in self._keys_by_id.get(keybinding, []))

    def get_keys(self, keybinding: Keybinding) -> list[KeyId]:
        return list(self._keys_by_id.get(keybinding, []))

    def get_definition(self, keybinding: Keybinding) -> KeybindingDefinition:
        return self._definitions[keybinding]

    def get_conflicts(self) -> list[KeybindingConflict]:
        return [
            KeybindingConflict(key=c.key, keybindings=list(c.keybindings))
            for c in self._conflicts
        ]

    def set_user_bindings(self, user_bindings: KeybindingsConfig) -> None:
        self._user_bindings = dict(user_bindings)
        self._rebuild()

    def get_user_bindings(self) -> KeybindingsConfig:
        return dict(self._user_bindings)

    def get_resolved_bindings(self) -> KeybindingsConfig:
        resolved: KeybindingsConfig = {}
        for binding_id in self._definitions:
            keys = self._keys_by_id.get(binding_id, [])
            resolved[binding_id] = keys[0] if len(keys) == 1 else list(keys)
        return resolved


_global_keybindings: KeybindingsManager | None = None


def set_keybindings(keybindings: KeybindingsManager) -> None:
    global _global_keybindings
    _global_keybindings = keybindings


def get_keybindings() -> KeybindingsManager:
    global _global_keybindings
    if _global_keybindings is None:
        _global_keybindings = KeybindingsManager(TUI_KEYBINDINGS)
    return _global_keybindings


# ── Compatibility layer for the pre-0.61 editor-only store ───────────────────

EditorAction = Literal[
    "cursorUp",
    "cursorDown",
    "cursorLeft",
    "cursorRight",
    "cursorWordLeft",
    "cursorWordRight",
    "cursorLineStart",
    "cursorLineEnd",
    "jumpForward",
    "jumpBackward",
    "pageUp",
    "pageDown",
    "deleteCharBackward",
    "deleteCharForward",
    "deleteWordBackward",
    "deleteWordForward",
    "deleteToLineStart",
    "deleteToLineEnd",
    "newLine",
    "submit",
    "tab",
    "selectUp",
    "selectDown",
    "selectPageUp",
    "selectPageDown",
    "selectConfirm",
    "selectCancel",
    "copy",
    "yank",
    "yankPop",
    "undo",
    "expandTools",
    "treeFoldOrUp",
    "treeUnfoldOrDown",
    "toggleSessionPath",
    "toggleSessionSort",
    "renameSession",
    "deleteSession",
    "deleteSessionNoninvasive",
]

_LEGACY_TO_NAMESPACED: dict[str, str] = {
    "cursorUp": "tui.editor.cursorUp",
    "cursorDown": "tui.editor.cursorDown",
    "cursorLeft": "tui.editor.cursorLeft",
    "cursorRight": "tui.editor.cursorRight",
    "cursorWordLeft": "tui.editor.cursorWordLeft",
    "cursorWordRight": "tui.editor.cursorWordRight",
    "cursorLineStart": "tui.editor.cursorLineStart",
    "cursorLineEnd": "tui.editor.cursorLineEnd",
    "jumpForward": "tui.editor.jumpForward",
    "jumpBackward": "tui.editor.jumpBackward",
    "pageUp": "tui.editor.pageUp",
    "pageDown": "tui.editor.pageDown",
    "deleteCharBackward": "tui.editor.deleteCharBackward",
    "deleteCharForward": "tui.editor.deleteCharForward",
    "deleteWordBackward": "tui.editor.deleteWordBackward",
    "deleteWordForward": "tui.editor.deleteWordForward",
    "deleteToLineStart": "tui.editor.deleteToLineStart",
    "deleteToLineEnd": "tui.editor.deleteToLineEnd",
    "yank": "tui.editor.yank",
    "yankPop": "tui.editor.yankPop",
    "undo": "tui.editor.undo",
    "newLine": "tui.input.newLine",
    "submit": "tui.input.submit",
    "tab": "tui.input.tab",
    "copy": "tui.input.copy",
    "selectUp": "tui.select.up",
    "selectDown": "tui.select.down",
    "selectPageUp": "tui.select.pageUp",
    "selectPageDown": "tui.select.pageDown",
    "selectConfirm": "tui.select.confirm",
    "selectCancel": "tui.select.cancel",
}

EditorKeybindingsConfig = dict[str, "KeyId | list[KeyId]"]

DEFAULT_EDITOR_KEYBINDINGS: dict[str, list[KeyId]] = {
    "cursorUp": ["up"],
    "cursorDown": ["down"],
    "cursorLeft": ["left", "ctrl+b"],
    "cursorRight": ["right", "ctrl+f"],
    "cursorWordLeft": ["alt+left", "ctrl+left", "alt+b"],
    "cursorWordRight": ["alt+right", "ctrl+right", "alt+f"],
    "cursorLineStart": ["home", "ctrl+home", "ctrl+a"],
    "cursorLineEnd": ["end", "ctrl+end", "ctrl+e"],
    "jumpForward": ["ctrl+]"],
    "jumpBackward": ["ctrl+alt+]"],
    "pageUp": ["pageUp", "ctrl+pageUp"],
    "pageDown": ["pageDown", "ctrl+pageDown"],
    "deleteCharBackward": ["backspace"],
    "deleteCharForward": ["delete", "ctrl+d"],
    "deleteWordBackward": ["ctrl+w", "alt+backspace"],
    "deleteWordForward": ["alt+d", "alt+delete"],
    "deleteToLineStart": ["ctrl+u"],
    "deleteToLineEnd": ["ctrl+k"],
    "newLine": ["shift+enter", "ctrl+j"],
    "submit": ["enter"],
    "tab": ["tab"],
    "selectUp": ["up"],
    "selectDown": ["down"],
    "selectPageUp": ["pageUp"],
    "selectPageDown": ["pageDown"],
    "selectConfirm": ["enter"],
    "selectCancel": ["escape", "ctrl+c"],
    "copy": ["ctrl+c"],
    "yank": ["ctrl+y"],
    "yankPop": ["alt+y"],
    "undo": ["ctrl+-"],
    "expandTools": ["ctrl+o"],
    "treeFoldOrUp": ["ctrl+left", "alt+left"],
    "treeUnfoldOrDown": ["ctrl+right", "alt+right"],
    "toggleSessionPath": ["ctrl+p"],
    "toggleSessionSort": ["ctrl+s"],
    "renameSession": ["ctrl+r"],
    "deleteSession": ["ctrl+d"],
    "deleteSessionNoninvasive": ["ctrl+backspace"],
}


class EditorKeybindingsManager:
    """
    Compatibility wrapper around the namespaced ``KeybindingsManager``.
    Legacy action names such as ``selectCancel`` still work.
    """

    def __init__(self, config: EditorKeybindingsConfig | None = None) -> None:
        self._action_to_keys: dict[str, list[KeyId]] = {}
        self._build_maps(config or {})

    def _build_maps(self, config: EditorKeybindingsConfig) -> None:
        self._action_to_keys.clear()
        for action, keys in DEFAULT_EDITOR_KEYBINDINGS.items():
            self._action_to_keys[action] = list(keys)
        for action, keys in config.items():
            if keys is None:
                continue
            self._action_to_keys[action] = keys if isinstance(keys, list) else [keys]

    def matches(self, data: str, action: str) -> bool:
        keys = self._action_to_keys.get(action)
        if not keys:
            return False
        return any(matches_key(data, key) for key in keys)

    def get_keys(self, action: str) -> list[KeyId]:
        return list(self._action_to_keys.get(action, []))

    def set_config(self, config: EditorKeybindingsConfig) -> None:
        self._build_maps(config)


_global_editor_keybindings: EditorKeybindingsManager | None = None


def get_editor_keybindings() -> EditorKeybindingsManager:
    global _global_editor_keybindings
    if _global_editor_keybindings is None:
        _global_editor_keybindings = EditorKeybindingsManager()
    return _global_editor_keybindings


def set_editor_keybindings(manager: EditorKeybindingsManager) -> None:
    global _global_editor_keybindings
    _global_editor_keybindings = manager
