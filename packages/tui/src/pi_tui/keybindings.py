"""
Editor keybindings — mirrors packages/tui/src/keybindings.ts

Provides EditorAction type, DEFAULT_EDITOR_KEYBINDINGS, and
EditorKeybindingsManager class.
"""
from __future__ import annotations

from typing import Literal

from .keys import KeyId, matches_key

# ─────────────────────────────────────────────────────────────────────────────
# EditorAction type
# ─────────────────────────────────────────────────────────────────────────────

EditorAction = Literal[
    # Cursor movement
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
    # Deletion
    "deleteCharBackward",
    "deleteCharForward",
    "deleteWordBackward",
    "deleteWordForward",
    "deleteToLineStart",
    "deleteToLineEnd",
    # Text input
    "newLine",
    "submit",
    "tab",
    # Selection/autocomplete
    "selectUp",
    "selectDown",
    "selectPageUp",
    "selectPageDown",
    "selectConfirm",
    "selectCancel",
    # Clipboard
    "copy",
    # Kill ring
    "yank",
    "yankPop",
    # Undo
    "undo",
    # Tool output
    "expandTools",
    # Session
    "toggleSessionPath",
    "toggleSessionSort",
    "renameSession",
    "deleteSession",
    "deleteSessionNoninvasive",
]

# ─────────────────────────────────────────────────────────────────────────────
# Default keybindings — mirrors DEFAULT_EDITOR_KEYBINDINGS in keybindings.ts
# ─────────────────────────────────────────────────────────────────────────────

EditorKeybindingsConfig = dict[str, "KeyId | list[KeyId]"]

DEFAULT_EDITOR_KEYBINDINGS: dict[str, list[KeyId]] = {
    # Cursor movement
    "cursorUp":          ["up"],
    "cursorDown":        ["down"],
    "cursorLeft":        ["left", "ctrl+b"],
    "cursorRight":       ["right", "ctrl+f"],
    "cursorWordLeft":    ["alt+left", "ctrl+left", "alt+b"],
    "cursorWordRight":   ["alt+right", "ctrl+right", "alt+f"],
    "cursorLineStart":   ["home", "ctrl+a"],
    "cursorLineEnd":     ["end", "ctrl+e"],
    "jumpForward":       ["ctrl+]"],
    "jumpBackward":      ["ctrl+alt+]"],
    "pageUp":            ["pageUp"],
    "pageDown":          ["pageDown"],
    # Deletion
    "deleteCharBackward": ["backspace"],
    "deleteCharForward":  ["delete", "ctrl+d"],
    "deleteWordBackward": ["ctrl+w", "alt+backspace"],
    "deleteWordForward":  ["alt+d", "alt+delete"],
    "deleteToLineStart":  ["ctrl+u"],
    "deleteToLineEnd":    ["ctrl+k"],
    # Text input
    "newLine":  ["shift+enter"],
    "submit":   ["enter"],
    "tab":      ["tab"],
    # Selection/autocomplete
    "selectUp":       ["up"],
    "selectDown":     ["down"],
    "selectPageUp":   ["pageUp"],
    "selectPageDown": ["pageDown"],
    "selectConfirm":  ["enter"],
    "selectCancel":   ["escape", "ctrl+c"],
    # Clipboard
    "copy": ["ctrl+c"],
    # Kill ring
    "yank":    ["ctrl+y"],
    "yankPop": ["alt+y"],
    # Undo
    "undo": ["ctrl+-"],
    # Tool output
    "expandTools": ["ctrl+o"],
    # Session
    "toggleSessionPath":        ["ctrl+p"],
    "toggleSessionSort":        ["ctrl+s"],
    "renameSession":            ["ctrl+r"],
    "deleteSession":            ["ctrl+d"],
    "deleteSessionNoninvasive": ["ctrl+backspace"],
}


# ─────────────────────────────────────────────────────────────────────────────
# EditorKeybindingsManager
# ─────────────────────────────────────────────────────────────────────────────

class EditorKeybindingsManager:
    """
    Manages keybindings for the editor.
    Mirrors EditorKeybindingsManager in keybindings.ts.
    """

    def __init__(self, config: EditorKeybindingsConfig | None = None) -> None:
        self._action_to_keys: dict[str, list[KeyId]] = {}
        self._build_maps(config or {})

    def _build_maps(self, config: EditorKeybindingsConfig) -> None:
        self._action_to_keys.clear()
        # Start with defaults
        for action, keys in DEFAULT_EDITOR_KEYBINDINGS.items():
            self._action_to_keys[action] = list(keys)
        # Override with user config
        for action, keys in config.items():
            if keys is None:
                continue
            self._action_to_keys[action] = keys if isinstance(keys, list) else [keys]

    def matches(self, data: str, action: str) -> bool:
        """Check if input data matches a specific action."""
        keys = self._action_to_keys.get(action)
        if not keys:
            return False
        return any(matches_key(data, k) for k in keys)

    def get_keys(self, action: str) -> list[KeyId]:
        """Get keys bound to an action."""
        return self._action_to_keys.get(action, [])

    def set_config(self, config: EditorKeybindingsConfig) -> None:
        """Update configuration."""
        self._build_maps(config)


# ─────────────────────────────────────────────────────────────────────────────
# Global instance — mirrors getEditorKeybindings / setEditorKeybindings
# ─────────────────────────────────────────────────────────────────────────────

_global_editor_keybindings: EditorKeybindingsManager | None = None


def get_editor_keybindings() -> EditorKeybindingsManager:
    global _global_editor_keybindings
    if _global_editor_keybindings is None:
        _global_editor_keybindings = EditorKeybindingsManager()
    return _global_editor_keybindings


def set_editor_keybindings(manager: EditorKeybindingsManager) -> None:
    global _global_editor_keybindings
    _global_editor_keybindings = manager
