"""
Keybindings management — mirrors packages/coding-agent/src/core/keybindings.ts

Provides KeybindingsManager and DEFAULT_KEYBINDINGS for configurable app/editor actions.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Union


# Application-level actions (coding agent specific)
APP_ACTIONS: list[str] = [
    "interrupt",
    "clear",
    "exit",
    "suspend",
    "cycleThinkingLevel",
    "cycleModelForward",
    "cycleModelBackward",
    "selectModel",
    "expandTools",
    "toggleThinking",
    "toggleSessionNamedFilter",
    "externalEditor",
    "followUp",
    "dequeue",
    "pasteImage",
    "newSession",
    "tree",
    "fork",
    "resume",
]

# Default editor actions (subset, since we don't use pi-tui)
EDITOR_ACTIONS: list[str] = [
    "moveLeft",
    "moveRight",
    "moveWordLeft",
    "moveWordRight",
    "moveToLineStart",
    "moveToLineEnd",
    "moveUp",
    "moveDown",
    "selectLeft",
    "selectRight",
    "selectWordLeft",
    "selectWordRight",
    "selectToLineStart",
    "selectToLineEnd",
    "selectAll",
    "deleteLeft",
    "deleteRight",
    "deleteWordLeft",
    "deleteWordRight",
    "deleteToLineStart",
    "deleteToLineEnd",
    "newline",
    "submit",
    "historyPrev",
    "historyNext",
    "tab",
]

# KeyId = str (e.g. "escape", "ctrl+c", "alt+enter")
KeyId = str

# Default application keybindings — mirrors DEFAULT_APP_KEYBINDINGS in TS
DEFAULT_APP_KEYBINDINGS: dict[str, Union[KeyId, list[KeyId]]] = {
    "interrupt": "escape",
    "clear": "ctrl+c",
    "exit": "ctrl+d",
    "suspend": "ctrl+z",
    "cycleThinkingLevel": "shift+tab",
    "cycleModelForward": "ctrl+p",
    "cycleModelBackward": "shift+ctrl+p",
    "selectModel": "ctrl+l",
    "expandTools": "ctrl+o",
    "toggleThinking": "ctrl+t",
    "toggleSessionNamedFilter": "ctrl+n",
    "externalEditor": "ctrl+g",
    "followUp": "alt+enter",
    "dequeue": "alt+up",
    "pasteImage": "ctrl+v",
    "newSession": [],
    "tree": [],
    "fork": [],
    "resume": [],
}

# Default editor keybindings (readline-compatible)
DEFAULT_EDITOR_KEYBINDINGS: dict[str, Union[KeyId, list[KeyId]]] = {
    "moveLeft": "left",
    "moveRight": "right",
    "moveWordLeft": "ctrl+left",
    "moveWordRight": "ctrl+right",
    "moveToLineStart": "home",
    "moveToLineEnd": "end",
    "moveUp": "up",
    "moveDown": "down",
    "selectLeft": "shift+left",
    "selectRight": "shift+right",
    "selectWordLeft": "shift+ctrl+left",
    "selectWordRight": "shift+ctrl+right",
    "selectToLineStart": "shift+home",
    "selectToLineEnd": "shift+end",
    "selectAll": "ctrl+a",
    "deleteLeft": "backspace",
    "deleteRight": "delete",
    "deleteWordLeft": "ctrl+backspace",
    "deleteWordRight": "ctrl+delete",
    "deleteToLineStart": "ctrl+u",
    "deleteToLineEnd": "ctrl+k",
    "newline": "shift+enter",
    "submit": "enter",
    "historyPrev": "ctrl+up",
    "historyNext": "ctrl+down",
    "tab": "tab",
}

# All default keybindings (app + editor) — mirrors DEFAULT_KEYBINDINGS in TS
DEFAULT_KEYBINDINGS: dict[str, Union[KeyId, list[KeyId]]] = {
    **DEFAULT_EDITOR_KEYBINDINGS,
    **DEFAULT_APP_KEYBINDINGS,
}


class KeybindingsManager:
    """
    Manages all keybindings (app + editor).
    Mirrors KeybindingsManager in TypeScript.
    """

    def __init__(self, config: dict[str, Union[KeyId, list[KeyId]]] | None = None):
        self._config: dict[str, Union[KeyId, list[KeyId]]] = config or {}
        self._app_action_to_keys: dict[str, list[KeyId]] = {}
        self._build_maps()

    @classmethod
    def create(cls, agent_dir: str | None = None) -> "KeybindingsManager":
        """
        Load keybindings from ~/.pi/agent/keybindings.json and merge with defaults.
        """
        if agent_dir is None:
            agent_dir = os.path.join(os.path.expanduser("~"), ".pi", "agent")

        keybindings_path = os.path.join(agent_dir, "keybindings.json")
        user_config: dict = {}

        if os.path.exists(keybindings_path):
            try:
                with open(keybindings_path, encoding="utf-8") as f:
                    user_config = json.load(f)
            except Exception:
                pass

        merged = {**DEFAULT_KEYBINDINGS, **user_config}
        return cls(merged)

    def _build_maps(self) -> None:
        """Build internal lookup maps from config."""
        self._app_action_to_keys.clear()
        for action in APP_ACTIONS:
            keys = self._config.get(action) or DEFAULT_APP_KEYBINDINGS.get(action, [])
            if isinstance(keys, str):
                self._app_action_to_keys[action] = [keys]
            else:
                self._app_action_to_keys[action] = list(keys)

    def get_keys_for_action(self, action: str) -> list[KeyId]:
        """Get the list of key IDs bound to an action."""
        if action in APP_ACTIONS:
            return self._app_action_to_keys.get(action, [])
        # Editor actions
        keys = self._config.get(action) or DEFAULT_EDITOR_KEYBINDINGS.get(action, [])
        if isinstance(keys, str):
            return [keys]
        return list(keys)

    def matches(self, action: str, key: KeyId) -> bool:
        """Check if a key matches an action."""
        return key in self.get_keys_for_action(action)

    def get_config(self) -> dict[str, Union[KeyId, list[KeyId]]]:
        """Get the full keybindings config."""
        return dict(self._config)

    def set_keybinding(self, action: str, keys: Union[KeyId, list[KeyId]]) -> None:
        """Update a keybinding at runtime."""
        self._config[action] = keys
        self._build_maps()
