"""Keybinding hint helpers. Mirrors keybinding-hints.ts"""
from __future__ import annotations


def key_hint(key: str, label: str) -> str:
    return f"{key} {label}"


def key_text(key: str) -> str:
    return key


def raw_key_hint(text: str) -> str:
    return text

from pi_tui.components.text import Text
class KeybindingHintsComponent(Text):
    def __init__(self, text: str = ''):
        super().__init__(text or 'hints')
