"""Diff renderer. Mirrors packages/coding-agent/src/modes/interactive/components/diff.ts"""
from __future__ import annotations

import difflib
from typing import Any


def render_diff(before: str, after: str, path: str = "file", options: dict[str, Any] | None = None) -> str:
    lines = difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="")
    return "\n".join(lines)

from pi_tui.components.text import Text
class DiffComponent(Text):
    def __init__(self, before: str = '', after: str = '', path: str = 'file'):
        from .diff import render_diff
        super().__init__(render_diff(before, after, path))
