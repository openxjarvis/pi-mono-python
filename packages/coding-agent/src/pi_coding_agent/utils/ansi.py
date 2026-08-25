"""
ANSI stripping — mirrors packages/coding-agent/src/utils/ansi.ts
"""
from __future__ import annotations

import re

_ST = r"(?:\u0007|\u001B\u005C|\u009C)"
_OSC = rf"(?:\u001B\][\s\S]*?{_ST})"
_CSI = r"[\u001B\u009B][\[\]()#;?]*(?:\d{1,4}(?:[;:]\d{0,4})*)?[\dA-PR-TZcf-nq-uy=><~]"
_ANSI_RE = re.compile(f"{_OSC}|{_CSI}")


def strip_ansi(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected a string, got `{type(value).__name__}`")
    if "\u001b" not in value and "\u009b" not in value:
        return value
    return _ANSI_RE.sub("", value)
