"""
Extract joined text from message content.
Mirrors packages/ai/src/utils/text.ts
"""
from __future__ import annotations

from typing import Any


def content_text(content: str | list[Any], separator: str = "\n") -> str:
    if isinstance(content, str):
        return content
    return separator.join(
        getattr(block, "text", "")
        for block in content
        if getattr(block, "type", None) == "text"
    )
