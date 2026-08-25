"""
Text helpers — mirrors packages/coding-agent/src/utils/text.ts
"""
from __future__ import annotations

from typing import Any


def split_bom(content: str) -> tuple[str, str]:
    """Split a leading UTF-8 byte order mark from decoded text.

    Returns (bom, text) where bom is '\\ufeff' or ''.
    """
    if content.startswith("\ufeff"):
        return "\ufeff", content[1:]
    return "", content


def strip_bom(content: str) -> str:
    """Remove a leading UTF-8 byte order mark from decoded text."""
    return split_bom(content)[1]


def loads_json(content: str) -> Any:
    """Parse JSON after stripping a leading UTF-8 BOM."""
    import json
    return json.loads(strip_bom(content))


def load_json_file(path: str) -> Any:
    """Read a JSON file, stripping a leading UTF-8 BOM."""
    with open(path, encoding="utf-8") as f:
        return loads_json(f.read())
