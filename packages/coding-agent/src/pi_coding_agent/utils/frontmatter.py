"""
YAML frontmatter parser for markdown content.

Mirrors utils/frontmatter.ts
"""

from __future__ import annotations

import re
from typing import Any

import yaml


_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from a string.

    Returns:
        (metadata, body) where metadata is the parsed YAML dict
        and body is the remaining content after the frontmatter.
    """
    match = _FM_RE.match(content)
    if not match:
        return {}, content

    try:
        data = yaml.safe_load(match.group(1)) or {}
        if not isinstance(data, dict):
            data = {}
    except yaml.YAMLError:
        data = {}

    body = content[match.end():]
    return data, body


def strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from the beginning of a string."""
    _, body = parse_frontmatter(content)
    return body


def stringify_frontmatter(metadata: dict[str, Any], body: str) -> str:
    """Prepend YAML frontmatter to body text."""
    if not metadata:
        return body
    fm = yaml.safe_dump(metadata, default_flow_style=False, allow_unicode=True).rstrip()
    return f"---\n{fm}\n---\n{body}"
