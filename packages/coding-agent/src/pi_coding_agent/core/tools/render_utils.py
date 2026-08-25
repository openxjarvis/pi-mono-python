"""
Tool render helpers — mirrors packages/coding-agent/src/core/tools/render-utils.ts
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pi_coding_agent.utils.ansi import strip_ansi
from pi_coding_agent.utils.paths import resolve_path
from pi_coding_agent.utils.shell import sanitize_binary_output


def shorten_path(path: Any) -> str:
    if not isinstance(path, str):
        return ""
    home = str(Path.home())
    if path.startswith(home):
        return f"~{path[len(home):]}"
    return path


def link_path(styled_text: str, raw_path: str, cwd: str) -> str:
    return styled_text


def str_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return None


def replace_tabs(text: str) -> str:
    return text.replace("\t", "   ")


def normalize_display_text(text: str) -> str:
    return text.replace("\r", "")


def get_text_output(result: Any | None, show_images: bool = True) -> str:
    if not result:
        return ""
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if not isinstance(content, list):
        return ""
    text_blocks = []
    image_blocks = []
    for item in content:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type == "text":
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", "")
            text_blocks.append(sanitize_binary_output(strip_ansi(text or "")).replace("\r", ""))
        elif item_type == "image":
            image_blocks.append(item)
    output = "\n".join(text_blocks)
    if image_blocks and not show_images:
        indicators = "\n".join("[image]" for _ in image_blocks)
        output = f"{output}\n{indicators}" if output else indicators
    return output


def invalid_arg_text(theme: Any | None = None) -> str:
    if theme is not None and hasattr(theme, "fg"):
        return theme.fg("error", "[invalid arg]")
    return "[invalid arg]"


def render_tool_path(
    raw_path: str | None,
    theme: Any,
    cwd: str,
    empty_fallback: str | None = None,
) -> str:
    if raw_path is None:
        return invalid_arg_text(theme)
    value = raw_path or empty_fallback
    if not value:
        return theme.fg("toolOutput", "...") if hasattr(theme, "fg") else "..."
    styled = theme.fg("accent", shorten_path(value)) if hasattr(theme, "fg") else shorten_path(value)
    return link_path(styled, value, cwd)
