"""
Edit file tool — mirrors packages/coding-agent/src/core/tools/edit.ts

Performs exact text replacement with fuzzy matching support.
"""
from __future__ import annotations

import asyncio
import difflib
import os
import re
from typing import Any, Callable

import aiofiles

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent

from .path_utils import resolve_to_cwd


def normalize_to_lf(text: str) -> str:
    """Convert all line endings to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def detect_line_ending(text: str) -> str:
    """Detect the dominant line ending in text."""
    crlf_count = text.count("\r\n")
    cr_count = text.count("\r") - crlf_count
    lf_count = text.count("\n") - crlf_count
    if crlf_count >= lf_count and crlf_count >= cr_count:
        return "\r\n"
    elif cr_count > lf_count:
        return "\r"
    return "\n"


def restore_line_endings(text: str, ending: str) -> str:
    """Restore line endings in text."""
    if ending == "\r\n":
        return text.replace("\n", "\r\n")
    elif ending == "\r":
        return text.replace("\n", "\r")
    return text


def strip_bom(text: str) -> tuple[str, str]:
    """Strip BOM from text, return (bom, text_without_bom)."""
    if text.startswith("\ufeff"):
        return "\ufeff", text[1:]
    return "", text


def normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fuzzy matching: collapse multiple whitespace to single space."""
    return re.sub(r"\s+", " ", text).strip()


def fuzzy_find_text(
    content: str,
    old_text: str,
) -> dict[str, Any]:
    """
    Find old_text in content, trying exact match first then fuzzy.
    Returns {"found": bool, "index": int, "match_length": int, "content_for_replacement": str}
    """
    # Try exact match first
    idx = content.find(old_text)
    if idx >= 0:
        return {
            "found": True,
            "index": idx,
            "match_length": len(old_text),
            "content_for_replacement": content,
        }

    # Try fuzzy: normalize both
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old_text)

    fuzzy_idx = fuzzy_content.find(fuzzy_old)
    if fuzzy_idx >= 0:
        return {
            "found": True,
            "index": fuzzy_idx,
            "match_length": len(fuzzy_old),
            "content_for_replacement": fuzzy_content,
        }

    return {"found": False, "index": -1, "match_length": 0, "content_for_replacement": content}


def generate_diff_string(old_content: str, new_content: str) -> dict[str, Any]:
    """Generate unified diff string between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile="original",
        tofile="modified",
        n=3,
    ))

    diff_str = "".join(diff)
    first_changed_line: int | None = None

    for i, line in enumerate(diff):
        if line.startswith("@@"):
            # Parse @@ -a,b +c,d @@ — extract c
            m = re.search(r"\+(\d+)", line)
            if m:
                first_changed_line = int(m.group(1))
                break

    return {"diff": diff_str, "first_changed_line": first_changed_line}


def create_edit_tool(cwd: str) -> AgentTool:
    """
    Create an edit tool for the given working directory.
    Mirrors createEditTool() in TypeScript.
    """

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: asyncio.Event | None = None,
        on_update: Callable | None = None,
    ) -> AgentToolResult:
        path: str = params["path"]
        old_text: str = params["oldText"]
        new_text: str = params["newText"]

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation aborted")

        absolute_path = resolve_to_cwd(path, cwd)

        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"File not found: {path}")
        if not os.access(absolute_path, os.R_OK | os.W_OK):
            raise PermissionError(f"Cannot read/write file: {path}")

        async with aiofiles.open(absolute_path, "r", encoding="utf-8", errors="replace") as f:
            raw_content = await f.read()

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation aborted")

        bom, content = strip_bom(raw_content)
        original_ending = detect_line_ending(content)
        normalized_content = normalize_to_lf(content)
        normalized_old = normalize_to_lf(old_text)
        normalized_new = normalize_to_lf(new_text)

        match_result = fuzzy_find_text(normalized_content, normalized_old)
        if not match_result["found"]:
            raise ValueError(
                f"Could not find the exact text in {path}. "
                "The old text must match exactly including all whitespace and newlines."
            )

        # Count non-overlapping occurrences — mirrors TS: split(old_str).length - 1
        fuzzy_content = normalize_for_fuzzy_match(normalized_content)
        fuzzy_old = normalize_for_fuzzy_match(normalized_old)
        occurrences = len(fuzzy_content.split(fuzzy_old)) - 1
        if occurrences > 1:
            raise ValueError(
                f"Found {occurrences} occurrences of the text in {path}. "
                "The text must be unique. Please provide more context to make it unique."
            )

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation aborted")

        base_content = match_result["content_for_replacement"]
        idx = match_result["index"]
        match_len = match_result["match_length"]

        new_content = base_content[:idx] + normalized_new + base_content[idx + match_len:]

        if base_content == new_content:
            raise ValueError(
                f"No changes made to {path}. "
                "The replacement produced identical content."
            )

        final_content = bom + restore_line_endings(new_content, original_ending)
        async with aiofiles.open(absolute_path, "w", encoding="utf-8") as f:
            await f.write(final_content)

        diff_result = generate_diff_string(base_content, new_content)

        return AgentToolResult(
            content=[TextContent(type="text", text=f"Successfully replaced text in {path}.")],
            details={
                "diff": diff_result["diff"],
                "first_changed_line": diff_result["first_changed_line"],
            },
        )

    return AgentTool(
        name="edit",
        label="edit",
        description=(
            "Edit a file by replacing exact text. "
            "The oldText must match exactly (including whitespace). "
            "Use this for precise, surgical edits."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
                "oldText": {"type": "string", "description": "Exact text to find and replace (must match exactly)"},
                "newText": {"type": "string", "description": "New text to replace the old text with"},
            },
            "required": ["path", "oldText", "newText"],
        },
        execute=execute,
    )


edit_tool = create_edit_tool(os.getcwd())
