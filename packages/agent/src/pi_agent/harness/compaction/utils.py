"""Compaction file-op helpers — mirrors harness/compaction/utils.ts."""
from __future__ import annotations

import json
from typing import Any, TypedDict

from pi_ai.types import Message
from pi_ai.utils.text import content_text

from pi_agent.types import AgentMessage


class FileOperations(TypedDict):
    read: set[str]
    written: set[str]
    edited: set[str]


def create_file_ops() -> FileOperations:
    return {"read": set(), "written": set(), "edited": set()}


def _role(message: Any) -> str | None:
    return getattr(message, "role", None) if not isinstance(message, dict) else message.get("role")


def _content(message: Any) -> Any:
    return getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")


def extract_file_ops_from_message(message: AgentMessage, file_ops: FileOperations) -> None:
    if _role(message) != "assistant":
        return
    content = _content(message)
    if not isinstance(content, list):
        return
    for block in content:
        block_type = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
        if block_type != "toolCall":
            continue
        args = getattr(block, "arguments", None) if not isinstance(block, dict) else block.get("arguments")
        name = getattr(block, "name", None) if not isinstance(block, dict) else block.get("name")
        if not isinstance(args, dict):
            continue
        path = args.get("path") if isinstance(args.get("path"), str) else None
        if not path:
            continue
        if name == "read":
            file_ops["read"].add(path)
        elif name == "write":
            file_ops["written"].add(path)
        elif name == "edit":
            file_ops["edited"].add(path)


def compute_file_lists(file_ops: FileOperations) -> dict[str, list[str]]:
    modified = set(file_ops["edited"]) | set(file_ops["written"])
    read_only = sorted(path for path in file_ops["read"] if path not in modified)
    return {"read_files": read_only, "modified_files": sorted(modified)}


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    sections: list[str] = []
    if read_files:
        sections.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        sections.append("<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


TOOL_RESULT_MAX_CHARS = 2000


def _safe_json_stringify(value: Any) -> str:
    try:
        encoded = json.dumps(value)
        return "undefined" if encoded is None else encoded
    except Exception:
        return "[unserializable]"


def _truncate_for_summary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated} more characters truncated]"


def serialize_conversation(messages: list[Message]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = _role(msg)
        if role == "user":
            content = content_text(_content(msg), "")
            if content:
                parts.append(f"[User]: {content}")
        elif role == "assistant":
            thinking_parts: list[str] = []
            tool_calls: list[str] = []
            for block in _content(msg) or []:
                block_type = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
                if block_type == "thinking":
                    thinking_parts.append(getattr(block, "thinking", None) if not isinstance(block, dict) else block.get("thinking"))
                elif block_type == "toolCall":
                    args = getattr(block, "arguments", {}) if not isinstance(block, dict) else block.get("arguments") or {}
                    args_str = ", ".join(f"{key}={_safe_json_stringify(value)}" for key, value in args.items())
                    name = getattr(block, "name", None) if not isinstance(block, dict) else block.get("name")
                    tool_calls.append(f"{name}({args_str})")
            if thinking_parts:
                parts.append("[Assistant thinking]: " + "\n".join(thinking_parts))
            content_blocks = _content(msg) or []
            if any((getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")) == "text" for block in content_blocks):
                parts.append(f"[Assistant]: {content_text(content_blocks)}")
            if tool_calls:
                parts.append("[Assistant tool calls]: " + "; ".join(tool_calls))
        elif role == "toolResult":
            content = content_text(_content(msg), "")
            if content:
                parts.append(f"[Tool result]: {_truncate_for_summary(content, TOOL_RESULT_MAX_CHARS)}")
    return "\n\n".join(parts)
