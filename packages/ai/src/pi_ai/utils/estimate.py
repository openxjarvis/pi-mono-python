"""
Context token estimation.
Mirrors packages/ai/src/utils/estimate.ts
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pi_ai.types import Context, Message, Usage

CHARS_PER_TOKEN = 4
ESTIMATED_IMAGE_CHARS = 4800


@dataclass
class ContextUsageEstimate:
    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


def calculate_context_tokens(usage: Usage) -> int:
    total = getattr(usage, "total_tokens", None) or 0
    if total:
        return int(total)
    return int(
        (getattr(usage, "input", 0) or 0)
        + (getattr(usage, "output", 0) or 0)
        + (getattr(usage, "cache_read", 0) or 0)
        + (getattr(usage, "cache_write", 0) or 0)
    )


def _safe_json_stringify(value: Any) -> str:
    try:
        serialized = json.dumps(value, default=str)
        return "undefined" if serialized is None else serialized
    except Exception:
        return "[unserializable]"


def estimate_text_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _estimate_text_and_image_content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    chars = 0
    for block in content or []:
        if getattr(block, "type", None) == "text":
            chars += len(getattr(block, "text", "") or "")
        else:
            chars += ESTIMATED_IMAGE_CHARS
    return chars


def estimate_text_and_image_content_tokens(content: Any) -> int:
    return (_estimate_text_and_image_content_chars(content) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def estimate_message_tokens(message: Message) -> int:
    role = getattr(message, "role", None)
    if role in ("user", "toolResult"):
        return estimate_text_and_image_content_tokens(getattr(message, "content", ""))
    chars = 0
    for block in getattr(message, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            chars += len(getattr(block, "text", "") or "")
        elif block_type == "thinking":
            chars += len(getattr(block, "thinking", "") or "")
        else:
            chars += len(getattr(block, "name", "") or "") + len(
                _safe_json_stringify(getattr(block, "arguments", None))
            )
    return (chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _get_last_assistant_usage_info(messages: list[Message]) -> tuple[Usage, int] | None:
    latest_prefix_timestamp = float("-inf")
    usage_info: tuple[Usage, int] | None = None
    for i, message in enumerate(messages):
        if getattr(message, "role", None) == "assistant":
            usage_applies = getattr(message, "timestamp", 0) >= latest_prefix_timestamp
            stop = getattr(message, "stop_reason", None)
            usage = getattr(message, "usage", None)
            if (
                usage_applies
                and stop not in ("aborted", "error")
                and usage is not None
                and calculate_context_tokens(usage) > 0
            ):
                usage_info = (usage, i)
        latest_prefix_timestamp = max(latest_prefix_timestamp, getattr(message, "timestamp", 0) or 0)
    return usage_info


def _estimate_messages(messages: list[Message]) -> ContextUsageEstimate:
    usage_info = _get_last_assistant_usage_info(messages)
    if usage_info:
        usage, index = usage_info
        usage_tokens = calculate_context_tokens(usage)
        trailing = sum(estimate_message_tokens(messages[i]) for i in range(index + 1, len(messages)))
        return ContextUsageEstimate(usage_tokens + trailing, usage_tokens, trailing, index)
    tokens = sum(estimate_message_tokens(m) for m in messages)
    return ContextUsageEstimate(tokens, 0, tokens, None)


def _estimate_tools_tokens(tools: list[Any] | None) -> int:
    if not tools:
        return 0
    return estimate_text_tokens(_safe_json_stringify([
        t.model_dump() if hasattr(t, "model_dump") else t for t in tools
    ]))


def estimate_context_tokens(context: Context | list[Message]) -> ContextUsageEstimate:
    if isinstance(context, list):
        return _estimate_messages(context)

    estimate = _estimate_messages(list(context.messages or []))
    if estimate.last_usage_index is not None:
        added_names: set[str] = set()
        for message in context.messages[estimate.last_usage_index + 1 :]:
            if getattr(message, "role", None) == "toolResult":
                added_names.update(getattr(message, "added_tool_names", None) or [])
        added_tools = [t for t in (context.tools or []) if t.name in added_names]
        added_tokens = _estimate_tools_tokens(added_tools)
        return ContextUsageEstimate(
            estimate.tokens + added_tokens,
            estimate.usage_tokens,
            estimate.trailing_tokens + added_tokens,
            estimate.last_usage_index,
        )

    prefix = (estimate_text_tokens(context.system_prompt) if context.system_prompt else 0) + _estimate_tools_tokens(
        context.tools
    )
    return ContextUsageEstimate(
        estimate.tokens + prefix,
        estimate.usage_tokens,
        estimate.trailing_tokens + prefix,
        estimate.last_usage_index,
    )
