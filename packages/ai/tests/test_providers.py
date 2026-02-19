"""Tests for provider utilities."""
from __future__ import annotations

import pytest

from pi_ai.utils.json_parse import parse_partial_json
from pi_ai.utils.validation import validate_tool_arguments
from pi_ai.types import Tool, ToolCall
from pi_ai.providers.transform_messages import transform_messages
from pi_ai import Context, UserMessage, AssistantMessage, TextContent, ToolCall, Usage
import time


# ── JSON parse tests ────────────────────────────────────────────────────────

def test_parse_partial_json_complete():
    result = parse_partial_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_parse_partial_json_truncated():
    # Missing closing brace
    result = parse_partial_json('{"key": "val')
    # Should return None or partial result
    # The parser attempts to fix it
    assert result is None or isinstance(result, dict)


def test_parse_partial_json_empty():
    assert parse_partial_json("") is None
    assert parse_partial_json("   ") is None


def test_parse_partial_json_nested():
    result = parse_partial_json('{"a": {"b": 1}, "c": [1, 2]}')
    assert result == {"a": {"b": 1}, "c": [1, 2]}


# ── Validation tests ─────────────────────────────────────────────────────────

def make_tool() -> Tool:
    return Tool(
        name="calculator",
        description="Calculates things",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    )


def make_tool_call(args: dict) -> ToolCall:
    return ToolCall(type="toolCall", id="tc1", name="calculator", arguments=args)


def test_validate_tool_arguments_valid():
    tool = make_tool()
    tc = make_tool_call({"a": 1, "b": 2})
    result = validate_tool_arguments(tool, tc)
    assert result == {"a": 1, "b": 2}


def test_validate_tool_arguments_missing_required():
    tool = make_tool()
    tc = make_tool_call({"a": 1})  # missing "b"
    with pytest.raises(ValueError, match="Missing required parameter"):
        validate_tool_arguments(tool, tc)


def test_validate_tool_arguments_extra_fields_ok():
    tool = make_tool()
    tc = make_tool_call({"a": 1, "b": 2, "extra": "ok"})
    result = validate_tool_arguments(tool, tc)
    assert "a" in result


# ── Transform messages tests ──────────────────────────────────────────────────

def test_transform_messages_passthrough():
    ts = int(time.time() * 1000)
    context = Context(
        system_prompt="You are helpful",
        messages=[
            UserMessage(role="user", content="Hello", timestamp=ts),
        ],
    )
    result = transform_messages(context, "anthropic-messages")
    assert len(result.messages) == 1
    assert result.system_prompt == "You are helpful"


def test_transform_messages_thinking_to_text():
    from pi_ai.types import ThinkingContent
    ts = int(time.time() * 1000)
    assistant_msg = AssistantMessage(
        role="assistant",
        content=[
            ThinkingContent(type="thinking", thinking="I think..."),
            TextContent(type="text", text="Answer"),
        ],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        usage=Usage(),
        stop_reason="stop",
        timestamp=ts,
    )
    context = Context(
        messages=[
            UserMessage(role="user", content="Hello", timestamp=ts),
            assistant_msg,
        ],
    )
    result = transform_messages(context, "openai-completions")
    # Thinking blocks should be converted to text
    for msg in result.messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                assert not isinstance(block, ThinkingContent), "Thinking block should be converted"
                if isinstance(block, TextContent):
                    # Either it's a <thinking> delimited text or the answer text
                    pass
