"""
Tests for agent_loop — mirrors packages/agent/test/agent-loop.test.ts
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from pi_ai.types import (
    AssistantMessage,
    Context,
    EventDone,
    EventError,
    EventStart,
    EventTextDelta,
    EventTextEnd,
    EventTextStart,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_agent import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    agent_loop,
)
from pi_agent.types import AgentEventAgentEnd, AgentEventAgentStart, AgentEventMessageEnd


def _ts() -> int:
    return int(time.time() * 1000)


def make_user_message(text: str = "Hello") -> UserMessage:
    return UserMessage(role="user", content=text, timestamp=_ts())


def make_assistant_message(model_id: str = "test-model", text: str = "Hi!") -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        api="anthropic-messages",
        provider="anthropic",
        model=model_id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=_ts(),
    )


async def _mock_stream_fn(model, context, options=None) -> AsyncGenerator:
    """Mock stream function that returns a simple text response."""
    partial = AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=_ts(),
    )
    yield EventStart(type="start", partial=partial)

    with_text = partial.model_copy(update={"content": [TextContent(type="text", text="")]})
    yield EventTextStart(type="text_start", content_index=0, partial=with_text)

    with_delta = partial.model_copy(update={"content": [TextContent(type="text", text="Hi!")]})
    yield EventTextDelta(type="text_delta", content_index=0, delta="Hi!", partial=with_delta)
    yield EventTextEnd(type="text_end", content_index=0, content="Hi!", partial=with_delta)

    final = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="Hi!")],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=_ts(),
    )
    yield EventDone(type="done", reason="stop", message=final)


@pytest.mark.asyncio
async def test_agent_loop_basic():
    """Test that agent_loop emits agent_start, message events, and agent_end."""
    from pi_ai import get_model
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")

    context = AgentContext(system_prompt="You are helpful", messages=[])
    prompts = [make_user_message("Hello")]

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")],
    )

    event_types = []
    stream = agent_loop(prompts, context, config, stream_fn=_mock_stream_fn)

    async for event in stream:
        event_types.append(event.type)

    assert "agent_start" in event_types
    assert "agent_end" in event_types
    assert "message_start" in event_types
    assert "message_end" in event_types


@pytest.mark.asyncio
async def test_agent_loop_returns_new_messages():
    """Test that agent_loop returns the new messages via result()."""
    from pi_ai import get_model
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")

    context = AgentContext(messages=[])
    prompts = [make_user_message("Hello")]

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")],
    )

    stream = agent_loop(prompts, context, config, stream_fn=_mock_stream_fn)
    # Drain the stream
    async for _ in stream:
        pass

    result = await stream.result()
    assert len(result) >= 1  # At least the user prompt


@pytest.mark.asyncio
async def test_agent_loop_with_tool():
    """Test that tools get called when the assistant returns a tool_use block."""
    from pi_ai import get_model
    from pi_ai.types import Model, ModelCost
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")

    tool_executed = []

    async def execute_calculator(tool_call_id, params, cancel=None, on_update=None):
        tool_executed.append(params)
        return AgentToolResult(
            content=[TextContent(type="text", text=str(params.get("a", 0) + params.get("b", 0)))],
            details={"sum": params.get("a", 0) + params.get("b", 0)},
        )

    calculator = AgentTool(
        name="calculator",
        label="calculator",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        execute=execute_calculator,
    )

    # Mock stream that returns a tool call, then after tool result, returns text
    call_count = [0]

    async def _stream_with_tool(m, ctx, opts=None):
        call_count[0] += 1
        partial = AssistantMessage(
            role="assistant", content=[], api=m.api, provider=m.provider,
            model=m.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)

        if call_count[0] == 1:
            # First call: return tool use
            tc = ToolCall(type="toolCall", id="tc1", name="calculator", arguments={"a": 2, "b": 3})
            with_tc = partial.model_copy(update={"content": [tc]})
            from pi_ai.types import EventToolCallEnd, EventToolCallStart
            yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_tc)
            yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tc, partial=with_tc)
            final = AssistantMessage(
                role="assistant", content=[tc], api=m.api, provider=m.provider,
                model=m.id, usage=Usage(), stop_reason="toolUse", timestamp=_ts(),
            )
            yield EventDone(type="done", reason="toolUse", message=final)
        else:
            # Subsequent call: return text
            with_text = partial.model_copy(update={"content": [TextContent(type="text", text="5")]})
            yield EventTextStart(type="text_start", content_index=0, partial=with_text)
            yield EventTextEnd(type="text_end", content_index=0, content="5", partial=with_text)
            final = AssistantMessage(
                role="assistant", content=[TextContent(type="text", text="5")],
                api=m.api, provider=m.provider, model=m.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
            )
            yield EventDone(type="done", reason="stop", message=final)

    context = AgentContext(messages=[], tools=[calculator])
    prompts = [make_user_message("What is 2+3?")]
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")],
    )

    stream = agent_loop(prompts, context, config, stream_fn=_stream_with_tool)
    event_types = []
    async for event in stream:
        event_types.append(event.type)

    assert tool_executed, "Tool should have been called"
    assert tool_executed[0] == {"a": 2, "b": 3}
    assert "tool_execution_start" in event_types
    assert "tool_execution_end" in event_types


@pytest.mark.asyncio
async def test_agent_loop_handles_error_event_payload():
    """EventError uses `error` field (not `message`) and must not crash."""
    from pi_ai import get_model
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")

    async def _stream_error(m, ctx, opts=None):
        partial = AssistantMessage(
            role="assistant",
            content=[],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)
        err_msg = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="")],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="error",
            error_message="boom",
            timestamp=_ts(),
        )
        yield EventError(type="error", reason="error", error=err_msg)

    context = AgentContext(messages=[])
    prompts = [make_user_message("Hello")]
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")],
    )

    stream = agent_loop(prompts, context, config, stream_fn=_stream_error)
    event_types = []
    async for event in stream:
        event_types.append(event.type)

    assert "agent_end" in event_types
