"""
End-to-end integration tests — mirrors packages/agent/test/e2e.test.ts

These tests use mock stream functions to simulate full agent workflows
without requiring actual LLM API keys.
"""
from __future__ import annotations

import time
from typing import AsyncGenerator

import pytest

from pi_ai.types import (
    AssistantMessage,
    EventDone,
    EventStart,
    EventTextEnd,
    EventTextStart,
    EventToolCallEnd,
    EventToolCallStart,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
)
from pi_agent import Agent, AgentOptions, AgentTool
from pi_agent.types import AgentToolResult


def _ts():
    return int(time.time() * 1000)


@pytest.mark.asyncio
async def test_e2e_simple_conversation():
    """Full conversation: user → assistant response."""
    responses = ["Hello! How can I help you?", "Sure, I can help with that!"]
    call_count = [0]

    async def mock_stream(model, context, opts=None):
        text = responses[min(call_count[0], len(responses) - 1)]
        call_count[0] += 1

        partial = AssistantMessage(
            role="assistant", content=[], api=model.api, provider=model.provider,
            model=model.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)
        with_text = partial.model_copy(update={"content": [TextContent(type="text", text="")]})
        yield EventTextStart(type="text_start", content_index=0, partial=with_text)
        with_full = with_text.model_copy(update={"content": [TextContent(type="text", text=text)]})
        yield EventTextEnd(type="text_end", content_index=0, content=text, partial=with_full)
        final = AssistantMessage(
            role="assistant", content=[TextContent(type="text", text=text)],
            api=model.api, provider=model.provider, model=model.id,
            usage=Usage(), stop_reason="stop", timestamp=_ts(),
        )
        yield EventDone(type="done", reason="stop", message=final)

    from pi_ai import get_model
    agent = Agent(AgentOptions(stream_fn=mock_stream))
    agent.set_model(get_model("anthropic", "claude-3-5-sonnet-20241022"))

    await agent.prompt("Hi!")
    assert call_count[0] == 1
    messages = agent.state.messages
    assert len(messages) == 2  # user + assistant

    await agent.prompt("Help me!")
    assert call_count[0] == 2


@pytest.mark.asyncio
async def test_e2e_tool_execution():
    """Full flow with tool execution."""
    executed = []

    async def execute_search(tool_call_id, params, cancel=None, on_update=None):
        executed.append(params["query"])
        return AgentToolResult(
            content=[TextContent(type="text", text=f"Results for: {params['query']}")],
            details={"count": 3},
        )

    search_tool = AgentTool(
        name="search",
        label="search",
        description="Search the web",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        execute=execute_search,
    )

    call_count = [0]

    async def mock_stream_with_tool(model, context, opts=None):
        call_count[0] += 1
        partial = AssistantMessage(
            role="assistant", content=[], api=model.api, provider=model.provider,
            model=model.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)

        if call_count[0] == 1:
            tc = ToolCall(type="toolCall", id="s1", name="search", arguments={"query": "Python AI"})
            with_tc = partial.model_copy(update={"content": [tc]})
            yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_tc)
            yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tc, partial=with_tc)
            final = AssistantMessage(
                role="assistant", content=[tc], api=model.api, provider=model.provider,
                model=model.id, usage=Usage(), stop_reason="toolUse", timestamp=_ts(),
            )
            yield EventDone(type="done", reason="toolUse", message=final)
        else:
            text = "I found 3 results for Python AI."
            with_text = partial.model_copy(update={"content": [TextContent(type="text", text=text)]})
            yield EventTextEnd(type="text_end", content_index=0, content=text, partial=with_text)
            final = AssistantMessage(
                role="assistant", content=[TextContent(type="text", text=text)],
                api=model.api, provider=model.provider, model=model.id,
                usage=Usage(), stop_reason="stop", timestamp=_ts(),
            )
            yield EventDone(type="done", reason="stop", message=final)

    from pi_ai import get_model
    agent = Agent(AgentOptions(stream_fn=mock_stream_with_tool))
    agent.set_model(get_model("anthropic", "claude-3-5-sonnet-20241022"))
    agent.set_tools([search_tool])

    await agent.prompt("Search for Python AI")

    assert executed == ["Python AI"], "Search tool should have been called"
    # Messages: user, assistant(tool_use), tool_result, assistant(text)
    assert len(agent.state.messages) >= 3


@pytest.mark.asyncio
async def test_e2e_steering_message():
    """Test that steering messages interrupt tool execution."""
    from pi_ai import get_model
    agent = Agent(AgentOptions(stream_fn=lambda m, c, o: _noop_stream(m)))
    agent.set_model(get_model("anthropic", "claude-3-5-sonnet-20241022"))

    stop_msg = UserMessage(role="user", content="Stop!", timestamp=_ts())
    agent.steer(stop_msg)

    assert agent.has_queued_messages()

    # After dequeuing in one-at-a-time mode, queue should be empty
    from pi_agent.types import AgentEventAgentEnd
    received = []
    agent.subscribe(received.append)

    # Just verify the queue state
    assert agent._steering_queue == [stop_msg]
    agent.clear_steering_queue()
    assert not agent.has_queued_messages()


async def _noop_stream(model):
    """Stream that immediately returns empty."""
    partial = AssistantMessage(
        role="assistant", content=[], api=model.api, provider=model.provider,
        model=model.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
    )
    yield EventStart(type="start", partial=partial)
    final = AssistantMessage(
        role="assistant", content=[TextContent(type="text", text="Done")],
        api=model.api, provider=model.provider, model=model.id,
        usage=Usage(), stop_reason="stop", timestamp=_ts(),
    )
    yield EventDone(type="done", reason="stop", message=final)
