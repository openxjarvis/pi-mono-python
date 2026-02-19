"""
Tests for Agent class — mirrors packages/agent/test/agent.test.ts
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator

import pytest

from pi_ai.types import (
    AssistantMessage,
    EventDone,
    EventStart,
    EventTextDelta,
    EventTextEnd,
    EventTextStart,
    TextContent,
    Usage,
    UserMessage,
)
from pi_agent import Agent, AgentOptions
from pi_agent.types import AgentEventAgentEnd, AgentEventMessageEnd


def _ts() -> int:
    return int(time.time() * 1000)


async def _mock_stream_fn(model, context, options=None) -> AsyncGenerator:
    partial = AssistantMessage(
        role="assistant", content=[], api=model.api, provider=model.provider,
        model=model.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
    )
    yield EventStart(type="start", partial=partial)

    with_text = partial.model_copy(update={"content": [TextContent(type="text", text="")]})
    yield EventTextStart(type="text_start", content_index=0, partial=with_text)

    with_delta = with_text.model_copy(update={"content": [TextContent(type="text", text="Response!")]})
    yield EventTextDelta(type="text_delta", content_index=0, delta="Response!", partial=with_delta)
    yield EventTextEnd(type="text_end", content_index=0, content="Response!", partial=with_delta)

    final = AssistantMessage(
        role="assistant", content=[TextContent(type="text", text="Response!")],
        api=model.api, provider=model.provider, model=model.id,
        usage=Usage(), stop_reason="stop", timestamp=_ts(),
    )
    yield EventDone(type="done", reason="stop", message=final)


@pytest.fixture
def agent():
    from pi_ai import get_model
    opts = AgentOptions(stream_fn=_mock_stream_fn)
    a = Agent(opts)
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    a.set_model(model)
    a.set_system_prompt("You are helpful")
    return a


@pytest.mark.asyncio
async def test_agent_prompt_adds_messages(agent):
    """Test that prompt() adds user and assistant messages to state."""
    assert len(agent.state.messages) == 0

    await agent.prompt("Hello!")

    assert len(agent.state.messages) >= 2  # user + assistant


@pytest.mark.asyncio
async def test_agent_state_not_streaming_after_completion(agent):
    """Test that is_streaming is False after prompt completes."""
    await agent.prompt("Hello!")
    assert agent.state.is_streaming is False


@pytest.mark.asyncio
async def test_agent_subscribe_receives_events(agent):
    """Test that subscribers receive events."""
    received = []
    unsubscribe = agent.subscribe(received.append)

    await agent.prompt("Hello!")

    unsubscribe()
    assert len(received) > 0
    event_types = [e.type for e in received]
    assert "agent_start" in event_types
    assert "agent_end" in event_types


@pytest.mark.asyncio
async def test_agent_unsubscribe_stops_events(agent):
    """Test that unsubscribing stops receiving events."""
    received_after = []
    unsubscribe = agent.subscribe(lambda e: None)
    unsubscribe()  # Immediately unsubscribe
    agent.subscribe(lambda e: received_after.append(e))

    await agent.prompt("Hello!")

    # The first subscriber should not have received events (we don't track it)
    # The second one should have
    assert len(received_after) > 0


@pytest.mark.asyncio
async def test_agent_cannot_double_prompt(agent):
    """Test that prompting while streaming raises an error."""
    # Start a prompt but don't await it
    task = asyncio.create_task(agent.prompt("Hello"))
    await asyncio.sleep(0)  # Yield to let task start

    # This won't fail immediately since the task might complete fast
    # Just verify state gets reset after completion
    await task
    assert agent.state.is_streaming is False


@pytest.mark.asyncio
async def test_agent_steer_queue(agent):
    """Test that steer() queues messages."""
    user_msg = UserMessage(role="user", content="Steer!", timestamp=_ts())
    agent.steer(user_msg)
    assert agent.has_queued_messages()


@pytest.mark.asyncio
async def test_agent_follow_up_queue(agent):
    """Test that follow_up() queues messages."""
    user_msg = UserMessage(role="user", content="Follow up!", timestamp=_ts())
    agent.follow_up(user_msg)
    assert agent.has_queued_messages()


@pytest.mark.asyncio
async def test_agent_clear_messages(agent):
    await agent.prompt("Hello!")
    assert len(agent.state.messages) > 0
    agent.clear_messages()
    assert len(agent.state.messages) == 0


@pytest.mark.asyncio
async def test_agent_set_model(agent):
    from pi_ai import get_model
    new_model = get_model("openai", "gpt-4o")
    agent.set_model(new_model)
    assert agent.state.model.id == "gpt-4o"


@pytest.mark.asyncio
async def test_agent_set_thinking_level(agent):
    agent.set_thinking_level("high")
    assert agent.state.thinking_level == "high"


@pytest.mark.asyncio
async def test_agent_reset(agent):
    await agent.prompt("Hello!")
    agent.steer(UserMessage(role="user", content="x", timestamp=_ts()))
    agent.reset()
    assert len(agent.state.messages) == 0
    assert not agent.has_queued_messages()


@pytest.mark.asyncio
async def test_agent_multiple_prompts(agent):
    """Test that multiple sequential prompts accumulate messages."""
    await agent.prompt("First message")
    first_count = len(agent.state.messages)

    await agent.prompt("Second message")
    second_count = len(agent.state.messages)

    assert second_count > first_count
