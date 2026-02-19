"""
Tests for AgentSession — mirrors packages/coding-agent/test/ agent session tests.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from typing import AsyncGenerator

import pytest

from pi_ai.types import (
    AssistantMessage,
    EventDone,
    EventStart,
    EventTextEnd,
    EventTextStart,
    TextContent,
    Usage,
    UserMessage,
)
from pi_ai import get_model
from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import Settings
from pi_agent import AgentOptions


def _ts():
    return int(time.time() * 1000)


async def _mock_stream_fn(model, context, opts=None):
    partial = AssistantMessage(
        role="assistant", content=[], api=model.api, provider=model.provider,
        model=model.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
    )
    yield EventStart(type="start", partial=partial)
    text = "I can help you with that!"
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


@pytest.fixture
def session_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def agent_session(session_dir):
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    settings = Settings(auto_compact=False)
    # Use new factory API: create a per-session manager
    session_manager = SessionManager.create(cwd=session_dir, session_dir=session_dir)

    session = AgentSession(
        cwd=session_dir,
        model=model,
        settings=settings,
        session_manager=session_manager,
    )
    # Inject mock stream function
    session._agent.stream_fn = _mock_stream_fn
    return session


@pytest.mark.asyncio
async def test_agent_session_creates_session_id(agent_session):
    assert agent_session.session_id
    # Session IDs are 8-char hex strings (matching TypeScript generate_id)
    assert len(agent_session.session_id) >= 8


@pytest.mark.asyncio
async def test_agent_session_prompt(agent_session):
    await agent_session.prompt("Hello!")

    state = agent_session.state
    assert len(state.messages) >= 2  # user + assistant


@pytest.mark.asyncio
async def test_agent_session_persists_messages(agent_session, session_dir):
    await agent_session.prompt("Hello!")

    # Check that messages were persisted via the session manager attached to the session
    messages = agent_session._session_manager.get_messages()
    assert len(messages) > 0


@pytest.mark.asyncio
async def test_agent_session_subscribe_events(agent_session):
    events = []
    unsub = agent_session.subscribe(events.append)
    await agent_session.prompt("Hello!")
    unsub()

    event_types = [e.type for e in events]
    assert "agent_start" in event_types
    assert "agent_end" in event_types


@pytest.mark.asyncio
async def test_agent_session_set_model(agent_session):
    new_model = get_model("openai", "gpt-4o")
    agent_session.set_model(new_model)
    assert agent_session.state.model.id == "gpt-4o"


@pytest.mark.asyncio
async def test_agent_session_set_thinking_level(agent_session):
    agent_session.set_thinking_level("high")
    assert agent_session.state.thinking_level == "high"

    # Verify it was persisted via the session manager
    entries = agent_session._session_manager.load_entries()
    level_entries = [e for e in entries if e.type == "thinking_level_change"]
    assert any(e.data.get("thinkingLevel") == "high" for e in level_entries)


@pytest.mark.asyncio
async def test_agent_session_fork(agent_session):
    await agent_session.prompt("Hello!")
    original_count = len(agent_session.state.messages)

    forked = agent_session.fork()
    assert forked.session_id != agent_session.session_id
    assert len(forked.state.messages) == original_count


@pytest.mark.asyncio
async def test_agent_session_get_session_info(agent_session):
    info = agent_session.get_session_info()
    assert "session_id" in info
    assert "cwd" in info
    assert "model" in info
    assert "message_count" in info


@pytest.mark.asyncio
async def test_agent_session_abort(agent_session):
    # Abort should not raise if not streaming
    agent_session.abort()
    assert not agent_session.state.is_streaming


def test_default_model_falls_back_when_pinned_model_has_no_auth(monkeypatch, session_dir):
    """
    If settings pin an unauthenticated model/provider (e.g. stale bedrock config),
    AgentSession should fall back to an authenticated default provider.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = Settings(
        auto_compact=False,
        provider="amazon-bedrock",
        model_id="amazon.nova-2-lite-v1:0",
    )
    session_manager = SessionManager.create(cwd=session_dir, session_dir=session_dir)
    session = AgentSession(
        cwd=session_dir,
        settings=settings,
        session_manager=session_manager,
    )
    assert session.model is not None
    assert session.model.provider == "google"
