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
    # set_model is now async (validates API key); use _agent.set_model for unit tests
    agent_session._agent.set_model(new_model)
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
    await agent_session.abort()
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


# ── New tests for Group 2 features ────────────────────────────────────────────

class TestSessionPersistenceOnMessageEnd:
    """2a: Per-message persistence using message_end (not agent_end)."""

    @pytest.mark.asyncio
    async def test_messages_persisted_after_prompt(self, agent_session):
        await agent_session.prompt("Hello!")
        msgs = agent_session._session_manager.get_messages()
        # At minimum, user + assistant messages should be saved
        assert len(msgs) >= 1
        roles = [m.get("role") for m in msgs]
        assert "user" in roles or "assistant" in roles


class TestAutoRetryLogic:
    """2b: Auto-retry with exponential backoff."""

    def _make_error_msg(self, error_text: str):
        return AssistantMessage(
            role="assistant", content=[], api="anthropic", provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            usage=Usage(), stop_reason="error",
            error_message=error_text,
            timestamp=_ts(),
        )

    def test_is_retryable_error_rate_limit(self, agent_session):
        msg = self._make_error_msg("rate limit exceeded, retry after 2 seconds")
        assert agent_session._is_retryable_error(msg) is True

    def test_is_retryable_error_overloaded(self, agent_session):
        msg = self._make_error_msg("The API is currently overloaded")
        assert agent_session._is_retryable_error(msg) is True

    def test_is_retryable_error_500(self, agent_session):
        msg = self._make_error_msg("500 internal server error")
        assert agent_session._is_retryable_error(msg) is True

    def test_is_not_retryable_stop(self, agent_session):
        msg = AssistantMessage(
            role="assistant", content=[TextContent(type="text", text="ok")],
            api="anthropic", provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            usage=Usage(), stop_reason="stop", timestamp=_ts(),
        )
        assert agent_session._is_retryable_error(msg) is False

    def test_is_not_retryable_overflow(self, agent_session):
        msg = self._make_error_msg("prompt is too long: 213462 tokens > 200000 maximum")
        # Overflow errors should not be retried (handled by compaction)
        assert agent_session._is_retryable_error(msg) is False

    @pytest.mark.asyncio
    async def test_retry_disabled_when_setting_off(self, agent_session, monkeypatch):
        """When retry is disabled, _handle_retryable_error returns False."""
        monkeypatch.setattr(
            agent_session._settings_manager, "get_retry_settings",
            lambda: {"enabled": False, "maxRetries": 3, "baseDelayMs": 2000}
        )
        msg = self._make_error_msg("rate limit exceeded")
        agent_session._retry_attempt = 0
        result = await agent_session._handle_retryable_error(msg)
        assert result is False

    @pytest.mark.asyncio
    async def test_retry_emits_auto_retry_start_event(self, agent_session, monkeypatch):
        """When retry fires, auto_retry_start event is emitted."""
        emitted = []
        agent_session.subscribe(lambda e: emitted.append(e))

        monkeypatch.setattr(
            agent_session._settings_manager, "get_retry_settings",
            lambda: {"enabled": True, "maxRetries": 3, "baseDelayMs": 10}  # tiny delay
        )

        msg = self._make_error_msg("overloaded")
        agent_session._retry_attempt = 0

        # Patch asyncio.sleep to return immediately
        async def fast_sleep(_): pass
        monkeypatch.setattr(asyncio, "sleep", fast_sleep)

        result = await agent_session._handle_retryable_error(msg)
        assert result is True
        retry_events = [e for e in emitted if (isinstance(e, dict) and e.get("type") == "auto_retry_start")]
        assert len(retry_events) >= 1

    @pytest.mark.asyncio
    async def test_retry_stops_after_max_retries(self, agent_session, monkeypatch):
        """After maxRetries attempts, emits auto_retry_end with success=False."""
        emitted = []
        agent_session.subscribe(lambda e: emitted.append(e))

        monkeypatch.setattr(
            agent_session._settings_manager, "get_retry_settings",
            lambda: {"enabled": True, "maxRetries": 2, "baseDelayMs": 10}
        )
        msg = self._make_error_msg("overloaded")
        # Simulate already at maxRetries
        agent_session._retry_attempt = 2  # will be incremented to 3 > maxRetries=2
        agent_session._retry_event = asyncio.Event()

        result = await agent_session._handle_retryable_error(msg)
        assert result is False
        end_events = [e for e in emitted if isinstance(e, dict) and e.get("type") == "auto_retry_end"]
        assert any(not e.get("success", True) for e in end_events)


class TestToolManagement:
    """2d: set_active_tools_by_name and tool registry."""

    def test_get_all_tool_names(self, agent_session):
        names = agent_session.get_all_tool_names()
        assert "bash" in names
        assert "read" in names
        assert "write" in names
        assert "edit" in names

    def test_get_active_tool_names(self, agent_session):
        names = agent_session.get_active_tool_names()
        assert len(names) > 0

    def test_set_active_tools_by_name(self, agent_session):
        agent_session.set_active_tools_by_name(["bash", "read"])
        active = agent_session.get_active_tool_names()
        assert set(active) == {"bash", "read"}
        # System prompt should be rebuilt to reflect new tool set
        prompt = agent_session.system_prompt
        assert "bash" in prompt

    def test_set_active_tools_ignores_unknown(self, agent_session):
        agent_session.set_active_tools_by_name(["bash", "nonexistent_tool"])
        active = agent_session.get_active_tool_names()
        assert "bash" in active
        assert "nonexistent_tool" not in active


class TestContextUsageAndStats:
    """2e + 2f: get_context_usage and get_session_stats."""

    def test_get_context_usage_returns_none_with_no_model(self, session_dir):
        settings = Settings(auto_compact=False)
        sm = SessionManager.create(cwd=session_dir, session_dir=session_dir)
        sess = AgentSession(
            cwd=session_dir,
            model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
            settings=settings,
            session_manager=sm,
        )
        # Force model to None
        sess._agent._state.model = None
        result = sess.get_context_usage()
        assert result is None

    def test_get_context_usage_returns_dict(self, agent_session):
        result = agent_session.get_context_usage()
        # May return None if no messages yet, or a dict with required keys
        if result is not None:
            assert "tokens" in result
            assert "contextWindow" in result
            assert "percent" in result

    def test_get_session_stats_structure(self, agent_session):
        stats = agent_session.get_session_stats()
        assert "sessionId" in stats
        assert "userMessages" in stats
        assert "assistantMessages" in stats
        assert "toolCalls" in stats
        assert "tokens" in stats
        assert isinstance(stats["tokens"], dict)
        assert "cost" in stats

    @pytest.mark.asyncio
    async def test_session_stats_message_counts(self, agent_session):
        await agent_session.prompt("Hello!")
        stats = agent_session.get_session_stats()
        assert stats["userMessages"] >= 1
        assert stats["assistantMessages"] >= 1


class TestModelCycling:
    """2g: cycle_model and set_model."""

    @pytest.mark.asyncio
    async def test_cycle_model_returns_none_if_single_model(self, agent_session, monkeypatch):
        """If only one model is available, cycle_model returns None."""
        current = agent_session.model

        async def single_available():
            return [current]

        monkeypatch.setattr(agent_session._model_registry, "get_available", single_available)
        result = await agent_session.cycle_model()
        assert result is None

    @pytest.mark.asyncio
    async def test_cycle_model_cycles_forward(self, agent_session, monkeypatch):
        """Cycling forward through multiple models."""
        from pi_ai import get_model as gm
        m1 = gm("anthropic", "claude-3-5-sonnet-20241022")
        m2 = gm("openai", "gpt-4o")

        async def fake_get_available():
            return [m1, m2]

        monkeypatch.setattr(agent_session._model_registry, "get_available", fake_get_available)
        # Set API key validation to always pass
        monkeypatch.setattr(
            agent_session._model_registry, "get_api_key",
            lambda p: "fake-key"
        )

        agent_session._agent.set_model(m1)
        result = await agent_session.cycle_model("forward")
        assert result is not None
        assert agent_session.model.id == m2.id

    @pytest.mark.asyncio
    async def test_set_model_raises_without_api_key(self, agent_session, monkeypatch):
        """set_model raises if no API key is configured for the provider."""
        monkeypatch.setattr(agent_session._model_registry, "get_api_key", lambda p: None)
        new_model = get_model("openai", "gpt-4o")
        with pytest.raises(RuntimeError, match="No API key"):
            await agent_session.set_model(new_model)


class TestThinkingLevelCycling:
    """2h: cycle_thinking_level."""

    def test_get_available_thinking_levels(self, agent_session):
        levels = agent_session.get_available_thinking_levels()
        assert "off" in levels
        assert isinstance(levels, list)

    def test_cycle_thinking_level_advances(self, agent_session, monkeypatch):
        monkeypatch.setattr(
            agent_session, "get_available_thinking_levels",
            lambda: ["off", "minimal", "low", "medium", "high"]
        )
        agent_session._agent.set_thinking_level("off")
        new_level = agent_session.cycle_thinking_level()
        assert new_level == "minimal"

    def test_cycle_thinking_level_wraps_around(self, agent_session, monkeypatch):
        monkeypatch.setattr(
            agent_session, "get_available_thinking_levels",
            lambda: ["off", "minimal"]
        )
        agent_session._agent.set_thinking_level("minimal")
        new_level = agent_session.cycle_thinking_level()
        assert new_level == "off"


class TestQueueManagement:
    """2i + 2j: clear_queue, is_streaming, pending_message_count."""

    def test_is_streaming_false_when_idle(self, agent_session):
        assert agent_session.is_streaming is False

    def test_pending_message_count_zero_initially(self, agent_session):
        assert agent_session.pending_message_count == 0

    def test_clear_queue_returns_dict(self, agent_session):
        result = agent_session.clear_queue()
        assert "steering" in result
        assert "followUp" in result
        assert isinstance(result["steering"], list)
        assert isinstance(result["followUp"], list)

    def test_is_retrying_false_when_idle(self, agent_session):
        assert agent_session.is_retrying is False

    def test_is_compacting_false_when_idle(self, agent_session):
        assert agent_session.is_compacting is False

    def test_retry_attempt_starts_at_zero(self, agent_session):
        assert agent_session.retry_attempt == 0


class TestSessionProperties:
    """Extra session properties added in Group 2."""

    def test_thinking_level_property(self, agent_session):
        level = agent_session.thinking_level
        assert isinstance(level, str)

    def test_system_prompt_property(self, agent_session):
        prompt = agent_session.system_prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_last_assistant_text_none_when_no_messages(self, agent_session):
        result = agent_session.get_last_assistant_text()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_last_assistant_text_after_prompt(self, agent_session):
        await agent_session.prompt("Hello!")
        result = agent_session.get_last_assistant_text()
        # Should return the text of the last assistant message
        assert result is not None
        assert len(result) > 0
