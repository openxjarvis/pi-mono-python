"""Tests for streaming functions (with mocked providers)."""
from __future__ import annotations

import pytest
from unittest.mock import patch

from pi_ai import (
    Context,
    UserMessage,
    AssistantMessage,
    SimpleStreamOptions,
    EventStart,
    EventTextStart,
    EventTextDelta,
    EventTextEnd,
    EventDone,
    EventError,
    TextContent,
    Usage,
    get_model,
    stream_simple,
    complete_simple,
)


def make_user_message(text: str) -> UserMessage:
    import time
    return UserMessage(role="user", content=text, timestamp=int(time.time() * 1000))


def make_context(text: str) -> Context:
    return Context(messages=[make_user_message(text)])


def make_assistant_message(model, text: str = "Hello!") -> AssistantMessage:
    import time
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


async def mock_stream_simple_fn(model, context, options=None):
    """A mock stream function that yields a simple text response."""
    import time
    partial = AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )
    yield EventStart(type="start", partial=partial)

    partial_with_text = partial.model_copy(update={"content": [TextContent(type="text", text="")]})
    yield EventTextStart(type="text_start", content_index=0, partial=partial_with_text)

    partial_with_delta = partial.model_copy(update={"content": [TextContent(type="text", text="Hello!")]})
    yield EventTextDelta(type="text_delta", content_index=0, delta="Hello!", partial=partial_with_delta)

    yield EventTextEnd(type="text_end", content_index=0, content="Hello!", partial=partial_with_delta)

    final = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="Hello!")],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )
    yield EventDone(type="done", reason="stop", message=final)


@pytest.mark.asyncio
async def test_stream_simple_collects_events():
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    context = make_context("Hello")

    provider = type("P", (), {"stream_simple": staticmethod(mock_stream_simple_fn), "stream": staticmethod(mock_stream_simple_fn)})()
    with patch("pi_ai.stream.get_api_provider", return_value=provider):
        events = []
        async for event in stream_simple(model, context):
            events.append(event)

    assert len(events) > 0
    assert isinstance(events[0], EventStart)
    assert any(isinstance(e, EventDone) for e in events)


@pytest.mark.asyncio
async def test_complete_simple_returns_assistant_message():
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    context = make_context("Hello")

    provider = type("P", (), {"stream_simple": staticmethod(mock_stream_simple_fn), "stream": staticmethod(mock_stream_simple_fn)})()
    with patch("pi_ai.stream.get_api_provider", return_value=provider):
        result = await complete_simple(model, context)

    assert isinstance(result, AssistantMessage)
    assert result.stop_reason == "stop"
    assert any(
        isinstance(b, TextContent) and "Hello" in b.text
        for b in result.content
    )


@pytest.mark.asyncio
async def test_stream_resolves_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    context = make_context("Hello")

    captured_opts = []

    async def capturing_stream(m, ctx, opts=None):
        captured_opts.append(opts)
        async for e in mock_stream_simple_fn(m, ctx, opts):
            yield e

    provider = type("P", (), {"stream_simple": staticmethod(capturing_stream), "stream": staticmethod(capturing_stream)})()
    with patch("pi_ai.stream.get_api_provider", return_value=provider):
        async for _ in stream_simple(model, context):
            pass

    assert captured_opts[0].api_key == "test-key-123"


@pytest.mark.asyncio
async def test_stream_unknown_api_raises():
    from pi_ai.types import Model, ModelCost
    model = Model(
        id="fake-model",
        name="Fake",
        api="unknown-api",
        provider="unknown",
        base_url="https://fake.api",
        cost=ModelCost(),
        context_window=4096,
        max_tokens=1024,
    )
    context = make_context("Hello")

    with pytest.raises(ValueError, match="No stream function"):
        async for _ in stream_simple(model, context):
            pass
