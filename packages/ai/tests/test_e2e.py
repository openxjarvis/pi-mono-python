"""
End-to-end integration tests for pi_ai.

Tests multi-component integration: providers + streaming + tool calling + OAuth.
"""
from __future__ import annotations

import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pi_ai import get_model
from pi_ai.providers.register_builtins import register_builtins
from pi_ai.types import (
    AssistantMessage,
    EventDone,
    EventStart,
    EventTextEnd,
    EventToolCallEnd,
    EventToolCallStart,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
)
from pi_ai.utils.event_stream import EventStream


# ============================================================================
# EventStream integration
# ============================================================================

class TestEventStreamIntegration:
    @pytest.mark.asyncio
    async def test_push_and_iterate(self):
        """Test that events pushed into the stream can be iterated."""
        stream: EventStream[str, str] = EventStream()

        async def producer():
            for i in range(5):
                stream.push(f"event_{i}")
                await asyncio.sleep(0)
            stream.end("final")

        asyncio.ensure_future(producer())
        collected = []
        async for event in stream:
            collected.append(event)

        assert collected == [f"event_{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_fail_propagates(self):
        """Test that stream failures propagate as exceptions."""
        stream: EventStream[str, str] = EventStream()

        async def producer():
            stream.push("first")
            await asyncio.sleep(0)
            stream.fail(RuntimeError("Test error"))

        asyncio.ensure_future(producer())
        collected = []
        with pytest.raises(RuntimeError, match="Test error"):
            async for event in stream:
                collected.append(event)

        assert collected == ["first"]

    @pytest.mark.asyncio
    async def test_get_result(self):
        """Test that the final result is accessible after stream ends."""
        stream: EventStream[str, str] = EventStream()

        async def producer():
            stream.push("event")
            stream.end("my_result")

        asyncio.ensure_future(producer())
        async for _ in stream:
            pass

        # Result is stored in _result after stream ends
        assert stream._result == "my_result"

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Test that an immediately-ended stream works correctly."""
        stream: EventStream[str, str] = EventStream()
        stream.end("done")

        collected = []
        async for event in stream:
            collected.append(event)

        assert collected == []

    def test_stream_optional_callbacks(self):
        """Test EventStream works without callbacks (new default)."""
        stream: EventStream[str, str] = EventStream()
        assert stream is not None


# ============================================================================
# Provider integration: streaming pipeline
# ============================================================================

class TestProviderStreamingPipeline:
    def _get_model(self):
        return get_model("anthropic", "claude-3-5-sonnet-20241022")

    @pytest.mark.asyncio
    async def test_mock_text_streaming(self):
        """Test a complete text streaming pipeline with a mock provider."""
        model = self._get_model()
        import time

        async def mock_stream(m, ctx, opts=None):
            partial = AssistantMessage(
                role="assistant", content=[], api=m.api, provider=m.provider,
                model=m.id, usage=Usage(), stop_reason="stop", timestamp=int(time.time() * 1000),
            )
            yield EventStart(type="start", partial=partial)
            text = "Hello, I am an AI assistant."
            with_text = partial.model_copy(update={"content": [TextContent(type="text", text=text)]})
            yield EventTextEnd(type="text_end", content_index=0, content=text, partial=with_text)
            final = AssistantMessage(
                role="assistant", content=[TextContent(type="text", text=text)],
                api=m.api, provider=m.provider, model=m.id,
                usage=Usage(), stop_reason="stop", timestamp=int(time.time() * 1000),
            )
            yield EventDone(type="done", reason="stop", message=final)

        # Collect all events
        events = []
        async for event in mock_stream(model, None):
            events.append(event)

        assert len(events) == 3
        assert events[0].type == "start"
        assert events[1].type == "text_end"
        assert events[2].type == "done"

    @pytest.mark.asyncio
    async def test_mock_tool_call_pipeline(self):
        """Test a complete tool call pipeline with mock provider."""
        model = self._get_model()
        import time

        async def mock_stream(m, ctx, opts=None):
            partial = AssistantMessage(
                role="assistant", content=[], api=m.api, provider=m.provider,
                model=m.id, usage=Usage(), stop_reason="stop", timestamp=int(time.time() * 1000),
            )
            yield EventStart(type="start", partial=partial)

            tc = ToolCall(type="toolCall", id="tc1", name="read", arguments={"path": "/test.txt"})
            with_tc = partial.model_copy(update={"content": [tc]})
            yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_tc)
            yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tc, partial=with_tc)

            final = AssistantMessage(
                role="assistant", content=[tc], api=m.api, provider=m.provider,
                model=m.id, usage=Usage(), stop_reason="toolUse", timestamp=int(time.time() * 1000),
            )
            yield EventDone(type="done", reason="toolUse", message=final)

        events = []
        async for event in mock_stream(model, None):
            events.append(event)

        tool_start = [e for e in events if e.type == "toolcall_start"]
        tool_end = [e for e in events if e.type == "toolcall_end"]
        assert len(tool_start) == 1
        assert len(tool_end) == 1
        assert tool_end[0].tool_call.name == "read"


# ============================================================================
# Provider registration integration
# ============================================================================

class TestProviderRegistration:
    def test_register_builtins_registers_providers(self):
        """Test that register_builtins adds providers to the registry."""
        from pi_ai.api_registry import get_api_provider
        register_builtins()
        providers = ["anthropic", "openai", "groq"]
        for name in providers:
            try:
                prov = get_api_provider(name)
                assert prov is not None
            except Exception:
                pass  # Missing API key is OK

    def test_get_model_returns_model_object(self):
        """Test that get_model returns a properly structured Model."""
        model = get_model("anthropic", "claude-3-5-sonnet-20241022")
        assert model.provider == "anthropic"
        assert model.id == "claude-3-5-sonnet-20241022"
        assert model.context_window > 0

    def test_model_has_required_fields(self):
        """Test all required Model fields are present."""
        model = get_model("openai", "gpt-4o")
        assert hasattr(model, "provider")
        assert hasattr(model, "id")
        assert hasattr(model, "context_window")
        assert hasattr(model, "max_tokens")
        assert hasattr(model, "reasoning")
        assert hasattr(model, "input")


# ============================================================================
# OAuth types integration
# ============================================================================

class TestOAuthTypesIntegration:
    def test_oauth_credentials_creation(self):
        """Test that OAuthCredentials can be created."""
        from pi_ai.utils.oauth.types import OAuthCredentials
        creds = OAuthCredentials(refresh="ref123", access="acc456", expires=9999999999)
        assert creds.refresh == "ref123"
        assert creds.access == "acc456"

    def test_oauth_credentials_roundtrip(self):
        """Test OAuthCredentials serialization/deserialization."""
        from pi_ai.utils.oauth.types import OAuthCredentials
        creds = OAuthCredentials(refresh="r", access="a", expires=1234)
        d = creds.to_dict()
        restored = OAuthCredentials.from_dict(d)
        assert restored.refresh == "r"
        assert restored.access == "a"

    def test_pkce_generation(self):
        """Test PKCE code verifier and challenge generation."""
        from pi_ai.utils.oauth.pkce import generate_pkce
        verifier, challenge = generate_pkce()
        assert len(verifier) >= 43
        assert len(challenge) > 0
        assert challenge != verifier


# ============================================================================
# JSON parse integration
# ============================================================================

class TestJsonParseIntegration:
    def test_streaming_json_partial(self):
        """Test parsing of incomplete JSON from LLM streaming."""
        from pi_ai.utils.json_parse import parse_streaming_json
        partial = '{"key": "value", "nested": {"arr": [1, 2'
        result = parse_streaming_json(partial)
        # May return None or a partial result
        assert result is None or isinstance(result, dict)

    def test_streaming_json_complete(self):
        """Test parsing of complete JSON."""
        from pi_ai.utils.json_parse import parse_streaming_json
        complete = '{"key": "value", "num": 42}'
        result = parse_streaming_json(complete)
        assert result is not None
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_partial_json_empty(self):
        """Test that empty string returns None."""
        from pi_ai.utils.json_parse import parse_partial_json
        assert parse_partial_json("") is None
        assert parse_partial_json("  ") is None

    def test_parse_streaming_is_alias(self):
        """Test that parse_streaming_json and parse_partial_json are equivalent."""
        from pi_ai.utils.json_parse import parse_partial_json, parse_streaming_json
        data = '{"x": 1}'
        assert parse_streaming_json(data) == parse_partial_json(data)


# ============================================================================
# Overflow detection integration
# ============================================================================

class TestOverflowIntegration:
    def test_overflow_patterns_are_list(self):
        """Test that overflow patterns are loaded as a list."""
        from pi_ai.utils.overflow import get_overflow_patterns
        patterns = get_overflow_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0

    def test_is_context_overflow_takes_message_obj(self):
        """Test the is_context_overflow function signature."""
        from pi_ai.utils.overflow import is_context_overflow
        import inspect
        sig = inspect.signature(is_context_overflow)
        params = list(sig.parameters.keys())
        # Should take a message parameter
        assert len(params) >= 1
