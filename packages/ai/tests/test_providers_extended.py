"""
Tests for extended provider modules (all mocked).

Covers simple_options, google_shared, openai_responses_shared,
and stream function scaffolding for bedrock, vertex, azure, responses, codex.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_model(
    id_: str = "claude-3-5-sonnet-20241022",
    provider: str = "anthropic",
    api: str = "anthropic-messages",
    max_tokens: int = 8096,
    reasoning: bool = False,
    input_types: list[str] | None = None,
):
    m = MagicMock()
    m.id = id_
    m.provider = provider
    m.api = api
    m.max_tokens = max_tokens
    m.reasoning = reasoning
    m.input = input_types or ["text"]
    m.headers = {}
    m.base_url = None
    m.cost = MagicMock(cache_read=0, cache_write=0, input=0, output=0)
    return m


def _make_context(messages=None, system_prompt=None, tools=None):
    ctx = MagicMock()
    ctx.messages = messages or []
    ctx.system_prompt = system_prompt
    ctx.tools = tools or []
    return ctx


# ---------------------------------------------------------------------------
# google_shared
# ---------------------------------------------------------------------------

class TestGoogleShared:
    def test_requires_tool_call_id_for_claude(self):
        from pi_ai.providers.google_shared import requires_tool_call_id
        assert requires_tool_call_id("claude-3-5-sonnet") is True
        assert requires_tool_call_id("gemini-1.5-pro") is False

    def test_is_thinking_part_true(self):
        from pi_ai.providers.google_shared import is_thinking_part
        assert is_thinking_part({"thought": True}) is True
        assert is_thinking_part({"thought": False}) is False
        assert is_thinking_part({}) is False

    def test_retain_thought_signature_incoming_wins(self):
        from pi_ai.providers.google_shared import retain_thought_signature
        assert retain_thought_signature("old", "new_sig") == "new_sig"

    def test_retain_thought_signature_keeps_existing_when_no_incoming(self):
        from pi_ai.providers.google_shared import retain_thought_signature
        assert retain_thought_signature("existing", None) == "existing"
        assert retain_thought_signature("existing", "") == "existing"

    def test_map_stop_reason_stop(self):
        from pi_ai.providers.google_shared import map_stop_reason
        assert map_stop_reason("STOP") == "stop"

    def test_map_stop_reason_max_tokens(self):
        from pi_ai.providers.google_shared import map_stop_reason
        assert map_stop_reason("MAX_TOKENS") == "length"

    def test_map_stop_reason_safety(self):
        from pi_ai.providers.google_shared import map_stop_reason
        assert map_stop_reason("SAFETY") == "error"

    def test_map_stop_reason_string(self):
        from pi_ai.providers.google_shared import map_stop_reason_string
        assert map_stop_reason_string("STOP") == "stop"
        assert map_stop_reason_string("MAX_TOKENS") == "length"
        assert map_stop_reason_string("OTHER") == "error"

    def test_map_tool_choice(self):
        from pi_ai.providers.google_shared import map_tool_choice
        assert map_tool_choice("auto") == "AUTO"
        assert map_tool_choice("none") == "NONE"
        assert map_tool_choice("any") == "ANY"
        assert map_tool_choice("unknown") == "AUTO"

    def test_convert_tools_empty(self):
        from pi_ai.providers.google_shared import convert_tools
        assert convert_tools([]) is None

    def test_convert_tools_basic(self):
        from pi_ai.providers.google_shared import convert_tools
        tool = MagicMock()
        tool.name = "my_tool"
        tool.description = "Does stuff"
        tool.parameters = {"type": "object", "properties": {}}
        result = convert_tools([tool])
        assert result is not None
        assert len(result) == 1
        assert result[0]["functionDeclarations"][0]["name"] == "my_tool"

    def test_convert_tools_use_parameters(self):
        from pi_ai.providers.google_shared import convert_tools
        tool = MagicMock()
        tool.name = "t"
        tool.description = "d"
        tool.parameters = {}
        result = convert_tools([tool], use_parameters=True)
        assert "parameters" in result[0]["functionDeclarations"][0]

    def test_convert_tools_use_parameters_json_schema(self):
        from pi_ai.providers.google_shared import convert_tools
        tool = MagicMock()
        tool.name = "t"
        tool.description = "d"
        tool.parameters = {}
        result = convert_tools([tool], use_parameters=False)
        assert "parametersJsonSchema" in result[0]["functionDeclarations"][0]


# ---------------------------------------------------------------------------
# openai_responses_shared
# ---------------------------------------------------------------------------

class TestOpenAIResponsesShared:
    def test_convert_responses_tools(self):
        from pi_ai.providers.openai_responses_shared import convert_responses_tools
        tool = MagicMock()
        tool.name = "bash"
        tool.description = "Run bash"
        tool.parameters = {"type": "object", "properties": {"cmd": {"type": "string"}}}
        result = convert_responses_tools([tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "bash"

    def test_convert_responses_tools_strict(self):
        from pi_ai.providers.openai_responses_shared import convert_responses_tools
        tool = MagicMock()
        tool.name = "t"
        tool.description = "d"
        tool.parameters = {}
        result = convert_responses_tools([tool], strict=True)
        assert result[0]["strict"] is True


# ---------------------------------------------------------------------------
# simple_options
# ---------------------------------------------------------------------------

class TestSimpleOptionsFull:
    def test_adjust_thinking_high(self):
        from pi_ai.providers.simple_options import adjust_max_tokens_for_thinking
        max_t, budget = adjust_max_tokens_for_thinking(32000, 200000, "high")
        # high budget is 16384
        assert max_t == 32000 + 16384
        assert budget == 16384

    def test_adjust_thinking_xhigh_clamped_to_high(self):
        from pi_ai.providers.simple_options import adjust_max_tokens_for_thinking
        max_t1, budget1 = adjust_max_tokens_for_thinking(32000, 200000, "xhigh")
        max_t2, budget2 = adjust_max_tokens_for_thinking(32000, 200000, "high")
        assert max_t1 == max_t2
        assert budget1 == budget2

    def test_custom_budgets_override_defaults(self):
        from pi_ai.providers.simple_options import adjust_max_tokens_for_thinking
        max_t, budget = adjust_max_tokens_for_thinking(
            32000, 200000, "low", custom_budgets={"low": 5000}
        )
        assert budget == 5000


# ---------------------------------------------------------------------------
# Provider stream functions return EventStream
# ---------------------------------------------------------------------------

class TestProviderStreamReturn:
    """Verify all provider stream functions return EventStreams immediately without blocking."""

    def test_amazon_bedrock_returns_event_stream(self):
        from pi_ai.providers.amazon_bedrock import stream_bedrock
        from pi_ai.utils.event_stream import EventStream

        with patch("asyncio.ensure_future"):
            stream = stream_bedrock(_make_model(), _make_context())
            assert isinstance(stream, EventStream)

    def test_google_vertex_returns_event_stream(self):
        from pi_ai.providers.google_vertex import stream_google_vertex
        from pi_ai.utils.event_stream import EventStream

        with patch("asyncio.ensure_future"):
            stream = stream_google_vertex(_make_model(), _make_context())
            assert isinstance(stream, EventStream)

    def test_openai_responses_returns_event_stream(self):
        from pi_ai.providers.openai_responses import stream_openai_responses
        from pi_ai.utils.event_stream import EventStream

        with patch("asyncio.ensure_future"):
            stream = stream_openai_responses(_make_model(), _make_context())
            assert isinstance(stream, EventStream)

    def test_azure_openai_responses_returns_event_stream(self):
        from pi_ai.providers.azure_openai_responses import stream_azure_openai_responses
        from pi_ai.utils.event_stream import EventStream

        with patch("asyncio.ensure_future"):
            stream = stream_azure_openai_responses(_make_model(), _make_context())
            assert isinstance(stream, EventStream)

    def test_openai_codex_responses_returns_event_stream(self):
        from pi_ai.providers.openai_codex_responses import stream_openai_codex_responses
        from pi_ai.utils.event_stream import EventStream

        with patch("asyncio.ensure_future"):
            stream = stream_openai_codex_responses(_make_model(), _make_context())
            assert isinstance(stream, EventStream)

    def test_google_gemini_cli_returns_event_stream(self):
        from pi_ai.providers.google_gemini_cli import stream_google_gemini_cli
        from pi_ai.utils.event_stream import EventStream

        with patch("asyncio.ensure_future"):
            stream = stream_google_gemini_cli(_make_model(), _make_context())
            assert isinstance(stream, EventStream)


# ---------------------------------------------------------------------------
# register_builtins
# ---------------------------------------------------------------------------

class TestRegisterBuiltins:
    def test_register_builtins_idempotent(self):
        from pi_ai.providers.register_builtins import register_builtins, reset_api_providers
        reset_api_providers()
        register_builtins()
        register_builtins()  # Should not raise or duplicate

    def test_all_core_providers_registered(self):
        from pi_ai.api_registry import get_api_provider
        from pi_ai.providers.register_builtins import register_builtins, reset_api_providers
        reset_api_providers()
        register_builtins()

        for api in ("anthropic-messages", "openai-completions", "google-generative-ai"):
            p = get_api_provider(api)
            assert p is not None, f"Provider {api!r} not registered"
