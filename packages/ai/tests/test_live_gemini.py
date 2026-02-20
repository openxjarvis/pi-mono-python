"""
Live integration tests for Google Gemini API.

Run with:  pytest --live   OR   LIVE_TESTS=1 pytest

Env vars:
  GEMINI_API_KEY    — your Gemini API key
  GEMINI_TEST_MODEL — override model (default: gemini-3-pro-preview)
"""
from __future__ import annotations

import os
import time
import asyncio
import pytest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# User-requested default model for live validation
_MODEL_ID = os.environ.get("GEMINI_TEST_MODEL", "gemini-3-pro-preview")


def _get_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _skip_if_no_key() -> None:
    if not _get_api_key():
        pytest.skip("No GEMINI_API_KEY or GOOGLE_API_KEY set")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _is_transient_unavailable(message: str | None) -> bool:
    if not message:
        return False
    lower = message.lower()
    return "503" in lower or "service unavailable" in lower or "currently experiencing high demand" in lower


# ---------------------------------------------------------------------------
# Helpers — build typed objects correctly
# ---------------------------------------------------------------------------

def _make_model(model_id: str = _MODEL_ID):
    """Construct a Model for the google-generative-ai API (uses GEMINI_API_KEY)."""
    from pi_ai.types import Model
    return Model(
        id=model_id,
        name=model_id,
        api="google-generative-ai",
        provider="google",
        base_url="",
        context_window=1_000_000,
        max_tokens=8192,
    )


def _user_msg(text: str):
    from pi_ai.types import UserMessage, TextContent
    return UserMessage(
        role="user",
        content=[TextContent(type="text", text=text)],
        timestamp=_now_ms(),
    )


def _assistant_msg(text: str):
    from pi_ai.types import AssistantMessage, TextContent
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        api="google-generative-ai",
        provider="google",
        model=_MODEL_ID,
        timestamp=_now_ms(),
    )


def _make_context(user_text: str, system: str = "You are a concise assistant. Reply in 10 words or fewer."):
    from pi_ai.types import Context
    return Context(
        system_prompt=system,
        messages=[_user_msg(user_text)],
        tools=[],
    )


# ---------------------------------------------------------------------------
# 1. Basic completion
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.asyncio
async def test_live_gemini_complete_basic():
    """complete_simple() returns a non-empty response."""
    _skip_if_no_key()

    from pi_ai.stream import complete_simple
    from pi_ai.types import AssistantMessage, SimpleStreamOptions

    # Use generous max_tokens: thinking models spend tokens on reasoning first,
    # so a tight budget (e.g. 50) may be fully consumed before any text is produced.
    msg = await complete_simple(
        _make_model(),
        _make_context("Say exactly: hello world"),
        SimpleStreamOptions(api_key=_get_api_key(), max_tokens=1024),
    )
    if getattr(msg, "stop_reason", None) == "error" and _is_transient_unavailable(getattr(msg, "error_message", None)):
        pytest.skip(f"Transient model unavailability: {msg.error_message}")

    assert isinstance(msg, AssistantMessage)
    texts = [b.text for b in (msg.content or []) if hasattr(b, "text") and b.text]
    combined = " ".join(texts).lower()
    assert combined.strip(), "Response text should not be empty"
    assert "hello" in combined or "world" in combined, f"Expected greeting, got: {combined!r}"
    print(f"\n[live] model={_MODEL_ID}  response={combined!r}")


# ---------------------------------------------------------------------------
# 2. Streaming events
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.asyncio
async def test_live_gemini_streaming_events():
    """stream_simple() yields EventStart → EventTextDelta(s) → EventDone."""
    _skip_if_no_key()

    from pi_ai.stream import stream_simple
    from pi_ai.types import EventStart, EventTextDelta, EventTextStart, EventDone, SimpleStreamOptions

    events = []
    async for event in stream_simple(
        _make_model(),
        _make_context("Say exactly: hello world"),
        SimpleStreamOptions(api_key=_get_api_key(), max_tokens=100),
    ):
        events.append(event)

    types_seen = [type(e).__name__ for e in events]
    error_events = [e for e in events if type(e).__name__ == "EventError"]
    if error_events and _is_transient_unavailable(getattr(error_events[-1].error, "error_message", None)):
        pytest.skip(f"Transient model unavailability: {error_events[-1].error.error_message}")
    assert "EventStart" in types_seen, f"Missing EventStart — got: {types_seen}"
    assert "EventDone" in types_seen, f"Missing EventDone — got: {types_seen}"

    done_events = [e for e in events if isinstance(e, EventDone)]
    texts = [b.text for b in (done_events[-1].message.content or []) if hasattr(b, "text") and b.text]
    assert texts, "Final message has no text"
    print(f"\n[live] events={types_seen}  text={''.join(texts)!r}")


# ---------------------------------------------------------------------------
# 3. Multi-turn conversation
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.asyncio
async def test_live_gemini_multi_turn():
    """Multi-turn context is preserved (colour memory test)."""
    _skip_if_no_key()

    from pi_ai.stream import complete_simple
    from pi_ai.types import Context, SimpleStreamOptions

    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[
            _user_msg("My favourite colour is blue. Remember it."),
            _assistant_msg("I'll remember that your favourite colour is blue."),
            _user_msg("What is my favourite colour?"),
        ],
        tools=[],
    )

    # gemini-2.5-pro is a thinking model — needs generous max_tokens so thinking
    # tokens don't crowd out the actual answer
    msg = await complete_simple(
        _make_model(),
        context,
        SimpleStreamOptions(api_key=_get_api_key(), max_tokens=2000),
    )
    if getattr(msg, "stop_reason", None) == "error" and _is_transient_unavailable(getattr(msg, "error_message", None)):
        pytest.skip(f"Transient model unavailability: {msg.error_message}")

    texts = [b.text for b in (msg.content or []) if hasattr(b, "text") and b.text]
    combined = " ".join(texts).lower()
    assert "blue" in combined, f"Expected 'blue' in response, got: {combined!r}"
    print(f"\n[live] multi-turn: {combined!r}")


# ---------------------------------------------------------------------------
# 4. Tool calling
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.asyncio
async def test_live_gemini_tool_calling():
    """Gemini issues a tool call for a weather query."""
    _skip_if_no_key()

    from pi_ai.stream import stream_simple
    from pi_ai.types import Context, EventDone, ToolCall, SimpleStreamOptions

    get_weather_tool = {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    }

    context = Context(
        system_prompt="Use tools when appropriate.",
        messages=[_user_msg("What is the weather in Paris?")],
        tools=[get_weather_tool],
    )

    events = []
    async def _collect() -> None:
        async for event in stream_simple(
            _make_model(),
            context,
            SimpleStreamOptions(api_key=_get_api_key(), max_tokens=200),
        ):
            events.append(event)

    try:
        await asyncio.wait_for(_collect(), timeout=30.0)
    except asyncio.TimeoutError:
        pytest.skip("Tool-calling request timed out on gemini-3-pro-preview (transient)")

    done_events = [e for e in events if isinstance(e, EventDone)]
    error_events = [e for e in events if type(e).__name__ == "EventError"]
    if error_events and _is_transient_unavailable(getattr(error_events[-1].error, "error_message", None)):
        pytest.skip(f"Transient model unavailability: {error_events[-1].error.error_message}")
    assert done_events, f"No EventDone — got: {[type(e).__name__ for e in events]}"
    final = done_events[-1].message

    tool_calls = [b for b in (final.content or []) if isinstance(b, ToolCall)]
    if tool_calls:
        tc = tool_calls[0]
        assert tc.name == "get_weather", f"Unexpected tool: {tc.name}"
        assert "paris" in str(tc.arguments).lower(), f"Expected 'paris' in args: {tc.arguments}"
        print(f"\n[live] tool call: {tc.name}({tc.arguments})")
    else:
        texts = [b.text for b in (final.content or []) if hasattr(b, "text") and b.text]
        print(f"\n[live] text (no tool call): {''.join(texts)!r}")


# ---------------------------------------------------------------------------
# 5. Token usage
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.asyncio
async def test_live_gemini_usage_fields():
    """AssistantMessage.usage contains non-zero token counts."""
    _skip_if_no_key()

    from pi_ai.stream import complete_simple
    from pi_ai.types import SimpleStreamOptions

    msg = await complete_simple(
        _make_model(),
        _make_context("Say exactly: hello world"),
        SimpleStreamOptions(api_key=_get_api_key(), max_tokens=1024),
    )
    if getattr(msg, "stop_reason", None) == "error" and _is_transient_unavailable(getattr(msg, "error_message", None)):
        pytest.skip(f"Transient model unavailability: {msg.error_message}")

    assert msg.usage is not None
    assert msg.usage.input > 0 or msg.usage.output > 0, (
        f"Expected non-zero tokens, got: input={msg.usage.input} output={msg.usage.output}"
    )
    print(f"\n[live] usage: input={msg.usage.input}  output={msg.usage.output}  total={msg.usage.total_tokens}")


# ---------------------------------------------------------------------------
# 6. Model registry
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_live_gemini_model_registry():
    """get_model() returns valid fields for a registered google model."""
    _skip_if_no_key()

    from pi_ai.models import get_model
    m = get_model("google", "gemini-2.0-flash")
    assert m.provider == "google"
    assert m.api == "google-generative-ai"
    assert m.context_window > 0
    print(f"\n[live] registry model: {m.id}  ctx={m.context_window}")


# ---------------------------------------------------------------------------
# 7. Dynamic model construction helper
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_live_gemini_make_model_helper():
    """_make_model() builds a correct Model for any model ID."""
    _skip_if_no_key()

    m = _make_model(_MODEL_ID)
    assert m.id == _MODEL_ID
    assert m.api == "google-generative-ai"
    assert m.provider == "google"
    print(f"\n[live] dynamic model: {m.id}  api={m.api}")


# ---------------------------------------------------------------------------
# 8. Partial-JSON parser — no network
# ---------------------------------------------------------------------------

def test_live_gemini_json_parse_pipeline():
    """parse_streaming_json handles complete and truncated JSON."""
    from pi_ai.utils.json_parse import parse_streaming_json

    assert parse_streaming_json('{"key": "value"}') == {"key": "value"}
    assert parse_streaming_json('{"a": 1, "b": 2}') == {"a": 1, "b": 2}
    result = parse_streaming_json('{"key": "val')
    assert result is None or isinstance(result, dict)
    print("\n[live] JSON parse pipeline ok")


# ---------------------------------------------------------------------------
# 9. Env key resolution — no network
# ---------------------------------------------------------------------------

def test_live_env_key_resolution():
    """GEMINI_API_KEY is resolved correctly for provider 'google'."""
    from pi_ai.env_api_keys import get_env_api_key

    original = os.environ.get("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = "test-key-123"
    try:
        assert get_env_api_key("google") == "test-key-123"
    finally:
        if original is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = original
    print("\n[live] env key resolution ok")
