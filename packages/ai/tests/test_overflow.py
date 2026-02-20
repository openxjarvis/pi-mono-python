"""
Tests for is_context_overflow — mirrors TS overflow.test.ts logic.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from pi_ai import is_context_overflow, get_overflow_patterns
from pi_ai.types import Usage


def _make_msg(stop_reason: str, error_message: str | None = None, input_tokens: int = 0, cache_read: int = 0):
    msg = MagicMock()
    msg.stop_reason = stop_reason
    msg.error_message = error_message
    usage = Usage(input=input_tokens, output=0, cache_read=cache_read, cache_write=0)
    msg.usage = usage
    return msg


class TestIsContextOverflow:
    def test_returns_false_for_normal_stop(self):
        msg = _make_msg("stop")
        assert is_context_overflow(msg) is False

    def test_returns_false_for_non_overflow_error(self):
        msg = _make_msg("error", "Internal server error")
        assert is_context_overflow(msg) is False

    def test_anthropic_overflow(self):
        msg = _make_msg("error", "prompt is too long: 213462 tokens > 200000 maximum")
        assert is_context_overflow(msg) is True

    def test_bedrock_overflow(self):
        msg = _make_msg("error", "input is too long for requested model")
        assert is_context_overflow(msg) is True

    def test_openai_overflow(self):
        msg = _make_msg("error", "Your input exceeds the context window of this model")
        assert is_context_overflow(msg) is True

    def test_google_gemini_overflow(self):
        msg = _make_msg("error", "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)")
        assert is_context_overflow(msg) is True

    def test_xai_grok_overflow(self):
        msg = _make_msg("error", "This model's maximum prompt length is 131072 but the request contains 537812 tokens")
        assert is_context_overflow(msg) is True

    def test_groq_overflow(self):
        msg = _make_msg("error", "Please reduce the length of the messages or completion")
        assert is_context_overflow(msg) is True

    def test_openrouter_overflow(self):
        msg = _make_msg("error", "This endpoint's maximum context length is 4096 tokens. However, you requested about 8192 tokens")
        assert is_context_overflow(msg) is True

    def test_llama_cpp_overflow(self):
        msg = _make_msg("error", "the request exceeds the available context size, try increasing it")
        assert is_context_overflow(msg) is True

    def test_lm_studio_overflow(self):
        msg = _make_msg("error", "tokens to keep from the initial prompt is greater than the context length")
        assert is_context_overflow(msg) is True

    def test_github_copilot_overflow(self):
        msg = _make_msg("error", "prompt token count of 4321 exceeds the limit of 4096")
        assert is_context_overflow(msg) is True

    def test_minimax_overflow(self):
        msg = _make_msg("error", "invalid params, context window exceeds limit")
        assert is_context_overflow(msg) is True

    def test_kimi_overflow(self):
        msg = _make_msg("error", "Your request exceeded model token limit: 4096 (requested: 8192)")
        assert is_context_overflow(msg) is True

    def test_cerebras_400_no_body(self):
        msg = _make_msg("error", "400 status code (no body)")
        assert is_context_overflow(msg) is True

    def test_cerebras_413_no_body(self):
        msg = _make_msg("error", "413 (no body)")
        assert is_context_overflow(msg) is True

    def test_429_not_overflow(self):
        """429 is rate limiting, NOT context overflow."""
        msg = _make_msg("error", "429 status code (no body)")
        assert is_context_overflow(msg) is False

    def test_generic_too_many_tokens(self):
        msg = _make_msg("error", "too many tokens in request")
        assert is_context_overflow(msg) is True

    def test_generic_token_limit_exceeded(self):
        msg = _make_msg("error", "token limit exceeded")
        assert is_context_overflow(msg) is True

    def test_silent_overflow_z_ai(self):
        """Silent overflow: usage.input > context_window with stop reason."""
        msg = _make_msg("stop", input_tokens=100000)
        assert is_context_overflow(msg, context_window=65536) is True

    def test_no_silent_overflow_within_window(self):
        msg = _make_msg("stop", input_tokens=1000)
        assert is_context_overflow(msg, context_window=65536) is False

    def test_silent_overflow_with_cache_read(self):
        """Cache read tokens count toward overflow detection."""
        msg = _make_msg("stop", input_tokens=30000, cache_read=40000)
        assert is_context_overflow(msg, context_window=65536) is True

    def test_no_context_window_no_silent_check(self):
        msg = _make_msg("stop", input_tokens=999999)
        assert is_context_overflow(msg) is False

    def test_get_overflow_patterns_returns_list(self):
        patterns = get_overflow_patterns()
        assert len(patterns) >= 10
        assert all(hasattr(p, "search") for p in patterns)
