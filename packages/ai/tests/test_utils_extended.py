"""
Tests for extended utility modules: sanitize_unicode, overflow, http_proxy.

Mirrors the test coverage intent from the plan.
"""

from __future__ import annotations

import os
import re
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# sanitize_unicode
# ---------------------------------------------------------------------------

class TestSanitizeUnicode:
    def test_normal_text_preserved(self):
        from pi_ai.utils.sanitize_unicode import sanitize_surrogates
        assert sanitize_surrogates("Hello World") == "Hello World"

    def test_empty_string(self):
        from pi_ai.utils.sanitize_unicode import sanitize_surrogates
        assert sanitize_surrogates("") == ""

    def test_emoji_preserved(self):
        from pi_ai.utils.sanitize_unicode import sanitize_surrogates
        # Valid emoji uses proper surrogate pairs; should be preserved
        text = "Hello 🙈 World"
        assert sanitize_surrogates(text) == text

    def test_unicode_text_preserved(self):
        from pi_ai.utils.sanitize_unicode import sanitize_surrogates
        text = "こんにちは 你好 مرحبا"
        assert sanitize_surrogates(text) == text

    def test_removes_unpaired_high_surrogate(self):
        from pi_ai.utils.sanitize_unicode import sanitize_surrogates
        # High surrogate without low surrogate
        unpaired = "\uD83D"  # High surrogate without low
        text = f"Text {unpaired} here"
        result = sanitize_surrogates(text)
        assert "\uD83D" not in result
        assert "Text" in result
        assert "here" in result

    def test_multiline_text_preserved(self):
        from pi_ai.utils.sanitize_unicode import sanitize_surrogates
        text = "Line 1\nLine 2\nLine 3"
        assert sanitize_surrogates(text) == text

    def test_json_safe_output(self):
        from pi_ai.utils.sanitize_unicode import sanitize_surrogates
        import json
        # Should be JSON-serializable after sanitization
        text = "Hello \uD800 World"  # Unpaired high surrogate
        sanitized = sanitize_surrogates(text)
        # Should not raise
        json.dumps(sanitized)


# ---------------------------------------------------------------------------
# overflow
# ---------------------------------------------------------------------------

class TestOverflow:
    def _make_msg(self, stop_reason: str, error_message: str | None = None, usage_input: int = 0, usage_cache_read: int = 0):
        msg = MagicMock()
        msg.stop_reason = stop_reason
        msg.error_message = error_message
        msg.usage = MagicMock()
        msg.usage.input = usage_input
        msg.usage.cache_read = usage_cache_read
        return msg

    def test_anthropic_overflow(self):
        from pi_ai.utils.overflow import is_context_overflow
        msg = self._make_msg("error", "prompt is too long: 213462 tokens > 200000 maximum")
        assert is_context_overflow(msg) is True

    def test_openai_overflow(self):
        from pi_ai.utils.overflow import is_context_overflow
        msg = self._make_msg("error", "Your input exceeds the context window of this model")
        assert is_context_overflow(msg) is True

    def test_google_overflow(self):
        from pi_ai.utils.overflow import is_context_overflow
        msg = self._make_msg("error", "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)")
        assert is_context_overflow(msg) is True

    def test_bedrock_overflow(self):
        from pi_ai.utils.overflow import is_context_overflow
        msg = self._make_msg("error", "input is too long for requested model")
        assert is_context_overflow(msg) is True

    def test_cerebras_413_overflow(self):
        from pi_ai.utils.overflow import is_context_overflow
        msg = self._make_msg("error", "413 status code (no body)")
        assert is_context_overflow(msg) is True

    def test_no_overflow_normal(self):
        from pi_ai.utils.overflow import is_context_overflow
        msg = self._make_msg("stop")
        assert is_context_overflow(msg) is False

    def test_no_overflow_rate_limit(self):
        from pi_ai.utils.overflow import is_context_overflow
        msg = self._make_msg("error", "Rate limit exceeded (429)")
        assert is_context_overflow(msg) is False

    def test_silent_overflow(self):
        from pi_ai.utils.overflow import is_context_overflow
        msg = self._make_msg("stop", usage_input=200001)
        assert is_context_overflow(msg, context_window=200000) is True

    def test_no_silent_overflow_within_limit(self):
        from pi_ai.utils.overflow import is_context_overflow
        msg = self._make_msg("stop", usage_input=100000)
        assert is_context_overflow(msg, context_window=200000) is False

    def test_get_overflow_patterns(self):
        from pi_ai.utils.overflow import get_overflow_patterns
        patterns = get_overflow_patterns()
        assert len(patterns) > 10
        assert all(isinstance(p, re.Pattern) for p in patterns)


# ---------------------------------------------------------------------------
# http_proxy
# ---------------------------------------------------------------------------

class TestHttpProxy:
    def test_no_proxy_env(self):
        from pi_ai.utils.http_proxy import get_proxy_url
        with patch.dict(os.environ, {}, clear=True):
            # Remove proxy vars if present
            for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                os.environ.pop(k, None)
            assert get_proxy_url() is None

    def test_https_proxy_env(self):
        from pi_ai.utils.http_proxy import get_proxy_url
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://proxy.example.com:8080"}, clear=False):
            assert get_proxy_url() == "http://proxy.example.com:8080"

    def test_http_proxy_fallback(self):
        from pi_ai.utils.http_proxy import get_proxy_url
        env = {"HTTP_PROXY": "http://proxy2.example.com:3128"}
        # Remove HTTPS_PROXY if set
        env_patch = {k: v for k, v in env.items()}
        for k in ("HTTPS_PROXY",):
            os.environ.pop(k, None)
        with patch.dict(os.environ, env_patch, clear=False):
            url = get_proxy_url()
            assert url is not None

    def test_get_proxies_none_when_no_proxy(self):
        from pi_ai.utils.http_proxy import get_proxies
        for k in ("HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(k, None)
        with patch.dict(os.environ, {}, clear=False):
            for k in ("HTTP_PROXY", "HTTPS_PROXY"):
                os.environ.pop(k, None)
            assert get_proxies() is None

    def test_get_proxies_dict_when_proxy_set(self):
        from pi_ai.utils.http_proxy import get_proxies
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://proxy.test:8080"}):
            proxies = get_proxies()
            assert proxies is not None
            assert "https://" in proxies or "http://" in proxies


# ---------------------------------------------------------------------------
# simple_options
# ---------------------------------------------------------------------------

class TestSimpleOptions:
    def _make_model(self, max_tokens: int = 100000):
        m = MagicMock()
        m.max_tokens = max_tokens
        return m

    def test_clamp_reasoning_xhigh(self):
        from pi_ai.providers.simple_options import clamp_reasoning
        assert clamp_reasoning("xhigh") == "high"

    def test_clamp_reasoning_passthrough(self):
        from pi_ai.providers.simple_options import clamp_reasoning
        for level in ("low", "medium", "high", "minimal"):
            assert clamp_reasoning(level) == level

    def test_clamp_reasoning_none(self):
        from pi_ai.providers.simple_options import clamp_reasoning
        assert clamp_reasoning(None) is None

    def test_adjust_max_tokens_for_thinking_basic(self):
        from pi_ai.providers.simple_options import adjust_max_tokens_for_thinking
        max_t, budget = adjust_max_tokens_for_thinking(32000, 100000, "low")
        assert max_t == 32000 + 2048  # low budget is 2048
        assert budget == 2048

    def test_adjust_max_tokens_caps_at_model_max(self):
        from pi_ai.providers.simple_options import adjust_max_tokens_for_thinking
        max_t, budget = adjust_max_tokens_for_thinking(32000, 33000, "high")
        # Would be 32000 + 16384 = 48384, but capped at model max 33000
        assert max_t == 33000

    def test_build_base_options(self):
        from pi_ai.providers.simple_options import build_base_options
        model = self._make_model()
        opts = build_base_options(model)
        assert opts.max_tokens <= 32000


# ---------------------------------------------------------------------------
# github_copilot_headers
# ---------------------------------------------------------------------------

class TestGithubCopilotHeaders:
    def _msg(self, role: str, content=None):
        m = MagicMock()
        m.role = role
        m.content = content or []
        return m

    def test_infer_initiator_user(self):
        from pi_ai.providers.github_copilot_headers import infer_copilot_initiator
        msgs = [self._msg("user")]
        assert infer_copilot_initiator(msgs) == "user"

    def test_infer_initiator_agent(self):
        from pi_ai.providers.github_copilot_headers import infer_copilot_initiator
        msgs = [self._msg("assistant")]
        assert infer_copilot_initiator(msgs) == "agent"

    def test_infer_initiator_empty(self):
        from pi_ai.providers.github_copilot_headers import infer_copilot_initiator
        assert infer_copilot_initiator([]) == "user"

    def test_has_vision_input_false(self):
        from pi_ai.providers.github_copilot_headers import has_copilot_vision_input
        msgs = [self._msg("user", [MagicMock(type="text")])]
        assert has_copilot_vision_input(msgs) is False

    def test_has_vision_input_true(self):
        from pi_ai.providers.github_copilot_headers import has_copilot_vision_input
        image_content = MagicMock()
        image_content.type = "image"
        msgs = [self._msg("user", [image_content])]
        assert has_copilot_vision_input(msgs) is True

    def test_build_headers_no_images(self):
        from pi_ai.providers.github_copilot_headers import build_copilot_dynamic_headers
        msgs = [self._msg("user")]
        headers = build_copilot_dynamic_headers(msgs, has_images=False)
        assert "X-Initiator" in headers
        assert "Openai-Intent" in headers
        assert "Copilot-Vision-Request" not in headers

    def test_build_headers_with_images(self):
        from pi_ai.providers.github_copilot_headers import build_copilot_dynamic_headers
        msgs = [self._msg("user")]
        headers = build_copilot_dynamic_headers(msgs, has_images=True)
        assert headers.get("Copilot-Vision-Request") == "true"
