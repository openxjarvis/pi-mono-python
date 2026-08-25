"""
Context overflow detection utilities.

Detects context window overflow errors from various LLM providers using
regex patterns and usage-based checks.

Mirrors overflow.ts
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pi_ai.types import AssistantMessage

# Regex patterns to detect context overflow errors from different providers.
OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"prompt is too long", re.IGNORECASE),  # Anthropic token overflow
    re.compile(r"request_too_large", re.IGNORECASE),  # Anthropic request byte-size overflow (HTTP 413)
    re.compile(r"input is too long for requested model", re.IGNORECASE),  # Amazon Bedrock
    re.compile(r"exceeds the context window", re.IGNORECASE),  # OpenAI (Completions & Responses)
    re.compile(
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
        re.IGNORECASE,
    ),  # OpenAI-compatible proxies (LiteLLM)
    re.compile(r"input token count.*exceeds the maximum", re.IGNORECASE),  # Google (Gemini)
    re.compile(r"maximum prompt length is \d+", re.IGNORECASE),  # xAI (Grok)
    re.compile(r"reduce the length of the messages", re.IGNORECASE),  # Groq
    re.compile(r"maximum context length is \d+ tokens", re.IGNORECASE),  # OpenRouter
    re.compile(r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?", re.IGNORECASE),  # OpenRouter/Poolside
    re.compile(
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
        re.IGNORECASE,
    ),  # Together AI
    re.compile(r"exceeds the limit of \d+", re.IGNORECASE),  # GitHub Copilot
    re.compile(r"exceeds the available context size", re.IGNORECASE),  # llama.cpp server
    re.compile(r"greater than the context length", re.IGNORECASE),  # LM Studio
    re.compile(r"context window exceeds limit", re.IGNORECASE),  # MiniMax
    re.compile(r"exceeded model token limit", re.IGNORECASE),  # Kimi For Coding
    re.compile(r"too large for model with \d+ maximum context length", re.IGNORECASE),  # Mistral
    re.compile(r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?", re.IGNORECASE),  # DS4
    re.compile(r"model_context_window_exceeded", re.IGNORECASE),  # z.ai
    re.compile(r"prompt too long; exceeded (?:max )?context length", re.IGNORECASE),  # Ollama
    re.compile(r"range of input length should be", re.IGNORECASE),  # DashScope / Qwen
    re.compile(r"context[_ ]length[_ ]exceeded", re.IGNORECASE),  # Generic fallback
    re.compile(r"too many tokens", re.IGNORECASE),  # Generic fallback
    re.compile(r"token limit exceeded", re.IGNORECASE),  # Generic fallback
    re.compile(r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)", re.IGNORECASE),  # Cerebras
]

# Patterns that indicate non-overflow errors (rate limiting, server errors).
NON_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(Throttling error|Service unavailable):", re.IGNORECASE),
    re.compile(r"rate limit", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
]


def is_context_overflow(message: "AssistantMessage", context_window: int | None = None) -> bool:
    """Check if an assistant message represents a context overflow error.

    Handles three cases:
    1. Error-based overflow: stop_reason "error" with a matching error pattern.
    2. Silent overflow: some providers succeed but usage.input exceeds the window.
    3. Length-stop overflow: Xiaomi-style "length" + zero output + filled window.
    """
    if message.stop_reason == "error" and message.error_message:
        is_non_overflow = any(p.search(message.error_message) for p in NON_OVERFLOW_PATTERNS)
        if not is_non_overflow and any(p.search(message.error_message) for p in OVERFLOW_PATTERNS):
            return True

    if context_window and message.stop_reason == "stop":
        input_tokens = message.usage.input + message.usage.cache_read
        if input_tokens > context_window:
            return True

    if context_window and message.stop_reason == "length" and message.usage.output == 0:
        input_tokens = message.usage.input + message.usage.cache_read
        if input_tokens >= context_window * 0.99:
            return True

    return False


def is_recoverable_length(message: "AssistantMessage", desired_max_output: int) -> bool:
    """True when a length stop ended below the intended output limit."""
    return message.stop_reason == "length" and desired_max_output > 0 and message.usage.output < desired_max_output


def get_overflow_patterns() -> list[re.Pattern[str]]:
    """Return the overflow patterns (for testing purposes)."""
    return list(OVERFLOW_PATTERNS)
