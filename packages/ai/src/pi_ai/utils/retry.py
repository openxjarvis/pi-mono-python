"""
Assistant-call retry policy and classifier.
Mirrors packages/ai/src/utils/retry.ts
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pi_ai.types import AssistantMessage


def _build_provider_error_pattern(patterns: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN = _build_provider_error_pattern([
    "GoUsageLimitError",
    "FreeUsageLimitError",
    "Monthly usage limit reached",
    "available balance",
    "insufficient_quota",
    "out of budget",
    "quota exceeded",
    "billing",
])

RETRYABLE_PROVIDER_ERROR_PATTERN = _build_provider_error_pattern([
    "overloaded",
    "rate.?limit",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "524",
    "service.?unavailable",
    "server.?error",
    "internal.?error",
    "provider.?returned.?error",
    "exceeded request buffer limit while retrying upstream",
    "network.?error",
    "connection.?error",
    "connection.?refused",
    "connection.?lost",
    "other side closed",
    "fetch failed",
    "getaddrinfo",
    "ENOTFOUND",
    "EAI_AGAIN",
    "upstream.?connect",
    "reset before headers",
    "socket hang up",
    "socket connection was closed",
    "timed? out",
    "timeout",
    "terminated",
    "websocket.?closed",
    "websocket.?error",
    "ended without",
    "stream ended before message_stop",
    "stream ended before a terminal response event",
    "http2 request did not get a response",
    "retry delay",
    "you can retry your request",
    "try your request again",
    "please retry your request",
    "ResourceExhausted",
])


@dataclass
class RetryPolicy:
    enabled: bool
    max_retries: int
    base_delay_ms: int


@dataclass
class RetryCallbacks:
    on_retry_scheduled: Callable[[int, int, float, str], Awaitable[None] | None] | None = None
    on_retry_attempt_start: Callable[[], Awaitable[None] | None] | None = None
    on_retry_finished: Callable[[bool, int, str | None], Awaitable[None] | None] | None = None


class RetrySleepAbortError(Exception):
    pass


async def _sleep(ms: float, cancel_event: asyncio.Event | None = None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RetrySleepAbortError()
    try:
        await asyncio.wait_for(asyncio.sleep(ms / 1000.0), timeout=None)
    except asyncio.CancelledError as exc:
        raise RetrySleepAbortError() from exc
    if cancel_event is not None and cancel_event.is_set():
        raise RetrySleepAbortError()


async def _maybe_await(value: object) -> None:
    if asyncio.iscoroutine(value) or isinstance(value, Awaitable):
        await value  # type: ignore[arg-type]


def is_retryable_assistant_error(message: AssistantMessage) -> bool:
    if getattr(message, "stop_reason", None) != "error":
        return False
    error_message = getattr(message, "error_message", None)
    if not error_message:
        return False
    if NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN.search(error_message):
        return False
    return bool(RETRYABLE_PROVIDER_ERROR_PATTERN.search(error_message))


async def retry_assistant_call(
    produce: Callable[[], Awaitable[AssistantMessage]],
    policy: RetryPolicy | None,
    cancel_event: asyncio.Event | None = None,
    callbacks: RetryCallbacks | None = None,
) -> AssistantMessage:
    max_attempts = policy.max_retries if policy and policy.enabled else 0
    attempt = 0
    last_retry: tuple[int, str] | None = None

    while True:
        response = await produce()
        stop = getattr(response, "stop_reason", None)

        if stop == "aborted":
            if last_retry and callbacks and callbacks.on_retry_finished:
                await _maybe_await(callbacks.on_retry_finished(False, last_retry[0], None))
            return response

        if stop != "error":
            if last_retry and callbacks and callbacks.on_retry_finished:
                await _maybe_await(callbacks.on_retry_finished(True, last_retry[0], None))
            return response

        if attempt >= max_attempts or not is_retryable_assistant_error(response):
            if last_retry and callbacks and callbacks.on_retry_finished:
                await _maybe_await(
                    callbacks.on_retry_finished(False, last_retry[0], getattr(response, "error_message", None))
                )
            return response

        assert policy is not None
        attempt += 1
        error_message = getattr(response, "error_message", None) or "Unknown error"
        last_retry = (attempt, error_message)
        delay_ms = policy.base_delay_ms * (2 ** (attempt - 1))
        if callbacks and callbacks.on_retry_scheduled:
            await _maybe_await(callbacks.on_retry_scheduled(attempt, max_attempts, delay_ms, error_message))

        try:
            await _sleep(delay_ms, cancel_event)
        except RetrySleepAbortError:
            if callbacks and callbacks.on_retry_finished:
                await _maybe_await(callbacks.on_retry_finished(False, attempt, error_message))
            return response.model_copy(update={"stop_reason": "aborted", "error_message": None})

        if callbacks and callbacks.on_retry_attempt_start:
            await _maybe_await(callbacks.on_retry_attempt_start())
