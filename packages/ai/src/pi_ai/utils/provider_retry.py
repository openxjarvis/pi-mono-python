"""
Interruptible provider HTTP retry, matching OpenAI/Anthropic SDK policy.
Mirrors packages/ai/src/utils/provider-retry.ts
"""
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")

DEFAULT_MAX_RETRY_DELAY_MS = 60_000


@dataclass
class ProviderRetryOptions:
    max_retries: int = 0
    max_retry_delay_ms: int | None = None
    cancel_event: asyncio.Event | None = None


class ProviderError(Exception):
    def __init__(
        self,
        message: str,
        status: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}


def _is_provider_error(error: BaseException) -> bool:
    return isinstance(error, ProviderError)


def _header(error: ProviderError, name: str) -> str | None:
    return error.headers.get(name.lower())


def _is_retryable_provider_error(error: ProviderError) -> bool:
    should_retry = _header(error, "x-should-retry")
    if should_retry == "true":
        return True
    if should_retry == "false":
        return False
    if error.status is None:
        return True
    return error.status in (408, 409, 429) or error.status >= 500


def _validate_server_retry_delay_ms(
    delay_ms: float,
    max_retry_delay_ms: int | None,
    provider_error_message: str,
) -> float:
    max_delay = DEFAULT_MAX_RETRY_DELAY_MS if max_retry_delay_ms is None else max_retry_delay_ms
    if max_delay > 0 and delay_ms > max_delay:
        raise RuntimeError(
            f"Server requested {int((delay_ms + 999) // 1000)}s retry delay "
            f"(max: {int((max_delay + 999) // 1000)}s). {provider_error_message}"
        )
    return delay_ms


def _get_retry_delay_ms(
    error: ProviderError,
    retry_index: int,
    max_retry_delay_ms: int | None,
) -> float:
    retry_after_ms = _header(error, "retry-after-ms")
    if retry_after_ms:
        try:
            return _validate_server_retry_delay_ms(float(retry_after_ms), max_retry_delay_ms, str(error))
        except ValueError:
            pass

    retry_after = _header(error, "retry-after")
    if retry_after:
        try:
            delay_ms = float(retry_after) * 1000
        except ValueError:
            parsed = time.mktime(time.strptime(retry_after, "unused")) if False else None
            try:
                from email.utils import parsedate_to_datetime

                delay_ms = parsedate_to_datetime(retry_after).timestamp() * 1000 - time.time() * 1000
            except Exception:
                delay_ms = 0
            if parsed is not None:
                delay_ms = 0
        return _validate_server_retry_delay_ms(delay_ms, max_retry_delay_ms, str(error))

    exponential_delay = min(0.5 * (2**retry_index), 8) * 1000
    return exponential_delay * (1 - random.random() * 0.25)


async def _abortable_sleep(ms: float, cancel_event: asyncio.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise asyncio.CancelledError("Request aborted")
    await asyncio.sleep(max(0.0, ms / 1000.0))
    if cancel_event is not None and cancel_event.is_set():
        raise asyncio.CancelledError("Request aborted")


async def retry_provider_request(
    request: Callable[[], Awaitable[T]],
    options: ProviderRetryOptions | None = None,
) -> T:
    opts = options or ProviderRetryOptions()
    retries_remaining = opts.max_retries
    while True:
        try:
            return await request()
        except Exception as error:
            if opts.cancel_event is not None and opts.cancel_event.is_set():
                raise asyncio.CancelledError("Request aborted") from error
            if retries_remaining <= 0 or not _is_provider_error(error) or not _is_retryable_provider_error(error):
                raise
            retry_index = opts.max_retries - retries_remaining
            retries_remaining -= 1
            await _abortable_sleep(
                _get_retry_delay_ms(error, retry_index, opts.max_retry_delay_ms),
                opts.cancel_event,
            )
