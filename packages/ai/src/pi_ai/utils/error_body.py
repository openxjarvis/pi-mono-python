"""
Normalize provider HTTP error objects.
Mirrors packages/ai/src/utils/error-body.ts
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

MAX_PROVIDER_ERROR_BODY_CHARS = 4000


@dataclass
class NormalizedProviderError:
    message: str
    message_carries_body: bool
    status: int | None = None
    body: str | None = None


def safe_json_stringify(value: Any) -> str:
    try:
        serialized = json.dumps(value, default=str)
        return str(value) if serialized is None else serialized
    except Exception:
        return str(value)


def truncate_error_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


def _extract_status(error: BaseException) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
    metadata = getattr(error, "metadata", None) or getattr(error, "$metadata", None)
    if isinstance(metadata, dict):
        code = metadata.get("httpStatusCode") or metadata.get("http_status_code")
        if isinstance(code, int):
            return code
    response = getattr(error, "response", None) or getattr(error, "$response", None)
    if response is not None:
        code = getattr(response, "status_code", None) or getattr(response, "statusCode", None)
        if isinstance(code, int):
            return code
        if isinstance(response, dict):
            code = response.get("statusCode") or response.get("status_code")
            if isinstance(code, int):
                return code
    return None


def _is_plain_non_empty_object(value: Any) -> bool:
    return isinstance(value, dict) and len(value) > 0


def _pick_body_text(error: BaseException) -> str | None:
    body = getattr(error, "body", None)
    if isinstance(body, str):
        return body
    nested = getattr(error, "error", None)
    if _is_plain_non_empty_object(nested):
        return safe_json_stringify(nested)
    response = getattr(error, "response", None) or getattr(error, "$response", None)
    response_body = getattr(response, "body", None) if response is not None else None
    if isinstance(response, dict):
        response_body = response.get("body")
    if isinstance(response_body, str):
        return response_body
    if _is_plain_non_empty_object(response_body):
        return safe_json_stringify(response_body)
    return None


def _extract_body(error: BaseException) -> str | None:
    body_text = _pick_body_text(error)
    if body_text is None:
        return None
    trimmed = body_text.strip()
    if not trimmed:
        return None
    return truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS)


def normalize_provider_error(error: object) -> NormalizedProviderError:
    if not isinstance(error, BaseException):
        return NormalizedProviderError(message=safe_json_stringify(error), message_carries_body=False)
    status = _extract_status(error)
    body = _extract_body(error)
    message = str(error)
    message_carries_body = body is None or body in message
    return NormalizedProviderError(
        message=message,
        message_carries_body=message_carries_body,
        status=status,
        body=body,
    )


def format_provider_error(norm: NormalizedProviderError, prefix: str | None = None) -> str:
    if norm.message_carries_body or norm.status is None or norm.body is None:
        if prefix is not None and norm.status is not None:
            return f"{prefix} ({norm.status}): {norm.message}"
        return norm.message
    if prefix is not None:
        return f"{prefix} ({norm.status}): {norm.body}"
    return f"{norm.status}: {norm.body}"
