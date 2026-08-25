"""
Header conversion helpers.
Mirrors packages/ai/src/utils/headers.ts
"""
from __future__ import annotations

from typing import Mapping


def headers_to_record(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key): str(value) for key, value in headers.items()}


def provider_headers_to_record(headers: Mapping[str, str | None] | None) -> dict[str, str] | None:
    if not headers:
        return None
    result = {key: value for key, value in headers.items() if value is not None}
    return result or None
