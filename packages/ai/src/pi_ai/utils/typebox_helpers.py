"""
TypeBox helpers counterpart — Python uses dict JSON Schema.
Mirrors packages/ai/src/utils/typebox-helpers.ts
"""
from __future__ import annotations

from typing import Any


def strip_schema_meta(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    if not schema:
        return schema
    cleaned = dict(schema)
    for key in ("$schema", "$defs", "definitions"):
        cleaned.pop(key, None)
    return cleaned
