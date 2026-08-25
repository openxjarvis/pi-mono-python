"""JSON helpers. Mirrors packages/coding-agent/src/utils/json.ts"""
from __future__ import annotations

import json
from typing import Any


def parse_json(text: str, default: Any = None) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def stringify_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
