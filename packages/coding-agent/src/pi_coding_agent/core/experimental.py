"""
Experimental feature flags — mirrors packages/coding-agent/src/core/experimental.ts
"""
from __future__ import annotations

import os
from typing import Any

PREFER_STRICT_TOOL_SAMPLING: dict[str, Any] = {"type": "json_schema", "strict": "prefer"}


def are_experimental_features_enabled() -> bool:
    return os.environ.get("PI_EXPERIMENTAL") == "1"


def get_experimental_tool_sampling() -> dict[str, Any] | None:
    return PREFER_STRICT_TOOL_SAMPLING if are_experimental_features_enabled() else None
