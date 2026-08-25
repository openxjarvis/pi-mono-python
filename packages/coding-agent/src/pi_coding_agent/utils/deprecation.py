"""
Deprecation warnings — mirrors packages/coding-agent/src/utils/deprecation.ts
"""
from __future__ import annotations

import sys

_emitted: set[str] = set()


def warn_deprecation(message: str) -> None:
    if message in _emitted:
        return
    _emitted.add(message)
    print(f"Deprecation warning: {message}", file=sys.stderr)


def clear_deprecation_warnings_for_tests() -> None:
    _emitted.clear()
