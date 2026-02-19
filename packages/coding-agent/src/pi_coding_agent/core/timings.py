"""
Central timing instrumentation for startup profiling.

Enable with PI_TIMING=1 environment variable.

Mirrors core/timings.ts
"""

from __future__ import annotations

import os
import sys
import time as _time

_ENABLED = os.environ.get("PI_TIMING") == "1"
_timings: list[dict[str, object]] = []
_last_time: float = _time.time() * 1000


def time(label: str) -> None:
    """Record a timing checkpoint (only active when PI_TIMING=1)."""
    global _last_time
    if not _ENABLED:
        return
    now = _time.time() * 1000
    _timings.append({"label": label, "ms": int(now - _last_time)})
    _last_time = now


def print_timings() -> None:
    """Print all recorded timings to stderr (only active when PI_TIMING=1)."""
    if not _ENABLED or not _timings:
        return
    print("\n--- Startup Timings ---", file=sys.stderr)
    for t in _timings:
        print(f"  {t['label']}: {t['ms']}ms", file=sys.stderr)
    total = sum(t["ms"] for t in _timings)  # type: ignore[misc]
    print(f"  TOTAL: {total}ms", file=sys.stderr)
    print("------------------------\n", file=sys.stderr)


def reset_timings() -> None:
    """Reset all timing data (for testing)."""
    global _last_time
    _timings.clear()
    _last_time = _time.time() * 1000
