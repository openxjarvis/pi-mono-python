"""
Root conftest.py — loads .env and registers custom markers.

Markers:
  @pytest.mark.live   — requires real API keys; skipped unless LIVE_TESTS=1
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Load .env file (if it exists) before any test collection
# ---------------------------------------------------------------------------

def _load_dotenv(path: Path) -> None:
    """Minimal .env parser — handles KEY=value, KEY="value", # comments."""
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, raw_val = line.partition("=")
            key = key.strip()
            raw_val = raw_val.strip()
            # Strip surrounding quotes
            if len(raw_val) >= 2 and raw_val[0] == raw_val[-1] and raw_val[0] in ('"', "'"):
                raw_val = raw_val[1:-1]
            # Only set if not already in environment (don't override shell env)
            if key and key not in os.environ:
                os.environ[key] = raw_val


_load_dotenv(Path(__file__).parent / ".env")

# Also try packages/*/  in case .env lives there
for _pkg_env in Path(__file__).parent.glob("packages/*/.env"):
    _load_dotenv(_pkg_env)


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: mark test as requiring live API keys (run with LIVE_TESTS=1 or --live flag)",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests marked with @pytest.mark.live (requires real API keys)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip @pytest.mark.live tests unless --live flag or LIVE_TESTS=1 is set."""
    run_live = config.getoption("--live") or os.environ.get("LIVE_TESTS", "").lower() in ("1", "true", "yes")
    skip_live = pytest.mark.skip(reason="Live API test — run with --live or LIVE_TESTS=1")
    for item in items:
        if "live" in item.keywords and not run_live:
            item.add_marker(skip_live)
