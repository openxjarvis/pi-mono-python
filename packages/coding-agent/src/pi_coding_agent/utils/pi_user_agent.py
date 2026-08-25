"""User-Agent string. Mirrors packages/coding-agent/src/utils/pi-user-agent.ts"""
from __future__ import annotations

from .version_check import get_package_version


def pi_user_agent() -> str:
    return f"pi-coding-agent/{get_package_version()}"
