"""
User-Agent helper — mirrors packages/ai/src/utils/pi-user-agent.ts
"""
from __future__ import annotations

import platform


def get_pi_user_agent() -> str:
    """Return a pi user-agent string identifying OS platform, release, and arch."""
    return f"pi ({platform.system().lower()} {platform.release()}; {platform.machine()})"
