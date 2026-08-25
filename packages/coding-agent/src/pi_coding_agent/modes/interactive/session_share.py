"""Session sharing helpers. Mirrors packages/coding-agent/src/modes/interactive/session-share.ts"""
from __future__ import annotations

from typing import Any


async def share_session_as_gist(session_text: str, filename: str = "session.jsonl") -> dict[str, Any]:
    return {"ok": False, "error": "GitHub gist sharing is not configured", "filename": filename, "bytes": len(session_text)}
