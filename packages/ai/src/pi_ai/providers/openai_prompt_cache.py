"""
OpenAI prompt-cache key helpers.
Mirrors packages/ai/src/api/openai-prompt-cache.ts
"""
from __future__ import annotations


def clamp_prompt_cache_key(session_id: str | None, max_len: int = 64) -> str | None:
    if not session_id:
        return None
    return session_id[:max_len]
