"""
pi-messages API — POST {model, context, options} to <baseUrl>/messages.
Mirrors packages/ai/src/api/pi-messages.ts
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator

import httpx

from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    EventDone,
    EventError,
    EventStart,
    EventTextDelta,
    EventTextEnd,
    EventTextStart,
    Model,
    SimpleStreamOptions,
    TextContent,
    Usage,
)
from pi_ai.utils.pi_user_agent import get_pi_user_agent


def _empty(model: Model) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


async def stream_simple_pi_messages(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    opts = options or SimpleStreamOptions()
    url = f"{(model.base_url or '').rstrip('/')}/messages"
    headers = {"User-Agent": get_pi_user_agent(), "Accept": "text/event-stream", "Content-Type": "application/json"}
    if opts.api_key:
        headers["Authorization"] = f"Bearer {opts.api_key}"
    if opts.headers:
        headers.update({k: v for k, v in opts.headers.items() if v is not None})

    payload = {
        "model": model.id,
        "context": context.model_dump() if hasattr(context, "model_dump") else context,
        "options": {
            "temperature": opts.temperature,
            "maxTokens": opts.max_tokens,
            "reasoning": opts.reasoning,
        },
    }
    partial = _empty(model)
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()
            yield EventStart(type="start", partial=partial)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data in ("", "[DONE]"):
                    continue
                event = json.loads(data)
                etype = event.get("type")
                if etype == "text_start":
                    yield EventTextStart(type="text_start", content_index=event.get("contentIndex", 0), partial=partial)
                elif etype == "text_delta":
                    yield EventTextDelta(
                        type="text_delta",
                        content_index=event.get("contentIndex", 0),
                        delta=event.get("delta", ""),
                        partial=partial,
                    )
                elif etype == "text_end":
                    text = event.get("content", "")
                    partial.content = [TextContent(type="text", text=text)]
                    yield EventTextEnd(
                        type="text_end",
                        content_index=event.get("contentIndex", 0),
                        content=text,
                        partial=partial,
                    )
                elif etype == "done":
                    partial.stop_reason = event.get("reason", "stop")
                    yield EventDone(type="done", reason=partial.stop_reason, message=partial)
                    return
                elif etype == "error":
                    partial.stop_reason = "error"
                    partial.error_message = event.get("errorMessage")
                    yield EventError(type="error", reason="error", error=partial.error_message or "error", message=partial)
                    return
            yield EventDone(type="done", reason="stop", message=partial)


stream_pi_messages = stream_simple_pi_messages
