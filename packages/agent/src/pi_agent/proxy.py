"""
Proxy stream function — mirrors packages/agent/src/proxy.ts

Allows routing LLM calls through a server endpoint.
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
    EventToolCallEnd,
    EventToolCallStart,
    Model,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    Usage,
)


async def stream_proxy(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    proxy_url: str = "/api/stream",
    extra_headers: dict[str, str] | None = None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """
    Stream LLM responses through a proxy server.
    Mirrors streamProxy() in TypeScript.

    The proxy server should accept the same request format and return
    Server-Sent Events (SSE) with AssistantMessageEvent JSON.
    """
    opts = options or SimpleStreamOptions()

    payload = {
        "model": model.model_dump(),
        "context": context.model_dump(),
        "options": opts.model_dump(exclude_none=True),
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if extra_headers:
        headers.update(extra_headers)

    partial: AssistantMessage | None = None

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                proxy_url,
                json=payload,
                headers=headers,
                timeout=300,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        event_data = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    event = _parse_proxy_event(event_data, model)
                    if event:
                        yield event

    except httpx.HTTPError as e:
        error_msg = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="error",
            error_message=str(e),
            timestamp=int(time.time() * 1000),
        )
        yield EventError(type="error", reason="error", error=error_msg)


def _parse_proxy_event(data: dict[str, Any], model: Model) -> AssistantMessageEvent | None:
    """Parse a proxy server-sent event into an AssistantMessageEvent."""
    event_type = data.get("type")

    if event_type == "start":
        partial = _parse_partial_message(data.get("partial", {}), model)
        return EventStart(type="start", partial=partial)

    elif event_type == "text_start":
        partial = _parse_partial_message(data.get("partial", {}), model)
        return EventTextStart(
            type="text_start",
            content_index=data.get("contentIndex", 0),
            partial=partial,
        )

    elif event_type == "text_delta":
        partial = _parse_partial_message(data.get("partial", {}), model)
        return EventTextDelta(
            type="text_delta",
            content_index=data.get("contentIndex", 0),
            delta=data.get("delta", ""),
            partial=partial,
        )

    elif event_type == "text_end":
        partial = _parse_partial_message(data.get("partial", {}), model)
        return EventTextEnd(
            type="text_end",
            content_index=data.get("contentIndex", 0),
            content=data.get("content", ""),
            partial=partial,
        )

    elif event_type == "done":
        message = _parse_assistant_message(data.get("message", {}), model)
        return EventDone(type="done", reason=data.get("reason", "stop"), message=message)

    elif event_type == "error":
        error = _parse_assistant_message(data.get("error", {}), model)
        return EventError(type="error", reason=data.get("reason", "error"), error=error)

    return None


def _parse_partial_message(data: dict[str, Any], model: Model) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api=data.get("api", model.api),
        provider=data.get("provider", model.provider),
        model=data.get("model", model.id),
        usage=Usage(),
        stop_reason=data.get("stopReason", "stop"),
        timestamp=data.get("timestamp", int(time.time() * 1000)),
    )


def _parse_assistant_message(data: dict[str, Any], model: Model) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api=data.get("api", model.api),
        provider=data.get("provider", model.provider),
        model=data.get("model", model.id),
        usage=Usage(),
        stop_reason=data.get("stopReason", "error"),
        error_message=data.get("errorMessage"),
        timestamp=data.get("timestamp", int(time.time() * 1000)),
    )
