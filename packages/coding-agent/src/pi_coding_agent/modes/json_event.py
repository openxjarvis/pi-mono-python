"""
JSON/RPC event shaping — mirrors packages/coding-agent/src/modes/json-event.ts
"""
from __future__ import annotations

from typing import Any


def _to_json_assistant_message_event(event: Any) -> Any:
    if isinstance(event, dict):
        event_type = event.get("type")
        if event_type == "toolcall_start":
            partial = event.get("partial") or {}
            content = partial.get("content") if isinstance(partial, dict) else getattr(partial, "content", [])
            index = event.get("contentIndex", event.get("content_index", 0))
            tool_call = content[index] if isinstance(content, list) and 0 <= index < len(content) else None
            payload = {k: v for k, v in event.items() if k != "partial"}
            if isinstance(tool_call, dict):
                payload["id"] = tool_call.get("id")
                payload["toolName"] = tool_call.get("name")
            return payload
        if "partial" in event:
            return {k: v for k, v in event.items() if k != "partial"}
        return event
    event_type = getattr(event, "type", None)
    if event_type == "toolcall_start":
        return event
    return event


def to_json_event(event: Any) -> Any:
    event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
    if event_type != "message_update":
        return event
    message = event.get("message") if isinstance(event, dict) else getattr(event, "message", None)
    role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
    if role != "assistant":
        raise ValueError("message_update message is not an assistant message")
    usage = message.get("usage") if isinstance(message, dict) else getattr(message, "usage", None)
    assistant_event = (
        event.get("assistantMessageEvent")
        or event.get("assistant_message_event")
        if isinstance(event, dict)
        else getattr(event, "assistant_message_event", None)
    )
    return {
        "type": "message_update",
        "usage": usage,
        "assistantMessageEvent": _to_json_assistant_message_event(assistant_event),
    }
