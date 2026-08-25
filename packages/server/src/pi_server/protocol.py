from __future__ import annotations

from typing import Any

from pi_ai.models import get_supported_thinking_levels


def to_protocol_json_value(value: object, seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value != value:
            raise TypeError("Protocol JSON numbers must be finite")
        return value
    if not isinstance(value, (list, dict)):
        raise TypeError(f"Unsupported protocol JSON value: {type(value).__name__}")
    seen = seen or set()
    ident = id(value)
    if ident in seen:
        raise TypeError("Protocol JSON values must not contain circular references")
    seen.add(ident)
    try:
        if isinstance(value, list):
            return [to_protocol_json_value(entry, seen) for entry in value]
        return {str(key): to_protocol_json_value(entry, seen) for key, entry in value.items()}
    finally:
        seen.discard(ident)


def sanitize_protocol_details(value: object, seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if value is None:
        return None
    if not isinstance(value, (list, dict)):
        return str(value)
    seen = seen or set()
    ident = id(value)
    if ident in seen:
        return "[Circular]"
    seen.add(ident)
    try:
        if isinstance(value, list):
            return [sanitize_protocol_details(entry, seen) for entry in value]
        result = {}
        for key, entry in value.items():
            normalized = sanitize_protocol_details(entry, seen)
            if normalized is not None:
                result[str(key)] = normalized
        return result
    finally:
        seen.discard(ident)


def to_protocol_usage(usage: Any | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif hasattr(usage, "__dict__") and not isinstance(usage, dict):
        usage = {**usage.__dict__}
    cost = usage.get("cost") or {}
    reasoning = usage.get("reasoning")
    result = {
        "input": int(usage.get("input") or 0),
        "output": int(usage.get("output") or 0),
        "cacheRead": int(usage.get("cacheRead") or usage.get("cache_read") or 0),
        "cacheWrite": int(usage.get("cacheWrite") or usage.get("cache_write") or 0),
        "totalTokens": int(usage.get("totalTokens") or usage.get("total_tokens") or 0),
        "cost": {
            "input": float(cost.get("input") or 0),
            "output": float(cost.get("output") or 0),
            "cacheRead": float(cost.get("cacheRead") or cost.get("cache_read") or 0),
            "cacheWrite": float(cost.get("cacheWrite") or cost.get("cache_write") or 0),
            "total": float(cost.get("total") or 0),
        },
    }
    if reasoning is not None:
        result["reasoning"] = int(reasoning)
    return result


def to_protocol_model_metadata(model: Any, authenticated: bool) -> dict[str, Any]:
    return {
        "provider": model.provider,
        "id": model.id,
        "name": model.name,
        "api": model.api,
        "reasoning": bool(getattr(model, "reasoning", False)),
        "input": list(getattr(model, "input", ["text"])),
        "contextWindow": max(1, int(getattr(model, "context_window", getattr(model, "contextWindow", 1)))),
        "maxTokens": max(1, int(getattr(model, "max_tokens", getattr(model, "maxTokens", 1)))),
        "cost": {
            "input": float(getattr(model.cost, "input", 0)),
            "output": float(getattr(model.cost, "output", 0)),
            "cacheRead": float(getattr(model.cost, "cache_read", getattr(model.cost, "cacheRead", 0))),
            "cacheWrite": float(getattr(model.cost, "cache_write", getattr(model.cost, "cacheWrite", 0))),
        },
        "supportedThinkingLevels": list(get_supported_thinking_levels(model)),
        "authenticated": authenticated,
    }


def to_protocol_user_message(message: Any, options: dict[str, str]) -> dict[str, Any]:
    content = message.content if not isinstance(message, dict) else message["content"]
    timestamp = message.timestamp if not isinstance(message, dict) else message["timestamp"]
    if isinstance(content, str):
        parts = [{"type": "text", "text": content}]
    else:
        parts = []
        for part in content:
            kind = part["type"] if isinstance(part, dict) else part.type
            if kind == "text":
                parts.append({"type": "text", "text": part["text"] if isinstance(part, dict) else part.text})
            elif kind == "image":
                parts.append(
                    {
                        "type": "image",
                        "data": part["data"] if isinstance(part, dict) else part.data,
                        "mimeType": part.get("mimeType", part.get("mime_type")) if isinstance(part, dict) else part.mime_type,
                    }
                )
    return {"id": options["id"], "role": "user", "content": parts, "timestamp": timestamp}
