from __future__ import annotations

from typing import Any, Literal

PROTOCOL_VERSION = 1

THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
SESSION_PHASES = ("idle", "turn", "compaction", "branch_summary", "retry")
PROTOCOL_ERROR_CODES = (
    "version",
    "busy",
    "session_locked",
    "not_found",
    "invalid_request",
    "not_implemented",
    "internal_error",
)
COMMANDS = (
    "list",
    "create",
    "attach",
    "detach",
    "prompt",
    "steer",
    "abort",
    "set_model",
    "set_thinking",
)

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]
SessionPhase = Literal["idle", "turn", "compaction", "branch_summary", "retry"]
ProtocolErrorCode = Literal[
    "version", "busy", "session_locked", "not_found", "invalid_request", "not_implemented", "internal_error"
]


def _is_id(value: object) -> bool:
    return isinstance(value, str) and len(value) > 0


def _is_timestamp(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_json_value(value: object, optional: bool = False, ancestors: set[int] | None = None) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or isinstance(value, (int, float, str)):
        return True
    if value is None:
        return True
    if not isinstance(value, (list, dict)):
        return optional and value is None
    ancestors = ancestors or set()
    ident = id(value)
    if ident in ancestors:
        return False
    ancestors.add(ident)
    try:
        if isinstance(value, list):
            return all(_is_json_value(item, False, ancestors) for item in value)
        return all(isinstance(key, str) and _is_json_value(item, True, ancestors) for key, item in value.items())
    finally:
        ancestors.discard(ident)


def _is_model_ref(value: object) -> bool:
    return isinstance(value, dict) and _is_id(value.get("provider")) and _is_id(value.get("id")) and len(value) == 2


def _is_usage(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    cost = value.get("cost")
    if not isinstance(cost, dict):
        return False
    required = ("input", "output", "cacheRead", "cacheWrite", "totalTokens")
    return all(isinstance(value.get(key), int) and value[key] >= 0 for key in required) and all(
        isinstance(cost.get(key), (int, float)) and cost[key] >= 0 for key in ("input", "output", "cacheRead", "cacheWrite", "total")
    )


def _is_user_content(value: object) -> bool:
    if not isinstance(value, dict) or "type" not in value:
        return False
    if value["type"] == "text":
        return isinstance(value.get("text"), str)
    if value["type"] == "image":
        return isinstance(value.get("data"), str) and _is_id(value.get("mimeType"))
    return False


def _is_assistant_content(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    kind = value.get("type")
    if kind == "text":
        return isinstance(value.get("text"), str)
    if kind == "thinking":
        return isinstance(value.get("thinking"), str)
    if kind == "toolCall":
        return _is_id(value.get("toolCallId")) and _is_id(value.get("toolName")) and _is_json_value(value.get("input"))
    return False


def _is_transcript_item(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    role = value.get("role")
    if role == "user":
        return (
            _is_id(value.get("id"))
            and isinstance(value.get("content"), list)
            and all(_is_user_content(part) for part in value["content"])
            and _is_timestamp(value.get("timestamp"))
        )
    if role == "assistant":
        return _is_id(value.get("id")) and isinstance(value.get("content"), list) and _is_model_ref(value.get("model"))
    if role == "tool":
        return _is_id(value.get("id")) and _is_id(value.get("toolCallId")) and _is_id(value.get("toolName"))
    return False


def _is_session_metadata(value: object) -> bool:
    return isinstance(value, dict) and _is_id(value.get("id")) and _is_timestamp(value.get("createdAt"))


def _is_session_snapshot(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        _is_id(value.get("id"))
        and isinstance(value.get("cwd"), str)
        and len(value["cwd"]) > 0
        and value.get("phase") in SESSION_PHASES
        and _is_model_ref(value.get("model"))
        and value.get("thinkingLevel") in THINKING_LEVELS
        and isinstance(value.get("attached"), bool)
        and isinstance(value.get("locked"), bool)
        and isinstance(value.get("revision"), int)
        and isinstance(value.get("transcript"), list)
        and isinstance(value.get("queuedSteer"), list)
        and isinstance(value.get("queuedSteerCount"), int)
    )


def _is_server_snapshot(value: object) -> bool:
    return (
        isinstance(value, dict)
        and _is_id(value.get("serverId"))
        and value.get("protocolVersion") == PROTOCOL_VERSION
        and isinstance(value.get("revision"), int)
        and isinstance(value.get("sessions"), list)
        and isinstance(value.get("models"), list)
    )


def _is_command(value: object) -> bool:
    if not isinstance(value, dict) or value.get("command") not in COMMANDS:
        return False
    command = value["command"]
    if command == "list":
        return True
    if command == "create":
        return True
    if command in {"attach", "detach", "abort"}:
        return _is_id(value.get("sessionId"))
    if command in {"prompt", "steer"}:
        return _is_id(value.get("sessionId")) and isinstance(value.get("text"), str)
    if command == "set_model":
        return _is_id(value.get("sessionId")) and _is_model_ref(value.get("model"))
    if command == "set_thinking":
        return _is_id(value.get("sessionId")) and value.get("thinkingLevel") in THINKING_LEVELS
    return False


def _is_command_result(value: object) -> bool:
    if not isinstance(value, dict) or value.get("command") not in COMMANDS:
        return False
    command = value["command"]
    if command == "list":
        return isinstance(value.get("sessions"), list)
    if command == "detach":
        return _is_id(value.get("sessionId"))
    return _is_session_snapshot(value.get("session"))


def _is_protocol_error(value: object) -> bool:
    return isinstance(value, dict) and value.get("code") in PROTOCOL_ERROR_CODES and isinstance(value.get("message"), str)


def is_client_message(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("type") == "hello":
        return isinstance(value.get("version"), int) and value["version"] >= 0
    if value.get("type") == "request":
        return _is_id(value.get("id")) and _is_command(value.get("request"))
    return False


def is_server_message(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    kind = value.get("type")
    if kind == "hello":
        return (
            value.get("version") == PROTOCOL_VERSION
            and _is_id(value.get("connectionId"))
            and _is_server_snapshot(value.get("snapshot"))
        )
    if kind == "hello_error":
        return _is_protocol_error(value.get("error"))
    if kind == "response":
        if not _is_id(value.get("id")) or not isinstance(value.get("ok"), bool):
            return False
        if value["ok"]:
            return _is_command_result(value.get("result"))
        return _is_protocol_error(value.get("error"))
    if kind == "event":
        event = value.get("event")
        if not isinstance(event, dict):
            return False
        event_type = event.get("type")
        if event_type == "server_snapshot":
            return _is_server_snapshot(event.get("snapshot"))
        if event_type == "session_snapshot":
            return _is_session_snapshot(event.get("snapshot"))
        if event_type == "session_progress":
            return _is_id(event.get("sessionId"))
        if event_type == "session_removed":
            return _is_id(event.get("sessionId"))
    return False


# Type aliases used by client/server ports
ClientMessage = dict[str, Any]
ServerMessage = dict[str, Any]
Command = dict[str, Any]
CommandResult = dict[str, Any]
ServerSnapshot = dict[str, Any]
SessionSnapshot = dict[str, Any]
SessionMetadata = dict[str, Any]
ServerEvent = dict[str, Any]
ProtocolError = dict[str, Any]
ModelRef = dict[str, str]
ModelMetadata = dict[str, Any]
ClientHello = dict[str, Any]
RequestEnvelope = dict[str, Any]
ResponseEnvelope = dict[str, Any]
EventEnvelope = dict[str, Any]
ServerHello = dict[str, Any]
ServerHelloError = dict[str, Any]
TranscriptProgress = dict[str, Any]
Usage = dict[str, Any]
UserTranscriptItem = dict[str, Any]
AssistantTranscriptItem = dict[str, Any]
ToolTranscriptItem = dict[str, Any]
TranscriptItem = dict[str, Any]
TextContent = dict[str, Any]
ThinkingContent = dict[str, Any]
ImageContent = dict[str, Any]
ToolCallContent = dict[str, Any]
