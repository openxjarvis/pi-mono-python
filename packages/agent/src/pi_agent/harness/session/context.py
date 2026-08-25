"""Build model-visible session context — mirrors harness/session/context.ts."""
from __future__ import annotations

from typing import Any, Callable, TypedDict

from pi_agent.harness.messages import create_branch_summary_message, create_compaction_summary_message
from pi_agent.harness.session.types import CustomEntry, Entry
from pi_agent.types import AgentMessage


class SessionContext(TypedDict):
    messages: list[AgentMessage]
    thinking_level: str
    model: dict[str, str] | None
    active_tool_names: list[str] | None


ContextEntryTransform = Callable[[list[Entry]], list[Entry]]
CustomEntryContextMessageProjector = Callable[[CustomEntry, int, list[Entry]], list[AgentMessage] | None]


class SessionContextBuildOptions(TypedDict, total=False):
    entry_transforms: list[ContextEntryTransform]
    entry_projectors: dict[str, CustomEntryContextMessageProjector]


def _message_role(message: Any) -> str | None:
    return getattr(message, "role", None) if not isinstance(message, dict) else message.get("role")


def _message_field(message: Any, snake: str, camel: str) -> Any:
    if isinstance(message, dict):
        return message.get(snake, message.get(camel))
    return getattr(message, snake, None)


def derive_session_context_state(path_entries: list[Entry]) -> dict[str, Any]:
    thinking_level = "off"
    model: dict[str, str] | None = None
    active_tool_names: list[str] | None = None
    for entry in path_entries:
        entry_type = entry.get("type")
        if entry_type == "thinking_level_change":
            thinking_level = entry.get("thinking_level") or entry.get("thinkingLevel") or thinking_level
        elif entry_type == "model_change":
            model = {
                "provider": entry["provider"],
                "model_id": entry.get("model_id") or entry.get("modelId"),
            }
        elif entry_type == "message" and _message_role(entry.get("message")) == "assistant":
            message = entry["message"]
            model = {
                "provider": _message_field(message, "provider", "provider"),
                "model_id": _message_field(message, "model", "model"),
            }
        elif entry_type == "active_tools_change":
            active_tool_names = list(entry.get("active_tool_names") or entry.get("activeToolNames") or [])
    return {"thinking_level": thinking_level, "model": model, "active_tool_names": active_tool_names}


def default_context_entry_transform(path_entries: list[Entry]) -> list[Entry]:
    compaction = None
    compaction_index = -1
    for index in range(len(path_entries) - 1, -1, -1):
        if path_entries[index].get("type") == "compaction":
            compaction = path_entries[index]
            compaction_index = index
            break
    return list(path_entries) if compaction is None else [compaction, *path_entries[compaction_index + 1 :]]


def build_context_entries(path_entries: list[Entry], options: SessionContextBuildOptions | None = None) -> list[Entry]:
    options = options or {}
    entries = default_context_entry_transform(path_entries)
    for transform in options.get("entry_transforms") or []:
        entries = list(transform(entries))
    return entries


def session_entry_to_context_messages(
    entry: Entry,
    index: int,
    entries: list[Entry],
    options: SessionContextBuildOptions | None = None,
) -> list[AgentMessage]:
    options = options or {}
    entry_type = entry.get("type")
    if entry_type == "message":
        message = entry["message"]
        if _message_role(message) == "assistant" and _message_field(message, "stop_reason", "stopReason") == "deferred":
            return []
        return [message]
    if entry_type == "compaction":
        return [
            create_compaction_summary_message(
                entry["summary"],
                entry.get("tokens_before", entry.get("tokensBefore", 0)),
                entry["timestamp"],
            ),
            *entry.get("retained_tail", entry.get("retainedTail", [])),
        ]
    if entry_type == "branch_summary" and entry.get("summary"):
        return [
            create_branch_summary_message(
                entry["summary"],
                entry.get("from_id", entry.get("fromId")),
                entry["timestamp"],
            )
        ]
    if entry_type == "custom":
        projectors = options.get("entry_projectors") or {}
        custom_type = entry.get("custom_type") or entry.get("customType")
        projector = projectors.get(custom_type) if custom_type else None
        return list(projector(entry, index, entries) or []) if projector else []
    return []


def build_session_context(
    path_entries: list[Entry],
    options: SessionContextBuildOptions | None = None,
) -> SessionContext:
    state = derive_session_context_state(path_entries)
    context_entries = build_context_entries(path_entries, options)
    messages: list[AgentMessage] = []
    for index, entry in enumerate(context_entries):
        messages.extend(session_entry_to_context_messages(entry, index, context_entries, options))
    return SessionContext(messages=messages, **state)
