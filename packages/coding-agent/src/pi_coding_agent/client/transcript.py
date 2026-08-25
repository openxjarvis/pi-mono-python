from __future__ import annotations

from copy import deepcopy
from typing import Any

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class TranscriptState(dict):
    def __init__(
        self,
        snapshot: dict[str, Any],
        progress_items: dict[str, dict[str, Any]] | None = None,
        progress_order: list[str] | None = None,
        tool_call_buffers: dict[str, str] | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.progress_items = progress_items or {}
        self.progress_order = progress_order or []
        self.tool_call_buffers = tool_call_buffers or {}
        super().__init__(
            snapshot=snapshot,
            progressItems=self.progress_items,
            progress_items=self.progress_items,
            progressOrder=self.progress_order,
            progress_order=self.progress_order,
            toolCallBuffers=self.tool_call_buffers,
            tool_call_buffers=self.tool_call_buffers,
        )


def is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == value and value not in (float("inf"), float("-inf"))
    if isinstance(value, list):
        return all(is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and is_json_value(item) for key, item in value.items())
    return False


def parse_partial_tool_input(value: str) -> Any:
    try:
        parsed = __import__("json").loads(value)
        if is_json_value(parsed):
            return parsed
    except Exception:
        pass
    return value


def create_transcript_state(snapshot: dict[str, Any]) -> TranscriptState:
    return TranscriptState(snapshot=deepcopy(snapshot))


def apply_transcript_snapshot(state: TranscriptState, snapshot: dict[str, Any]) -> TranscriptState:
    if state.snapshot.get("id") == snapshot.get("id") and snapshot.get("revision", 0) < state.snapshot.get("revision", 0):
        return state
    return create_transcript_state(snapshot)


def apply_transcript_progress(state: TranscriptState, progress: dict[str, Any]) -> TranscriptState:
    kind = progress.get("type")
    if kind in {"item_started", "item_updated"}:
        return _set_progress_item(state, progress["item"])
    if kind == "item_finished":
        buffers = {key: value for key, value in state.tool_call_buffers.items() if not key.startswith(f"{progress['item']['id']}:")}
        return _set_progress_item(
            TranscriptState(state.snapshot, dict(state.progress_items), list(state.progress_order), buffers),
            progress["item"],
        )

    item = state.progress_items.get(progress.get("messageId"))
    if item is None:
        item = next((candidate for candidate in state.snapshot.get("transcript", []) if candidate.get("id") == progress.get("messageId")), None)
    if item is None or item.get("role") != "assistant":
        return state

    tool_call_buffers = dict(state.tool_call_buffers)
    content = []
    for index, part in enumerate(item.get("content") or []):
        part = deepcopy(part)
        if index != progress.get("contentIndex"):
            content.append(part)
            continue
        if progress.get("kind") == "text" and part.get("type") == "text":
            part["text"] = part.get("text", "") + progress.get("delta", "")
        elif progress.get("kind") == "thinking" and part.get("type") == "thinking":
            part["thinking"] = part.get("thinking", "") + progress.get("delta", "")
        elif progress.get("kind") == "toolCall" and part.get("type") == "toolCall":
            key = f"{progress['messageId']}:{progress['contentIndex']}"
            existing = state.tool_call_buffers.get(key)
            if existing is None:
                existing = part["input"] if isinstance(part.get("input"), str) else ""
            buffer = existing + progress.get("delta", "")
            tool_call_buffers[key] = buffer
            part["input"] = parse_partial_tool_input(buffer)
        content.append(part)
    return _set_progress_item(
        TranscriptState(state.snapshot, dict(state.progress_items), list(state.progress_order), tool_call_buffers),
        {**item, "content": content},
    )


def select_transcript(state: TranscriptState) -> list[dict[str, Any]]:
    transcript = [state.progress_items.get(item["id"], item) for item in state.snapshot.get("transcript", [])]
    ids = {item["id"] for item in transcript}
    for item_id in state.progress_order:
        if item_id in ids:
            continue
        item = state.progress_items.get(item_id)
        if item:
            transcript.append(item)
            ids.add(item_id)
    for item in state.snapshot.get("queuedSteer", []):
        if item["id"] in ids:
            continue
        transcript.append(item)
        ids.add(item["id"])
    return transcript


def _set_progress_item(state: TranscriptState, item: dict[str, Any]) -> TranscriptState:
    progress_items = dict(state.progress_items)
    progress_order = list(state.progress_order) if item["id"] in progress_items else [*state.progress_order, item["id"]]
    progress_items[item["id"]] = deepcopy(item)
    return TranscriptState(state.snapshot, progress_items, progress_order, dict(state.tool_call_buffers))


# Backward-compatible empty transcript list used by older imports.
class Transcript(list):
    pass
