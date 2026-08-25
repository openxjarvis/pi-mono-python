"""JSONL v4 codec — mirrors harness/session/jsonl/codec.ts."""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from pi_agent.harness.session.jsonl.errors import JsonlDecodeError
from pi_agent.harness.session.jsonl.types import JsonlSessionMetadata, JsonlV4Header
from pi_agent.harness.session.state import SessionMutation
from pi_agent.harness.session.types import Entry, LaneRecord
from pi_agent.harness.types import Result, err, ok

ENTRY_TYPES = {
    "message",
    "model_change",
    "thinking_level_change",
    "active_tools_change",
    "compaction",
    "branch_summary",
    "custom",
}
RECORD_TYPES = {
    "operation_started",
    "abort_requested",
    "operation_finished",
    "step_attempt",
    "tool_started",
    "queue_enqueued",
    "queue_cancelled",
    "write_deferred",
    "usage",
}
OPERATION_KINDS = {"run", "compaction", "navigation"}

_SNAKE_TO_CAMEL_SPECIAL = {
    "cache_write_1h": "cacheWrite1h",
}
_CAMEL_TO_SNAKE_SPECIAL = {value: key for key, value in _SNAKE_TO_CAMEL_SPECIAL.items()}
_SNAKE_RE = re.compile(r"_([a-z0-9])")
_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
_NULLABLE_KEYS = {
    "parent_id",
    "leaf_id",
    "source_leaf_id",
    "parentId",
    "leafId",
    "sourceLeafId",
}


def _snake_to_camel(key: str) -> str:
    if key in _SNAKE_TO_CAMEL_SPECIAL:
        return _SNAKE_TO_CAMEL_SPECIAL[key]
    return _SNAKE_RE.sub(lambda match: match.group(1).upper(), key)


def _camel_to_snake(key: str) -> str:
    if key in _CAMEL_TO_SNAKE_SPECIAL:
        return _CAMEL_TO_SNAKE_SPECIAL[key]
    if "_" in key:
        return key
    return _CAMEL_RE.sub("_", key).lower()


def _to_jsonable(value: Any, *, convert_keys: bool = True) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if item is None and key not in _NULLABLE_KEYS:
                continue
            next_key = _snake_to_camel(key) if convert_keys else key
            out[next_key] = _to_jsonable(item, convert_keys=convert_keys)
        return out
    if isinstance(value, list):
        return [_to_jsonable(item, convert_keys=convert_keys) for item in value]
    return value


def _from_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {_camel_to_snake(key): _from_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    return value


def _is_object(value: object) -> bool:
    return isinstance(value, dict)


def _parse_object(line: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise JsonlDecodeError("syntax", "is not valid JSON", error) from error
    if not _is_object(value):
        raise JsonlDecodeError("schema", "is not a JSON object")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise JsonlDecodeError("schema", f"has invalid {field}")
    return value


def _require_sequence(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise JsonlDecodeError("schema", "has invalid seq")
    return value


def _require_timestamp(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise JsonlDecodeError("schema", "has invalid timestamp")
    return value


def _require_nullable_id(value: object, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise JsonlDecodeError("schema", f"has invalid {field}")
    return value


def _decode_header(line: str) -> JsonlV4Header:
    value = _parse_object(line)
    if value.get("kind") != "header":
        raise JsonlDecodeError("schema", "is not a header")
    if value.get("version") != 4:
        raise JsonlDecodeError("schema", "has unsupported session version")
    parent_session_id = value.get("parentSessionId", value.get("parent_session_id"))
    if parent_session_id is not None and not isinstance(parent_session_id, str):
        raise JsonlDecodeError("schema", "has invalid parentSessionId")
    legacy = value.get("legacyParentSessionPath", value.get("legacy_parent_session_path"))
    if legacy is not None and not isinstance(legacy, str):
        raise JsonlDecodeError("schema", "has invalid legacyParentSessionPath")
    if parent_session_id is not None and legacy is not None:
        raise JsonlDecodeError("schema", "has both parentSessionId and legacyParentSessionPath")
    metadata_value = value.get("metadata")
    if metadata_value is not None and not _is_object(metadata_value):
        raise JsonlDecodeError("schema", "has invalid metadata")
    header: JsonlV4Header = {
        "kind": "header",
        "version": 4,
        "id": _require_string(value.get("id"), "id"),
        "created_at": _require_timestamp(value.get("createdAt", value.get("created_at"))),
        "cwd": _require_string(value.get("cwd"), "cwd"),
    }
    if parent_session_id is not None:
        header["parent_session_id"] = parent_session_id
    if legacy is not None:
        header["legacy_parent_session_path"] = legacy
    if metadata_value is not None:
        header["metadata"] = metadata_value
    return header


def parse_header(line: str) -> Result:
    try:
        return ok(_decode_header(line))
    except JsonlDecodeError as error:
        return err(error)


def encode_header(header: JsonlV4Header) -> str:
    payload = _to_jsonable(header)
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"


def metadata_from_header(header: JsonlV4Header, path: str, modified_at: float) -> JsonlSessionMetadata:
    metadata: JsonlSessionMetadata = {
        "id": header["id"],
        "created_at": header["created_at"],
        "cwd": header["cwd"],
        "path": path,
        "modified_at": int(modified_at),
        "source_format": 4,
    }
    if "parent_session_id" in header:
        metadata["parent_session_id"] = header["parent_session_id"]
    if "legacy_parent_session_path" in header:
        metadata["legacy_parent_session_path"] = header["legacy_parent_session_path"]
    if "metadata" in header:
        metadata["metadata"] = header["metadata"]
    return metadata


def _parse_entry_mutation(value: dict[str, Any], seq: int) -> SessionMutation:
    raw_lane = value.get("lane")
    lane = None if raw_lane is None else _require_string(raw_lane, "lane")
    entry_id = _require_string(value.get("id"), "id")
    entry_type = _require_string(value.get("type"), "entry type")
    if entry_type not in ENTRY_TYPES:
        raise JsonlDecodeError("schema", f"has unknown entry type {entry_type}")
    parent_id = _require_nullable_id(value.get("parentId", value.get("parent_id")), "parentId")
    timestamp = _require_timestamp(value.get("timestamp"))
    if entry_type == "custom":
        _require_string(value.get("customType", value.get("custom_type")), "customType")
    entry_fields = {key: item for key, item in value.items() if key not in ("kind", "lane")}
    entry = _from_jsonable({**entry_fields, "id": entry_id, "type": entry_type, "parentId": parent_id, "seq": seq, "timestamp": timestamp})
    mutation: SessionMutation = {"kind": "entry", "entry": entry}
    if lane is not None:
        mutation["lane"] = lane
    return mutation


def _parse_record_mutation(value: dict[str, Any], seq: int) -> SessionMutation:
    record_id = _require_string(value.get("id"), "id")
    lane = _require_string(value.get("lane"), "lane")
    record_type = _require_string(value.get("type"), "record type")
    if record_type not in RECORD_TYPES:
        raise JsonlDecodeError("schema", f"has unknown record type {record_type}")
    timestamp = _require_timestamp(value.get("timestamp"))
    if record_type == "operation_started":
        intent = value.get("intent")
        if not _is_object(intent):
            raise JsonlDecodeError("schema", "has invalid intent")
        operation_kind = _require_string(intent.get("kind"), "operation kind")
        if operation_kind not in OPERATION_KINDS:
            raise JsonlDecodeError("schema", f"has unknown operation kind {operation_kind}")
    if record_type == "operation_finished":
        _require_string(value.get("runId", value.get("run_id")), "runId")
    record_fields = {key: item for key, item in value.items() if key != "kind"}
    record = _from_jsonable({**record_fields, "id": record_id, "lane": lane, "type": record_type, "seq": seq, "timestamp": timestamp})
    return {"kind": "record", "record": record}


def _parse_lane_mutation(value: dict[str, Any], seq: int) -> SessionMutation:
    return {
        "kind": "lane",
        "seq": seq,
        "lane": _require_string(value.get("lane"), "lane"),
        "leaf_id": _require_nullable_id(value.get("leafId", value.get("leaf_id")), "leafId"),
    }


def _parse_fact_mutation(value: dict[str, Any], seq: int) -> SessionMutation:
    if value.get("fact") == "name":
        name = value.get("name")
        if name is not None and not isinstance(name, str):
            raise JsonlDecodeError("schema", "has invalid name")
        return {"kind": "fact", "seq": seq, "fact": "name", "name": name}
    if value.get("fact") == "label":
        label = value.get("label")
        if label is not None and not isinstance(label, str):
            raise JsonlDecodeError("schema", "has invalid label")
        return {
            "kind": "fact",
            "seq": seq,
            "fact": "label",
            "target_id": _require_string(value.get("targetId", value.get("target_id")), "targetId"),
            "label": label,
        }
    raise JsonlDecodeError("schema", "has unknown fact type")


def _decode_mutation(line: str) -> SessionMutation:
    value = _parse_object(line)
    seq = _require_sequence(value.get("seq"))
    kind = value.get("kind")
    if kind == "entry":
        return _parse_entry_mutation(value, seq)
    if kind == "record":
        return _parse_record_mutation(value, seq)
    if kind == "lane":
        return _parse_lane_mutation(value, seq)
    if kind == "fact":
        return _parse_fact_mutation(value, seq)
    raise JsonlDecodeError("schema", "has unknown mutation kind")


def parse_mutation(line: str) -> Result:
    try:
        return ok(_decode_mutation(line))
    except JsonlDecodeError as error:
        return err(error)


def decode_line(line: str) -> dict[str, Any]:
    text = line.strip()
    if not text:
        return {}
    return json.loads(text)


def decode_jsonl(text: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    header = None
    mutations: list[dict[str, Any]] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        obj = decode_line(raw)
        if obj.get("type") == "session" or (obj.get("id") and header is None and "type" not in obj) or obj.get("type") == "header":
            if header is None:
                header = obj
                continue
        mutations.append(obj)
    return header, mutations


def encode_mutation(mutation: SessionMutation) -> str:
    kind = mutation["kind"]
    if kind == "entry":
        payload = {"kind": "entry", **_to_jsonable(mutation["entry"])}
        if mutation.get("lane") is not None:
            payload["lane"] = mutation["lane"]
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"
    if kind == "record":
        payload = {"kind": "record", **_to_jsonable(mutation["record"])}
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"
    return json.dumps(_to_jsonable(mutation), separators=(",", ":"), ensure_ascii=False) + "\n"
