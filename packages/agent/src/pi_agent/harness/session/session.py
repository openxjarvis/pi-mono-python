"""Session facade — mirrors harness/session/session.ts."""
from __future__ import annotations

from typing import Any, Callable

from pi_ai.utils.uuid import uuidv7

from pi_agent.harness.session.types import (
    Entry,
    EntryQuery,
    IdGenerator,
    LanePointer,
    LaneRecord,
    LogItem,
    LogOptions,
    NewRecord,
    OperationStartedRecord,
    ProvisionedEntry,
    RecordQuery,
    SessionError,
    SessionMetadata,
    SessionStats,
)
from pi_agent.types import AgentMessage


def _invalid_payload(reason: str) -> None:
    raise SessionError("invalid_payload", f"Durable payload {reason}")


def assert_valid_limit(limit: int | None) -> None:
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise SessionError("invalid_query", "limit must be a positive integer")


def assert_valid_cursor(after_seq: int | None) -> None:
    if after_seq is not None and (not isinstance(after_seq, int) or isinstance(after_seq, bool) or after_seq < 0):
        raise SessionError("invalid_query", "cursor sequence must be a non-negative integer")


def assert_json_serializable(value: object) -> None:
    from pydantic import BaseModel

    active: set[int] = set()
    stack: list[dict[str, object]] = [{"value": value}]
    while stack:
        frame = stack.pop()
        if "exit" in frame:
            active.discard(id(frame["exit"]))
            continue
        candidate = frame["value"]
        if candidate is None or isinstance(candidate, (str, bool)):
            continue
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            if isinstance(candidate, float) and (candidate != candidate or candidate in (float("inf"), float("-inf"))):
                _invalid_payload("contains a non-finite number")
            continue
        if isinstance(candidate, BaseModel):
            stack.append({"value": candidate.model_dump()})
            continue
        if not isinstance(candidate, (dict, list)):
            _invalid_payload(f"contains {type(candidate).__name__}")
        ident = id(candidate)
        if ident in active:
            _invalid_payload("contains a cycle")
        active.add(ident)
        stack.append({"exit": candidate})
        if isinstance(candidate, list):
            for item in reversed(candidate):
                stack.append({"value": item})
            continue
        for key in reversed(list(candidate.keys())):
            if not isinstance(key, str):
                _invalid_payload("contains a non-string key")
            stack.append({"value": candidate[key]})


class _DefaultIdGenerator:
    def next(self) -> str:
        return uuidv7()


class Session:
    def __init__(self, storage: Any, options: dict[str, Any] | None = None) -> None:
        self._storage = storage
        options = options or {}
        self.id_generator: IdGenerator = options.get("id_generator") or _DefaultIdGenerator()

    async def get_metadata(self) -> SessionMetadata:
        return await self._storage.get_metadata()

    def view(self, lane: str) -> Any:
        if lane == "main":
            return self

        async def get_leaf_id() -> str | None:
            return await self._get_leaf_id_for_lane(lane)

        async def find_entry(query: EntryQuery | None = None) -> Entry | None:
            results = await self._query_entries(query or {}, 1)
            return results[0] if results else None

        async def find_entry_on_branch(query: dict[str, Any] | None = None) -> Entry | None:
            results = await self._query_branch_entries(lane, query or {}, 1)
            return results[0] if results else None

        return _SessionView(
            get_leaf_id=get_leaf_id,
            get_entry=self.get_entry,
            get_stats=self.get_stats,
            get_name=self.get_name,
            set_name=self.set_name,
            get_label=self.get_label,
            set_label=self.set_label,
            find_entries=lambda query=None: self._query_entries(query or {}),
            find_entry=find_entry,
            find_entries_on_branch=lambda query=None: self._query_branch_entries(lane, query or {}),
            find_entry_on_branch=find_entry_on_branch,
            append_message=lambda message: self._append_message_to_lane(lane, message),
            append_custom_entry=lambda custom_type, data=None: self._append_custom_entry_to_lane(
                lane, custom_type, data
            ),
        )

    async def get_leaf_id(self) -> str | None:
        return await self._get_leaf_id_for_lane("main")

    async def get_entry(self, entry_id: str) -> Entry | None:
        return await self._storage.get_entry(entry_id)

    async def get_stats(self) -> SessionStats:
        return await self._storage.get_stats()

    async def get_name(self) -> str | None:
        return await self._storage.get_name()

    async def set_name(self, name: str | None) -> None:
        await self._storage.set_name(name)

    async def get_label(self, target_id: str) -> str | None:
        return await self._storage.get_label(target_id)

    async def set_label(self, target_id: str, label: str | None) -> None:
        await self._storage.set_label(target_id, label)

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        return await self._query_entries(query or {})

    async def find_entry(self, query: EntryQuery | None = None) -> Entry | None:
        results = await self._query_entries(query or {}, 1)
        return results[0] if results else None

    async def find_entries_on_branch(self, query: dict[str, Any] | None = None) -> list[Entry]:
        return await self._query_branch_entries("main", query or {})

    async def find_entry_on_branch(self, query: dict[str, Any] | None = None) -> Entry | None:
        results = await self._query_branch_entries("main", query or {}, 1)
        return results[0] if results else None

    async def append_message(self, message: AgentMessage) -> str:
        return await self._append_message_to_lane("main", message)

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        return await self._append_custom_entry_to_lane("main", custom_type, data)

    async def get_lanes(self) -> list[LanePointer]:
        return await self._storage.get_lanes()

    async def create_lane(self, lane: str, at: str | None) -> None:
        await self._storage.create_lane(lane, at)

    async def move_lane(self, lane: str, to: str | None) -> None:
        await self._storage.move_lane(lane, to)

    async def append_entry(self, entry: ProvisionedEntry, lane: str) -> Entry:
        return await self._commit_entry(entry, lane)

    async def append_record(self, record: NewRecord) -> LaneRecord:
        return await self._commit_record(record)

    async def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]:
        return await self._query_records(query or {})

    async def find_open_operations(self, lane: str, options: dict[str, Any] | None = None) -> list[OperationStartedRecord]:
        assert_valid_limit((options or {}).get("limit"))
        return await self._storage.find_open_operations(lane, options)

    async def get_log(self, options: LogOptions | None = None) -> list[LogItem]:
        return await self._query_log(options or {})

    async def _get_leaf_id_for_lane(self, lane: str) -> str | None:
        pointer = next((item for item in await self.get_lanes() if item["lane"] == lane), None)
        if pointer is None:
            raise SessionError("invalid_lane", f"Lane not found: {lane}")
        return pointer["leaf_id"]

    async def _query_entries(self, query: dict[str, Any], result_limit: int | None = None) -> list[Entry]:
        if result_limit is None:
            result_limit = query.get("limit")
        assert_valid_limit(query.get("limit"))
        cursor = query.get("cursor") or {}
        assert_valid_cursor(cursor.get("after_seq", cursor.get("afterSeq")))
        storage_query = query if result_limit == query.get("limit") else {**query, "limit": result_limit}
        return await self._storage.find_entries(storage_query)

    async def _query_branch_entries(
        self,
        default_lane: str,
        query: dict[str, Any],
        result_limit: int | None = None,
    ) -> list[Entry]:
        if result_limit is None:
            result_limit = query.get("limit")
        assert_valid_limit(query.get("limit"))
        cursor = query.get("cursor") or {}
        assert_valid_cursor(cursor.get("after_seq", cursor.get("afterSeq")))
        start = query.get("start")
        if start is None:
            start = await self._get_leaf_id_for_lane(default_lane)
        if start is None:
            return []
        storage_query = query if result_limit == query.get("limit") else {**query, "limit": result_limit}
        return await self._storage.find_entries_on_branch({**storage_query, "start": start})

    async def _query_records(self, query: dict[str, Any]) -> list[LaneRecord]:
        assert_valid_limit(query.get("limit"))
        assert_valid_cursor(query.get("after_seq"))
        if query.get("operation_kind") is not None and query.get("type") != "operation_started":
            raise SessionError("invalid_query", 'operationKind requires type "operation_started"')
        return await self._storage.find_records(query)

    async def _query_log(self, options: dict[str, Any]) -> list[LogItem]:
        assert_valid_limit(options.get("limit"))
        assert_valid_cursor(options.get("after_seq"))
        return await self._storage.get_log(options)

    async def _append_message_to_lane(self, lane: str, message: AgentMessage) -> str:
        entry = await self._commit_entry({"type": "message", "id": self.id_generator.next(), "message": message}, lane)
        return entry["id"]

    async def _append_custom_entry_to_lane(self, lane: str, custom_type: str, data: Any = None) -> str:
        payload: dict[str, Any] = {"type": "custom", "id": self.id_generator.next(), "custom_type": custom_type}
        if data is not None:
            payload["data"] = data
        entry = await self._commit_entry(payload, lane)
        return entry["id"]

    async def _commit_entry(self, entry: ProvisionedEntry, lane: str) -> Entry:
        assert_json_serializable(entry)
        return await self._storage.append_entry(entry, lane)

    async def _commit_record(self, record: NewRecord) -> LaneRecord:
        assert_json_serializable(record)
        return await self._storage.append_record(record)


class _SessionView:
    def __init__(self, **methods: Callable[..., Any]) -> None:
        for name, method in methods.items():
            setattr(self, name, method)
