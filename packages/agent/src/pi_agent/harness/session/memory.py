"""In-memory session storage/repo — mirrors harness/session/memory.ts."""
from __future__ import annotations

import copy
import time
from typing import Any

from pi_ai.utils.uuid import uuidv7

from pi_agent.harness.session.session import Session
from pi_agent.harness.session.state import SessionState
from pi_agent.harness.session.types import (
    Entry,
    ForkOptions,
    LanePointer,
    LaneRecord,
    LogItem,
    NewRecord,
    OperationStartedRecord,
    ProvisionedEntry,
    SessionCreateOptions,
    SessionError,
    SessionMetadata,
    SessionStats,
)


class InMemorySessionStorage:
    def __init__(self, metadata: SessionMetadata) -> None:
        self._metadata = copy.deepcopy(metadata)
        self._state = SessionState()

    def fork(self, metadata: SessionMetadata, options: ForkOptions) -> InMemorySessionStorage:
        storage = InMemorySessionStorage(metadata)
        for mutation in self._state.create_fork_mutations(options):
            storage._state.apply_mutation(mutation)
        return storage

    async def get_metadata(self) -> SessionMetadata:
        return copy.deepcopy(self._metadata)

    async def get_lanes(self) -> list[LanePointer]:
        return self._state.get_lanes()

    async def create_lane(self, lane: str, at: str | None) -> None:
        self._state.validate_new_lane(lane)
        self._state.validate_target(at)
        self._state.apply_mutation({"kind": "lane", "seq": self._state.next_sequence, "lane": lane, "leaf_id": at})

    async def move_lane(self, lane: str, to: str | None) -> None:
        self._state.require_lane(lane)
        self._state.validate_target(to)
        self._state.apply_mutation({"kind": "lane", "seq": self._state.next_sequence, "lane": lane, "leaf_id": to})

    async def append_entry(self, new_entry: ProvisionedEntry, lane: str) -> Entry:
        parent_id = self._state.require_lane(lane)
        self._state.validate_unused_id(new_entry["id"])
        entry = {
            **copy.deepcopy(new_entry),
            "parent_id": parent_id,
            "seq": self._state.next_sequence,
            "timestamp": int(time.time() * 1000),
        }
        self._state.apply_mutation({"kind": "entry", "lane": lane, "entry": entry})
        return copy.deepcopy(entry)

    async def append_record(self, new_record: NewRecord) -> LaneRecord:
        self._state.require_lane(new_record["lane"])
        self._state.validate_unused_id(new_record["id"])
        current = self._state.find_open_operations(new_record["lane"], {"limit": 1})
        current_open = current[0]["id"] if current else None
        if new_record.get("type") == "operation_started" and current_open is not None:
            raise SessionError("storage", f"Lane {new_record['lane']} already has an open operation {current_open}")
        record = {
            **copy.deepcopy(new_record),
            "seq": self._state.next_sequence,
            "timestamp": int(time.time() * 1000),
        }
        self._state.apply_mutation({"kind": "record", "record": record})
        return copy.deepcopy(record)

    async def get_entry(self, entry_id: str) -> Entry | None:
        entry = self._state.get_entry(entry_id)
        return None if entry is None else copy.deepcopy(entry)

    async def find_entries(self, query: dict[str, Any] | None = None) -> list[Entry]:
        return copy.deepcopy(self._state.find_entries(query or {}))

    async def find_entries_on_branch(self, query: dict[str, Any]) -> list[Entry]:
        return copy.deepcopy(self._state.find_entries_on_branch(query))

    async def find_records(self, query: dict[str, Any] | None = None) -> list[LaneRecord]:
        return copy.deepcopy(self._state.find_records(query or {}))

    async def find_open_operations(
        self, lane: str, options: dict[str, Any] | None = None
    ) -> list[OperationStartedRecord]:
        return copy.deepcopy(self._state.find_open_operations(lane, options))

    async def get_log(self, options: dict[str, Any] | None = None) -> list[LogItem]:
        return copy.deepcopy(self._state.get_log(options or {}))

    async def get_name(self) -> str | None:
        return self._state.get_name()

    async def set_name(self, name: str | None) -> None:
        self._state.apply_mutation({"kind": "fact", "seq": self._state.next_sequence, "fact": "name", "name": name})

    async def get_label(self, entry_id: str) -> str | None:
        return self._state.get_label(entry_id)

    async def set_label(self, entry_id: str, label: str | None) -> None:
        self._state.validate_target(entry_id)
        self._state.apply_mutation(
            {
                "kind": "fact",
                "seq": self._state.next_sequence,
                "fact": "label",
                "target_id": entry_id,
                "label": label,
            }
        )

    async def get_stats(self) -> SessionStats:
        return copy.deepcopy(self._state.get_stats())


class InMemorySessionRepo:
    def __init__(self) -> None:
        self._sessions: dict[str, InMemorySessionStorage] = {}

    async def create(self, options: SessionCreateOptions | None = None) -> Session:
        options = options or {}
        session_id = options.get("id") or uuidv7()
        if session_id in self._sessions:
            raise SessionError("already_exists", f"Session already exists: {session_id}")
        metadata: SessionMetadata = {"id": session_id, "created_at": int(time.time() * 1000)}
        if options.get("parent_session_id") is not None:
            metadata["parent_session_id"] = options["parent_session_id"]
        storage = InMemorySessionStorage(metadata)
        self._sessions[session_id] = storage
        return Session(storage)

    async def open(self, metadata: SessionMetadata) -> Session:
        return Session(self._require_storage(metadata["id"]))

    async def list(self, options: Any = None) -> list[SessionMetadata]:
        return [await storage.get_metadata() for storage in self._sessions.values()]

    async def delete(self, metadata: SessionMetadata) -> None:
        self._sessions.pop(metadata["id"], None)

    async def fork(self, source: SessionMetadata, options: dict[str, Any] | None = None) -> Session:
        options = options or {}
        source_storage = self._require_storage(source["id"])
        session_id = options.get("id") or uuidv7()
        if session_id in self._sessions:
            raise SessionError("already_exists", f"Session already exists: {session_id}")
        metadata: SessionMetadata = {
            "id": session_id,
            "created_at": int(time.time() * 1000),
            "parent_session_id": options.get("parent_session_id") or source["id"],
        }
        storage = source_storage.fork(metadata, options)
        self._sessions[session_id] = storage
        return Session(storage)

    def _require_storage(self, session_id: str) -> InMemorySessionStorage:
        storage = self._sessions.get(session_id)
        if storage is None:
            raise SessionError("not_found", f"Session not found: {session_id}")
        return storage
