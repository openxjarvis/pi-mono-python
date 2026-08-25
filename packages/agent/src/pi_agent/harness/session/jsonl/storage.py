"""JSONL v4 session storage — mirrors harness/session/jsonl/storage.ts."""
from __future__ import annotations

import copy
import time
from typing import Any, Awaitable, Callable, TypeVar

from pi_agent.harness.session.jsonl.codec import (
    encode_header,
    encode_mutation,
    metadata_from_header,
    parse_header,
    parse_mutation,
)
from pi_agent.harness.session.jsonl.errors import JsonlDecodeError, file_result, invalid_file
from pi_agent.harness.session.jsonl.types import JsonlSessionMetadata, JsonlV4Header
from pi_agent.harness.session.state import SessionMutation, SessionState
from pi_agent.harness.session.types import (
    Entry,
    ForkOptions,
    LanePointer,
    LaneRecord,
    LogItem,
    NewRecord,
    OperationStartedRecord,
    ProvisionedEntry,
    SessionError,
    SessionStats,
)

T = TypeVar("T")


async def publish_file_atomically(fs: Any, destination_path: str, populate: Callable[[str], Awaitable[None]]) -> None:
    temp_path = f"{destination_path}.tmp"
    try:
        await populate(temp_path)
        file_result(await fs.rename_file(temp_path, destination_path), f"Failed to publish staged file {destination_path}")
    except Exception:
        await fs.remove(temp_path, {"force": True})
        raise


class JsonlSessionStorage:
    def __init__(self, fs: Any, metadata: JsonlSessionMetadata) -> None:
        self._fs = fs
        self._metadata = copy.deepcopy(metadata)
        self._state = SessionState()
        self._tail: Awaitable[None] = _resolved()

    @staticmethod
    async def create(fs: Any, path: str, header: JsonlV4Header) -> JsonlSessionStorage:
        file_result(await fs.write_file(path, encode_header(header)), f"Failed to initialize session {path}")
        file_info = file_result(await fs.file_info(path), f"Failed to read session metadata {path}")
        return JsonlSessionStorage(fs, metadata_from_header(header, path, file_info["mtime_ms"]))

    @staticmethod
    async def load(fs: Any, path: str) -> JsonlSessionStorage:
        content = file_result(await fs.read_text_file(path), f"Failed to read session {path}")
        physical_lines = content.split("\n")
        if physical_lines and physical_lines[-1] == "":
            physical_lines.pop()
        if not physical_lines or not physical_lines[0]:
            raise invalid_file(path, 1, JsonlDecodeError("schema", "is missing a header"))
        header_result = parse_header(physical_lines[0])
        if not header_result["ok"]:
            raise invalid_file(path, 1, header_result["error"])
        file_info = file_result(await fs.file_info(path), f"Failed to read session metadata {path}")
        storage = JsonlSessionStorage(fs, metadata_from_header(header_result["value"], path, file_info["mtime_ms"]))
        for index in range(1, len(physical_lines)):
            line = physical_lines[index]
            mutation_result = parse_mutation(line)
            if not mutation_result["ok"]:
                is_torn_tail = index == len(physical_lines) - 1 and mutation_result["error"].kind == "syntax"
                if is_torn_tail:
                    valid_prefix = "\n".join(physical_lines[:index]) + "\n"

                    async def _populate(temp_path: str, prefix: str = valid_prefix) -> None:
                        file_result(await fs.write_file(temp_path, prefix), f"Failed to stage torn-tail repair {path}")

                    await publish_file_atomically(fs, path, _populate)
                    return storage
                raise invalid_file(path, index + 1, mutation_result["error"])
            try:
                storage._apply_mutation(mutation_result["value"])
            except SessionError as error:
                if error.code == "invalid_entry":
                    raise invalid_file(path, index + 1, error) from error
                raise
        if not content.endswith("\n"):
            file_result(await fs.append_file(path, "\n"), f"Failed to repair unterminated session tail {path}")
        return storage

    async def fork(self, path: str, header: JsonlV4Header, options: ForkOptions) -> JsonlSessionStorage:
        mutations = self._state.create_fork_mutations(options)

        async def _populate(temp_path: str) -> None:
            target = await JsonlSessionStorage.create(self._fs, temp_path, header)
            for mutation in mutations:
                await target._append_mutation(mutation)
                target._apply_mutation(mutation)

        await publish_file_atomically(self._fs, path, _populate)
        return await JsonlSessionStorage.load(self._fs, path)

    async def drain(self) -> None:
        await self._tail

    async def get_metadata(self) -> JsonlSessionMetadata:
        return copy.deepcopy(self._metadata)

    async def get_lanes(self) -> list[LanePointer]:
        return self._state.get_lanes()

    def create_lane(self, lane: str, at: str | None) -> Awaitable[None]:
        return self._enqueue(lambda: self._create_lane(lane, at))

    async def _create_lane(self, lane: str, at: str | None) -> None:
        self._state.validate_new_lane(lane)
        self._state.validate_target(at)
        mutation: SessionMutation = {"kind": "lane", "seq": self._state.next_sequence, "lane": lane, "leaf_id": at}
        await self._append_mutation(mutation)
        self._apply_mutation(mutation)

    def move_lane(self, lane: str, to: str | None) -> Awaitable[None]:
        return self._enqueue(lambda: self._move_lane(lane, to))

    async def _move_lane(self, lane: str, to: str | None) -> None:
        self._state.require_lane(lane)
        self._state.validate_target(to)
        mutation: SessionMutation = {"kind": "lane", "seq": self._state.next_sequence, "lane": lane, "leaf_id": to}
        await self._append_mutation(mutation)
        self._apply_mutation(mutation)

    def append_entry(self, new_entry: ProvisionedEntry, lane: str) -> Awaitable[Entry]:
        return self._enqueue(lambda: self._append_entry(new_entry, lane))

    async def _append_entry(self, new_entry: ProvisionedEntry, lane: str) -> Entry:
        parent_id = self._state.require_lane(lane)
        self._state.validate_unused_id(new_entry["id"])
        entry = {
            **copy.deepcopy(new_entry),
            "parent_id": parent_id,
            "seq": self._state.next_sequence,
            "timestamp": int(time.time() * 1000),
        }
        mutation: SessionMutation = {"kind": "entry", "lane": lane, "entry": entry}
        await self._append_mutation(mutation)
        self._apply_mutation(mutation)
        return copy.deepcopy(entry)

    def append_record(self, new_record: NewRecord) -> Awaitable[LaneRecord]:
        return self._enqueue(lambda: self._append_record(new_record))

    async def _append_record(self, new_record: NewRecord) -> LaneRecord:
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
        mutation: SessionMutation = {"kind": "record", "record": record}
        await self._append_mutation(mutation)
        self._apply_mutation(mutation)
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

    def set_name(self, name: str | None) -> Awaitable[None]:
        return self._enqueue(lambda: self._set_name(name))

    async def _set_name(self, name: str | None) -> None:
        mutation: SessionMutation = {"kind": "fact", "seq": self._state.next_sequence, "fact": "name", "name": name}
        await self._append_mutation(mutation)
        self._apply_mutation(mutation)

    async def get_label(self, entry_id: str) -> str | None:
        return self._state.get_label(entry_id)

    def set_label(self, entry_id: str, label: str | None) -> Awaitable[None]:
        return self._enqueue(lambda: self._set_label(entry_id, label))

    async def _set_label(self, entry_id: str, label: str | None) -> None:
        self._state.validate_target(entry_id)
        mutation: SessionMutation = {
            "kind": "fact",
            "seq": self._state.next_sequence,
            "fact": "label",
            "target_id": entry_id,
            "label": label,
        }
        await self._append_mutation(mutation)
        self._apply_mutation(mutation)

    async def get_stats(self) -> SessionStats:
        return copy.deepcopy(self._state.get_stats())

    def _enqueue(self, operation: Callable[[], Awaitable[T]]) -> Awaitable[T]:
        async def _run() -> T:
            await self._tail
            return await operation()

        result = _run()

        async def _consume() -> None:
            try:
                await result
            except Exception:
                pass

        self._tail = _consume()
        return result

    async def _append_mutation(self, mutation: SessionMutation) -> None:
        file_result(
            await self._fs.append_file(self._metadata["path"], encode_mutation(mutation)),
            f"Failed to append session {self._metadata['path']}",
        )

    def _apply_mutation(self, mutation: SessionMutation) -> None:
        self._state.apply_mutation(mutation)


def _resolved() -> Awaitable[None]:
    async def _done() -> None:
        return None

    return _done()
