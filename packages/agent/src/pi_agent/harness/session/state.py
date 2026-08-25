"""In-memory session mutation state — mirrors harness/session/state.ts."""
from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any, Callable, Literal

from pi_agent.harness.session.session import assert_valid_cursor, assert_valid_limit
from pi_agent.harness.session.types import (
    Entry,
    EntryOrder,
    ForkOptions,
    LanePointer,
    LaneRecord,
    LogItem,
    OperationStartedRecord,
    SessionError,
    SessionStats,
)

SessionMutation = dict[str, Any]
InvalidMutation = Callable[[str], Any]


def _invalid_mutation(message: str) -> None:
    raise SessionError("invalid_entry", f"Invalid session mutation: {message}")


def _ordered(items: list[Any], order: EntryOrder | None) -> Iterator[Any]:
    if order == "oldestFirst":
        yield from items
        return
    for index in range(len(items) - 1, -1, -1):
        yield items[index]


def _usage_get(usage: Any, snake: str, camel: str, default: Any = 0) -> Any:
    if usage is None:
        return default
    if hasattr(usage, snake):
        return getattr(usage, snake)
    if isinstance(usage, dict):
        if snake in usage:
            return usage[snake]
        if camel in usage:
            return usage[camel]
    return default


class SessionState:
    def __init__(self) -> None:
        self._sequence = 0
        self._used_ids: set[str] = set()
        self._entries: list[Entry] = []
        self._entries_by_id: dict[str, Entry] = {}
        self._records: list[LaneRecord] = []
        self._open_operations_by_lane: dict[str, dict[str, OperationStartedRecord]] = {}
        self._lanes: dict[str, str | None] = {"main": None}
        self._log: list[LogItem] = []
        self._stats: SessionStats = {
            "message_count": 0,
            "cached_tokens": 0,
            "uncached_tokens": 0,
            "total_tokens": 0,
            "cost_total": 0,
        }
        self._name: str | None = None
        self._labels: dict[str, str] = {}

    @property
    def next_sequence(self) -> int:
        return self._sequence + 1

    def get_lanes(self) -> list[LanePointer]:
        return [{"lane": lane, "leaf_id": leaf_id} for lane, leaf_id in self._lanes.items()]

    def require_lane(self, lane: str) -> str | None:
        if lane not in self._lanes:
            raise SessionError("invalid_lane", f"Lane not found: {lane}")
        return self._lanes[lane]

    def validate_new_lane(self, lane: str) -> None:
        if lane in self._lanes:
            raise SessionError("already_exists", f"Lane already exists: {lane}")

    def validate_target(self, target_id: str | None) -> None:
        if target_id is not None and target_id not in self._entries_by_id:
            raise SessionError("not_found", f"Entry not found: {target_id}")

    def validate_unused_id(self, entry_id: str) -> None:
        if entry_id in self._used_ids:
            raise SessionError("already_exists", f"Session id already exists: {entry_id}")

    def apply_mutation(self, mutation: SessionMutation, invalid: InvalidMutation | None = None) -> None:
        invalid = invalid or _invalid_mutation
        kind = mutation["kind"]
        if kind == "entry":
            seq = mutation["entry"]["seq"]
        elif kind == "record":
            seq = mutation["record"]["seq"]
        else:
            seq = mutation["seq"]
        if seq != self._sequence + 1:
            invalid(f"has non-consecutive seq {seq}")

        if kind == "entry":
            entry = mutation["entry"]
            if entry["id"] in self._used_ids:
                invalid(f"contains duplicate id {entry['id']}")
            if mutation.get("lane") is not None:
                if mutation["lane"] not in self._lanes:
                    invalid(f"references missing lane {mutation['lane']}")
                if entry.get("parent_id") != self._lanes[mutation["lane"]]:
                    invalid("does not chain to the lane leaf")
            if entry.get("parent_id") is not None and entry["parent_id"] not in self._entries_by_id:
                invalid(f"references missing parent {entry['parent_id']}")
            self._sequence = seq
            self._used_ids.add(entry["id"])
            self._entries.append(entry)
            self._entries_by_id[entry["id"]] = entry
            if mutation.get("lane") is not None:
                self._lanes[mutation["lane"]] = entry["id"]
            self._log.append({"kind": "entry", "seq": seq, "entry": entry})
            if entry.get("type") == "message":
                self._stats["message_count"] += 1
            return

        if kind == "record":
            record = mutation["record"]
            if record["lane"] not in self._lanes:
                invalid(f"references missing lane {record['lane']}")
            if record["id"] in self._used_ids:
                invalid(f"contains duplicate id {record['id']}")
            self._sequence = seq
            self._used_ids.add(record["id"])
            self._records.append(record)
            if record.get("type") == "operation_started":
                self._open_operations_by_lane.setdefault(record["lane"], {})[record["id"]] = record
            elif record.get("type") == "operation_finished":
                open_ops = self._open_operations_by_lane.get(record["lane"])
                if open_ops is not None:
                    open_ops.pop(record.get("run_id"), None)
            self._log.append({"kind": "record", "seq": seq, "record": record})
            if record.get("type") == "usage":
                usage = record.get("usage")
                cost = _usage_get(usage, "cost", "cost", {})
                self._stats["cached_tokens"] += _usage_get(usage, "cache_read", "cacheRead")
                self._stats["uncached_tokens"] += _usage_get(usage, "input", "input") + _usage_get(
                    usage, "cache_write", "cacheWrite"
                )
                self._stats["total_tokens"] += _usage_get(usage, "total_tokens", "totalTokens")
                if hasattr(cost, "total"):
                    self._stats["cost_total"] += cost.total
                elif isinstance(cost, dict):
                    self._stats["cost_total"] += cost.get("total", 0)
            return

        if kind == "lane":
            if mutation.get("leaf_id") is not None and mutation["leaf_id"] not in self._entries_by_id:
                invalid(f"references missing lane target {mutation['leaf_id']}")
            self._sequence = seq
            self._lanes[mutation["lane"]] = mutation.get("leaf_id")
            self._log.append(
                {"kind": "lane", "seq": seq, "lane": mutation["lane"], "leaf_id": mutation.get("leaf_id")}
            )
            return

        if mutation.get("fact") == "label" and mutation.get("target_id") not in self._entries_by_id:
            invalid(f"references missing label target {mutation['target_id']}")
        self._sequence = seq
        if mutation.get("fact") == "name":
            self._name = mutation.get("name")
            self._log.append({"kind": "fact", "seq": seq, "fact": "name", "name": mutation.get("name")})
        else:
            if mutation.get("label") is None:
                self._labels.pop(mutation["target_id"], None)
            else:
                self._labels[mutation["target_id"]] = mutation["label"]
            self._log.append(
                {
                    "kind": "fact",
                    "seq": seq,
                    "fact": "label",
                    "target_id": mutation["target_id"],
                    "label": mutation.get("label"),
                }
            )

    def get_entry(self, entry_id: str) -> Entry | None:
        return self._entries_by_id.get(entry_id)

    def find_entries(self, query: dict[str, Any] | None = None) -> list[Entry]:
        query = query or {}
        assert_valid_limit(query.get("limit"))
        cursor = query.get("cursor") or {}
        after_seq = cursor.get("after_seq", cursor.get("afterSeq"))
        assert_valid_cursor(after_seq)
        results: list[Entry] = []
        for entry in _ordered(self._entries, query.get("order")):
            if not self._matches_entry_query(entry, query):
                continue
            results.append(entry)
            if query.get("limit") is not None and len(results) == query["limit"]:
                break
        return results

    def find_entries_on_branch(self, query: dict[str, Any]) -> list[Entry]:
        assert_valid_limit(query.get("limit"))
        cursor = query.get("cursor") or {}
        assert_valid_cursor(cursor.get("after_seq", cursor.get("afterSeq")))
        results: list[Entry] = []
        if query.get("order") == "oldestFirst":
            walked = list(self._walk_to_root(query["start"]))
            walked.reverse()
            for entry in walked:
                reached_bound = entry["id"] == query.get("stop_at_id") or entry.get("type") == query.get("stop_at_type")
                if self._matches_entry_query(entry, query):
                    results.append(entry)
                if reached_bound or (query.get("limit") is not None and len(results) == query["limit"]):
                    break
        else:
            for entry in self._walk_to_root(query["start"], query):
                if self._matches_entry_query(entry, query):
                    results.append(entry)
                if query.get("limit") is not None and len(results) == query["limit"]:
                    break
        return results

    def find_records(self, query: dict[str, Any] | None = None) -> list[LaneRecord]:
        query = query or {}
        assert_valid_limit(query.get("limit"))
        assert_valid_cursor(query.get("after_seq"))
        results: list[LaneRecord] = []
        for record in _ordered(self._records, query.get("order")):
            if not self._matches_record_query(record, query):
                continue
            results.append(record)
            if query.get("limit") is not None and len(results) == query["limit"]:
                break
        return results

    def find_open_operations(self, lane: str, options: dict[str, Any] | None = None) -> list[OperationStartedRecord]:
        options = options or {}
        assert_valid_limit(options.get("limit"))
        open_ops = self._open_operations_by_lane.get(lane)
        values = list(reversed(list(open_ops.values()))) if open_ops else []
        limit = options.get("limit")
        return values if limit is None else values[:limit]

    def get_log(self, options: dict[str, Any] | None = None) -> list[LogItem]:
        options = options or {}
        assert_valid_limit(options.get("limit"))
        assert_valid_cursor(options.get("after_seq"))
        results: list[LogItem] = []
        for item in self._log:
            if options.get("after_seq") is not None and item["seq"] <= options["after_seq"]:
                continue
            results.append(item)
            if options.get("limit") is not None and len(results) == options["limit"]:
                break
        return results

    def get_name(self) -> str | None:
        return self._name

    def get_label(self, entry_id: str) -> str | None:
        return self._labels.get(entry_id)

    def get_stats(self) -> SessionStats:
        return self._stats

    def create_fork_mutations(self, options: ForkOptions) -> list[SessionMutation]:
        if options.get("scope") == "tree":
            copied_entries = self.find_entries({"order": "oldestFirst"})
            fork_lanes = self.get_lanes()
        else:
            selected_entry_id = options.get("entry_id", self.require_lane("main"))
            target_id: str | None = None
            if selected_entry_id is not None:
                entry = self.get_entry(selected_entry_id)
                if not entry or entry.get("type") != "message":
                    raise SessionError("invalid_fork_target", f"Fork target is not a message entry: {selected_entry_id}")
                position = options.get("position") or ("at" if options.get("entry_id") is None else "before")
                target_id = entry["id"] if position == "at" else entry.get("parent_id")
            copied_entries = [] if target_id is None else self.find_entries_on_branch(
                {"start": target_id, "order": "oldestFirst"}
            )
            fork_lanes = [{"lane": "main", "leaf_id": target_id}]

        mutations: list[SessionMutation] = []
        sequence = 1
        for source_entry in copied_entries:
            cloned = copy.deepcopy(source_entry)
            cloned["seq"] = sequence
            sequence += 1
            mutations.append({"kind": "entry", "entry": cloned})
        for pointer in fork_lanes:
            mutations.append(
                {"kind": "lane", "seq": sequence, "lane": pointer["lane"], "leaf_id": pointer["leaf_id"]}
            )
            sequence += 1
        if self._name is not None:
            mutations.append({"kind": "fact", "seq": sequence, "fact": "name", "name": self._name})
            sequence += 1
        for entry in copied_entries:
            label = self._labels.get(entry["id"])
            if label is not None:
                mutations.append(
                    {"kind": "fact", "seq": sequence, "fact": "label", "target_id": entry["id"], "label": label}
                )
                sequence += 1
        return mutations

    def _walk_to_root(
        self,
        start: str | None,
        bounds: dict[str, Any] | None = None,
    ) -> Iterator[Entry]:
        if start is None:
            return
        visited: set[str] = set()
        current = self._entries_by_id.get(start)
        if current is None:
            raise SessionError("not_found", f"Entry not found: {start}")
        while current:
            if current["id"] in visited:
                raise SessionError("invalid_entry", f"Session branch contains a cycle at {current['id']}")
            visited.add(current["id"])
            yield current
            if (
                current["id"] == (bounds or {}).get("stop_at_id")
                or current.get("type") == (bounds or {}).get("stop_at_type")
                or current.get("parent_id") is None
            ):
                break
            parent_id = current["parent_id"]
            current = self._entries_by_id.get(parent_id)
            if current is None:
                raise SessionError("invalid_entry", f"Entry not found: {parent_id}")

    def _matches_entry_query(self, entry: Entry, query: dict[str, Any]) -> bool:
        if query.get("type") is not None and entry.get("type") != query["type"]:
            return False
        if query.get("custom_type") is not None and not (
            entry.get("type") == "custom" and entry.get("custom_type") == query["custom_type"]
        ):
            return False
        cursor = query.get("cursor")
        if cursor is not None:
            after_seq = cursor.get("after_seq", cursor.get("afterSeq"))
            if after_seq is not None:
                if query.get("order") == "oldestFirst":
                    if entry["seq"] <= after_seq:
                        return False
                elif entry["seq"] >= after_seq:
                    return False
        return True

    def _matches_record_query(self, record: LaneRecord, query: dict[str, Any]) -> bool:
        if query.get("lane") is not None and record.get("lane") != query["lane"]:
            return False
        if query.get("type") is not None and record.get("type") != query["type"]:
            return False
        if query.get("run_id") is not None:
            if record.get("type") == "operation_started":
                if record.get("id") != query["run_id"]:
                    return False
            elif record.get("run_id") != query["run_id"]:
                return False
        if query.get("operation_kind") is not None:
            if not (
                record.get("type") == "operation_started"
                and (record.get("intent") or {}).get("kind") == query["operation_kind"]
            ):
                return False
        if query.get("after_seq") is not None and record["seq"] <= query["after_seq"]:
            return False
        return True
