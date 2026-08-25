"""SQLite session repository — mirrors sqlite/repo.ts."""
from __future__ import annotations

import asyncio
import copy
import inspect
import json
import time
from collections.abc import Callable
from typing import Any, TypedDict

from pi_agent.harness.session.session import Session
from pi_agent.harness.session.types import (
    Entry,
    EntryQuery,
    ForkOptions,
    LaneRecord,
    LogItem,
    NewRecord,
    OperationStartedRecord,
    ProvisionedEntry,
    RecordQuery,
    SessionError,
    SessionStats,
)
from pi_agent.harness.types import Result
from pi_ai.utils.uuid import uuidv7
from pydantic import BaseModel

from pi_session_backend_sqlite.sqlite.branch_cache import (
    append_entry_to_branch_cache,
    build_cached_branch,
    delete_branch_cache,
    rebuild_branch_cache,
)
from pi_session_backend_sqlite.sqlite.migrations import apply_migrations
from pi_session_backend_sqlite.sqlite.sql import sql
from pi_session_backend_sqlite.sqlite.storage.branch_entries import query_cached_branch_rows, read_cached_branch
from pi_session_backend_sqlite.sqlite.storage.branch_tips import read_branch_tip_ids
from pi_session_backend_sqlite.sqlite.storage.entries import (
    delete_entry_rows,
    entry_payload,
    id_exists_in_entries,
    insert_entry_row,
    read_entry_row,
    read_entry_rows,
)
from pi_session_backend_sqlite.sqlite.storage.facts import (
    append_fact,
    delete_fact_rows,
    read_fact_rows,
    read_latest_fact,
    read_latest_label_facts,
)
from pi_session_backend_sqlite.sqlite.storage.lanes import (
    create_initial_lane,
    delete_lane_rows,
    finish_lane_operation,
    read_lane,
    read_lane_head,
    read_lane_move_rows,
    read_lanes,
    set_lane_leaf,
    start_lane_operation,
)
from pi_session_backend_sqlite.sqlite.storage.lanes import create_lane as insert_lane
from pi_session_backend_sqlite.sqlite.storage.lanes import move_lane as update_lane
from pi_session_backend_sqlite.sqlite.storage.records import (
    append_record_row,
    delete_record_rows,
    id_exists_in_records,
    read_open_operation_rows,
    read_record_rows,
)
from pi_session_backend_sqlite.sqlite.storage.session_sequences import (
    advance_sequence,
    create_sequence,
    delete_sequence,
    get_next_sequence,
    set_next_sequence,
)
from pi_session_backend_sqlite.sqlite.storage.session_stats import (
    add_usage_to_stats,
    create_stats,
    delete_stats,
    increment_message_count,
    read_stats,
)
from pi_session_backend_sqlite.sqlite.storage.sessions import (
    decode_session_metadata,
    delete_session_row,
    insert_session_row,
    read_session_row,
    read_session_rows,
    session_exists,
)
from pi_session_backend_sqlite.sqlite.storage.writer_leases import (
    WriterLease,
    acquire_writer_lease,
    delete_writer_lease,
    release_writer_lease,
    renew_writer_lease,
)
from pi_session_backend_sqlite.sqlite.types import (
    SqliteDatabase,
    SqliteDatabaseFactory,
    SqliteSessionCreateOptions,
    SqliteSessionListOptions,
    SqliteSessionMetadata,
    SqliteSessionRepositoryEnv,
)


class SqliteWriterLeaseOptions(TypedDict, total=False):
    ttl_ms: int
    heartbeat_interval_ms: int


class SqliteSessionRepositoryOptions(TypedDict, total=False):
    env: SqliteSessionRepositoryEnv
    sqlite: SqliteDatabaseFactory
    database_path: str
    writer_lease: SqliteWriterLeaseOptions


class _ResolvedWriterLeaseOptions(TypedDict):
    ttl_ms: int
    heartbeat_interval_ms: int


def _option(mapping: dict[str, Any], snake: str, camel: str, default: Any = None) -> Any:
    if snake in mapping:
        return mapping[snake]
    if camel in mapping:
        return mapping[camel]
    return default


def _resolve_writer_lease_options(options: dict[str, Any] | None) -> _ResolvedWriterLeaseOptions:
    options = options or {}
    ttl_ms = _option(options, "ttl_ms", "ttlMs", 30_000)
    heartbeat_interval_ms = _option(options, "heartbeat_interval_ms", "heartbeatIntervalMs", 10_000)
    if not isinstance(ttl_ms, int) or isinstance(ttl_ms, bool) or ttl_ms <= 0:
        raise ValueError("writerLease.ttlMs must be positive")
    if (
        not isinstance(heartbeat_interval_ms, int)
        or isinstance(heartbeat_interval_ms, bool)
        or heartbeat_interval_ms <= 0
        or heartbeat_interval_ms >= ttl_ms
    ):
        raise ValueError("writerLease.heartbeatIntervalMs must be positive and less than ttlMs")
    return {"ttl_ms": ttl_ms, "heartbeat_interval_ms": heartbeat_interval_ms}


def _active_writer_error(session_id: str) -> SessionError:
    return SessionError("storage", f"SQLite session {session_id} already has an active writer")


def _lost_writer_error(session_id: str) -> SessionError:
    return SessionError("storage", f"SQLite session {session_id} writer lease was lost")


def _claim_writer_lease(
    db: SqliteDatabase,
    session_id: str,
    options: _ResolvedWriterLeaseOptions,
) -> WriterLease:
    now = int(time.time() * 1000)
    lease = acquire_writer_lease(db, session_id, uuidv7(), now, now + options["ttl_ms"])
    if not lease:
        raise _active_writer_error(session_id)
    return lease


class SerialOperationQueue:
    def __init__(self) -> None:
        self._tail: asyncio.Future[Any] | None = None

    def enqueue(self, operation: Callable[[], Any]) -> asyncio.Future[Any]:
        previous = self._tail

        async def _run() -> Any:
            if previous is not None:
                try:
                    await previous
                except Exception:
                    pass
            result = operation()
            if inspect.isawaitable(result):
                return await result
            return result

        task = asyncio.ensure_future(_run())
        self._tail = task
        return task

    async def drain(self) -> None:
        if self._tail is not None:
            try:
                await self._tail
            except Exception:
                pass


def _result_or_throw(result: Result, message: str) -> Any:
    if not result.get("ok"):
        error = result["error"]
        code = "not_found" if getattr(error, "code", None) == "not_found" else "storage"
        raise SessionError(code, f"{message}: {error}", error if isinstance(error, Exception) else None)
    return result["value"]


def _get_parent_path(path: str) -> str:
    normalized = path.rstrip("\\/")
    last_slash = max(normalized.rfind("/"), normalized.rfind("\\"))
    if last_slash < 0:
        return "."
    if last_slash == 0:
        return normalized[:1]
    return normalized[:last_slash]


def _configure_sqlite_database(db: SqliteDatabase) -> None:
    sql("PRAGMA journal_mode=WAL").exec(db)
    sql("PRAGMA synchronous=FULL").exec(db)
    sql("PRAGMA busy_timeout=5000").exec(db)


def _json_dumps(value: Any) -> str:
    def _default(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return item.model_dump()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_default)


def _entry_row_from_cached(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "seq": row["entry_seq"], "type": row["type"]}


def _read_object_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(row["payload"])
    if not isinstance(payload, dict):
        raise ValueError("Payload is not an object")
    return payload


def _payload_get(payload: dict[str, Any], snake: str, camel: str) -> Any:
    if snake in payload:
        return payload[snake]
    return payload.get(camel)


def decode_entry(row: dict[str, Any]) -> Entry:
    try:
        payload = _read_object_payload(row)
        base = {
            "id": row["id"],
            "seq": row["seq"],
            "parent_id": row["parent_id"],
            "timestamp": row["timestamp"],
        }
        entry_type = row["type"]
        if entry_type == "message":
            message = payload.get("message")
            if not isinstance(message, dict):
                raise ValueError("Missing message")
            entry: dict[str, Any] = {**base, "type": "message", "message": message}
            if payload.get("terminate") is True:
                entry["terminate"] = True
            return entry
        if entry_type == "model_change":
            provider = payload.get("provider")
            model_id = _payload_get(payload, "model_id", "modelId")
            if not isinstance(provider, str) or not isinstance(model_id, str):
                raise ValueError("Invalid model_change payload")
            return {**base, "type": "model_change", "provider": provider, "model_id": model_id}
        if entry_type == "thinking_level_change":
            thinking_level = _payload_get(payload, "thinking_level", "thinkingLevel")
            if not isinstance(thinking_level, str):
                raise ValueError("Invalid thinking_level_change payload")
            return {**base, "type": "thinking_level_change", "thinking_level": thinking_level}
        if entry_type == "active_tools_change":
            active_tool_names = _payload_get(payload, "active_tool_names", "activeToolNames")
            if not isinstance(active_tool_names, list) or any(not isinstance(value, str) for value in active_tool_names):
                raise ValueError("Invalid active_tools_change payload")
            return {**base, "type": "active_tools_change", "active_tool_names": active_tool_names}
        if entry_type == "compaction":
            summary = payload.get("summary")
            retained_tail = _payload_get(payload, "retained_tail", "retainedTail")
            tokens_before = _payload_get(payload, "tokens_before", "tokensBefore")
            if not isinstance(summary, str) or not isinstance(retained_tail, list) or not isinstance(tokens_before, (int, float)):
                raise ValueError("Invalid compaction payload")
            compaction: dict[str, Any] = {
                **base,
                "type": "compaction",
                "summary": summary,
                "retained_tail": retained_tail,
                "tokens_before": tokens_before,
            }
            if "details" in payload:
                compaction["details"] = payload["details"]
            if "usage" in payload:
                compaction["usage"] = payload["usage"]
            return compaction
        if entry_type == "branch_summary":
            from_id = _payload_get(payload, "from_id", "fromId")
            summary = payload.get("summary")
            if not isinstance(from_id, str) or not isinstance(summary, str):
                raise ValueError("Invalid branch_summary payload")
            branch_summary: dict[str, Any] = {**base, "type": "branch_summary", "from_id": from_id, "summary": summary}
            if "details" in payload:
                branch_summary["details"] = payload["details"]
            if "usage" in payload:
                branch_summary["usage"] = payload["usage"]
            return branch_summary
        if entry_type == "custom":
            custom_type = _payload_get(payload, "custom_type", "customType")
            if not isinstance(custom_type, str):
                raise ValueError("Invalid custom payload")
            custom: dict[str, Any] = {**base, "type": "custom", "custom_type": custom_type}
            if "data" in payload:
                custom["data"] = payload["data"]
            return custom
        raise ValueError(f"Unknown entry type {entry_type}")
    except SessionError:
        raise
    except Exception as error:
        raise SessionError(
            "invalid_entry",
            f"Invalid SQLite session entry {row['id']}: failed to decode entry {row['id']}",
            error if isinstance(error, Exception) else None,
        ) from error


def _record_run_id(record: NewRecord) -> str | None:
    if record.get("type") == "operation_started":
        return record.get("id")
    return record.get("run_id", record.get("runId"))


def _record_op_kind(record: NewRecord) -> str | None:
    if record.get("type") == "operation_started":
        intent = record.get("intent") or {}
        return intent.get("kind")
    return None


def decode_record(row: dict[str, Any]) -> LaneRecord:
    try:
        parsed = json.loads(row["payload"])
        if not isinstance(parsed, dict):
            raise ValueError("Record payload is not an object")
        return {**parsed, "seq": row["seq"], "timestamp": row["timestamp"]}
    except Exception as error:
        raise SessionError(
            "storage",
            f"Invalid SQLite session record at sequence {row['seq']}: failed to decode payload",
            error if isinstance(error, Exception) else None,
        ) from error


def _validate_cached_branch_rows(rows: list[dict[str, Any]], query: dict[str, Any]) -> None:
    if not rows or query.get("type") is not None or _option(query, "custom_type", "customType") is not None:
        return
    path = sorted(rows, key=lambda item: item["entry_seq"])
    should_include_root = (
        _option(query, "stop_at_id", "stopAtId") is None
        and _option(query, "stop_at_type", "stopAtType") is None
        and query.get("cursor") is None
        and (query.get("order") == "oldestFirst" or query.get("limit") is None)
    )
    if should_include_root and path[0].get("parent_id") is not None:
        raise SessionError("invalid_entry", f"Entry {path[0].get('parent_id')} not found")
    for index in range(1, len(path)):
        previous = path[index - 1]
        current = path[index]
        if current.get("parent_id") != previous["id"]:
            raise SessionError("invalid_entry", f"Entry {current.get('parent_id')} not found")


def _matches_entry_query(entry: Entry, query: EntryQuery | dict[str, Any]) -> bool:
    custom_type = _option(query, "custom_type", "customType")
    cursor = query.get("cursor")
    if query.get("type") is not None and entry.get("type") != query.get("type"):
        return False
    if custom_type is not None and not (entry.get("type") == "custom" and entry.get("custom_type") == custom_type):
        return False
    if cursor is not None:
        after_seq = _option(cursor, "after_seq", "afterSeq")
        if query.get("order") == "oldestFirst":
            if entry["seq"] <= after_seq:
                return False
        elif entry["seq"] >= after_seq:
            return False
    return True


def _assert_unused_id(db: SqliteDatabase, session_id: str, entry_id: str) -> None:
    if id_exists_in_entries(db, session_id, entry_id) or id_exists_in_records(db, session_id, entry_id):
        raise SessionError("already_exists", f"ID already exists: {entry_id}")


def _require_session_row(db: SqliteDatabase, session_id: str) -> dict[str, Any]:
    row = read_session_row(db, session_id)
    if not row:
        raise SessionError("not_found", f"Session not found: {session_id}")
    return row


class SqliteSessionStorage:
    def __init__(
        self,
        db: SqliteDatabase,
        metadata: SqliteSessionMetadata,
        lease: WriterLease,
        lease_options: _ResolvedWriterLeaseOptions,
        on_release: Callable[[], None],
    ) -> None:
        self._db = db
        self._metadata = metadata
        self._lease = lease
        self._lease_options = lease_options
        self._on_release = on_release
        self._operations = SerialOperationQueue()
        self._heartbeat_handle: asyncio.TimerHandle | None = None
        self._lease_error: SessionError | None = None
        self._closing = False
        self._release_promise: asyncio.Future[None] | None = None
        self._schedule_heartbeat()

    async def release(self) -> None:
        if self._release_promise is None:
            self._release_promise = asyncio.ensure_future(self._finish_release())
        await self._release_promise

    async def _finish_release(self) -> None:
        self._closing = True
        if self._heartbeat_handle is not None:
            self._heartbeat_handle.cancel()
            self._heartbeat_handle = None
        try:
            await self._operations.enqueue(
                lambda: self._db.transaction(
                    lambda: release_writer_lease(self._db, self._metadata["id"], self._lease)
                )
            )
        finally:
            self._on_release()

    def _enqueue_write(self, operation: Callable[[], Any]) -> asyncio.Future[Any]:
        if self._closing:

            async def _closed() -> Any:
                raise SessionError("storage", f"SQLite session {self._metadata['id']} is closed")

            return asyncio.ensure_future(_closed())

        def _run() -> Any:
            if self._lease_error:
                raise self._lease_error

            def _txn() -> Any:
                now = int(time.time() * 1000)
                if not renew_writer_lease(
                    self._db,
                    self._metadata["id"],
                    self._lease,
                    now,
                    now + self._lease_options["ttl_ms"],
                ):
                    self._lease_error = _lost_writer_error(self._metadata["id"])
                    if self._heartbeat_handle is not None:
                        self._heartbeat_handle.cancel()
                        self._heartbeat_handle = None
                    raise self._lease_error
                return operation()

            return self._db.transaction(_txn)

        return self._operations.enqueue(_run)

    def _schedule_heartbeat(self) -> None:
        if self._closing or self._lease_error is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        def _fire() -> None:
            self._heartbeat_handle = None
            if self._closing or self._lease_error is not None:
                return
            loop.create_task(self._heartbeat())

        self._heartbeat_handle = loop.call_later(self._lease_options["heartbeat_interval_ms"] / 1000.0, _fire)

    async def _heartbeat(self) -> None:
        try:
            await self._operations.enqueue(self._heartbeat_once)
        except Exception:
            pass
        finally:
            self._schedule_heartbeat()

    def _heartbeat_once(self) -> None:
        if self._closing or self._lease_error is not None:
            return

        def _txn() -> None:
            now = int(time.time() * 1000)
            if not renew_writer_lease(
                self._db,
                self._metadata["id"],
                self._lease,
                now,
                now + self._lease_options["ttl_ms"],
            ):
                self._lease_error = _lost_writer_error(self._metadata["id"])

        self._db.transaction(_txn)

    async def get_metadata(self) -> SqliteSessionMetadata:
        return decode_session_metadata(_require_session_row(self._db, self._metadata["id"]), self._metadata["path"])

    def is_for_session(self, session_id: str) -> bool:
        return self._metadata["id"] == session_id

    async def get_lanes(self) -> list[dict[str, Any]]:
        return [{"lane": row["lane"], "leaf_id": row["leaf_id"]} for row in read_lanes(self._db, self._metadata["id"])]

    async def create_lane(self, lane: str, at: str | None) -> None:
        def _write() -> None:
            if read_lane(self._db, self._metadata["id"], lane):
                raise SessionError("already_exists", f"Lane already exists: {lane}")
            if at is not None and not read_entry_row(self._db, self._metadata["id"], at):
                raise SessionError("not_found", f"Entry not found: {at}")
            seq = get_next_sequence(self._db, self._metadata["id"])
            insert_lane(self._db, self._metadata["id"], seq, lane, at)
            advance_sequence(self._db, self._metadata["id"], seq)

        await self._enqueue_write(_write)

    async def move_lane(self, lane: str, to: str | None) -> None:
        def _write() -> None:
            if not read_lane(self._db, self._metadata["id"], lane):
                raise SessionError("invalid_lane", f"Lane not found: {lane}")
            if to is not None and not read_entry_row(self._db, self._metadata["id"], to):
                raise SessionError("not_found", f"Entry not found: {to}")
            seq = get_next_sequence(self._db, self._metadata["id"])
            update_lane(self._db, self._metadata["id"], seq, lane, to)
            advance_sequence(self._db, self._metadata["id"], seq)

        await self._enqueue_write(_write)

    async def append_entry(self, entry: ProvisionedEntry, lane: str) -> Entry:
        def _write() -> Entry:
            parent_id = read_lane_head(self._db, self._metadata["id"], lane)["leaf_id"]
            _assert_unused_id(self._db, self._metadata["id"], entry["id"])
            seq = get_next_sequence(self._db, self._metadata["id"])
            committed = {**entry, "parent_id": parent_id, "seq": seq, "timestamp": int(time.time() * 1000)}
            insert_entry_row(
                self._db,
                self._metadata["id"],
                {
                    "seq": seq,
                    "id": committed["id"],
                    "parent_id": committed["parent_id"],
                    "type": committed["type"],
                    "timestamp": committed["timestamp"],
                    "payload": _json_dumps(entry_payload(committed)),
                },
            )
            set_lane_leaf(self._db, self._metadata["id"], lane, committed["id"])
            custom_type = committed.get("custom_type", committed.get("customType")) if committed["type"] == "custom" else None
            append_entry_to_branch_cache(
                self._db,
                self._metadata["id"],
                committed["id"],
                seq,
                committed["type"],
                custom_type,
                committed["parent_id"],
            )
            if committed["type"] == "message":
                increment_message_count(self._db, self._metadata["id"])
            advance_sequence(self._db, self._metadata["id"], seq)
            return copy.deepcopy(committed)

        return await self._enqueue_write(_write)

    async def append_record(self, record: NewRecord) -> LaneRecord:
        def _write() -> LaneRecord:
            if not read_lane(self._db, self._metadata["id"], record["lane"]):
                raise SessionError("invalid_lane", f"Lane not found: {record['lane']}")
            _assert_unused_id(self._db, self._metadata["id"], record["id"])
            seq = get_next_sequence(self._db, self._metadata["id"])
            committed: LaneRecord = {**record, "seq": seq, "timestamp": int(time.time() * 1000)}
            if record.get("type") == "operation_started":
                start_lane_operation(self._db, self._metadata["id"], record["lane"], record["id"])
            append_record_row(
                self._db,
                self._metadata["id"],
                {
                    "seq": seq,
                    "id": record["id"],
                    "lane": record["lane"],
                    "run_id": _record_run_id(record),
                    "type": record["type"],
                    "op_kind": _record_op_kind(record),
                    "timestamp": committed["timestamp"],
                    "payload": _json_dumps(record),
                },
            )
            if record.get("type") == "operation_finished":
                finish_lane_operation(
                    self._db,
                    self._metadata["id"],
                    record["lane"],
                    _option(record, "run_id", "runId"),
                )
            if record.get("type") == "usage":
                add_usage_to_stats(self._db, self._metadata["id"], record.get("usage"))
            advance_sequence(self._db, self._metadata["id"], seq)
            return copy.deepcopy(committed)

        return await self._enqueue_write(_write)

    async def get_entry(self, entry_id: str) -> Entry | None:
        row = read_entry_row(self._db, self._metadata["id"], entry_id)
        return decode_entry(row) if row else None

    async def find_entries(self, query: EntryQuery | dict[str, Any] | None = None) -> list[Entry]:
        query = query or {}
        custom_type = _option(query, "custom_type", "customType")
        sql_type = query.get("type") if query.get("type") is not None else ("custom" if custom_type is not None else None)
        sql_limit = query.get("limit") if custom_type is None else None
        rows = read_entry_rows(
            self._db,
            self._metadata["id"],
            {
                "cursor": query.get("cursor"),
                "limit": sql_limit,
                "order": query.get("order"),
                "type": sql_type,
            },
        )
        entries = [entry for entry in (decode_entry(row) for row in rows) if _matches_entry_query(entry, query)]
        limit = query.get("limit")
        return entries if limit is None else entries[:limit]

    async def find_entries_on_branch(self, query: dict[str, Any]) -> list[Entry]:
        cached = read_cached_branch(self._db, self._metadata["id"], query["start"])
        if not cached:
            if not read_entry_row(self._db, self._metadata["id"], query["start"]):
                raise SessionError("not_found", f"Entry not found: {query['start']}")
            raise SessionError("invalid_entry", f"Branch cache missing entry {query['start']}")
        rows = query_cached_branch_rows(self._db, self._metadata["id"], cached, query)
        _validate_cached_branch_rows(rows, query)
        entries = [
            entry
            for entry in (decode_entry(_entry_row_from_cached(row)) for row in rows)
            if _matches_entry_query(entry, query)
        ]
        limit = query.get("limit")
        return entries if limit is None else entries[:limit]

    async def find_records(self, query: RecordQuery | dict[str, Any] | None = None) -> list[LaneRecord]:
        rows = read_record_rows(self._db, self._metadata["id"], query or {})
        return [decode_record(row) for row in rows]

    async def find_open_operations(
        self,
        lane: str,
        options: dict[str, Any] | None = None,
    ) -> list[OperationStartedRecord]:
        rows = read_open_operation_rows(self._db, self._metadata["id"], lane, options)
        records: list[OperationStartedRecord] = []
        for row in rows:
            record = decode_record(row)
            if record.get("type") != "operation_started":
                raise SessionError("storage", "Expected operation_started record")
            records.append(record)
        return records

    async def get_log(self, options: dict[str, Any] | None = None) -> list[LogItem]:
        options = options or {}
        after_seq = _option(options, "after_seq", "afterSeq", 0)
        limit = options.get("limit")
        entry_rows = read_entry_rows(
            self._db,
            self._metadata["id"],
            {"after_seq": after_seq, "order": "oldestFirst", "limit": limit},
        )
        record_rows = read_record_rows(
            self._db,
            self._metadata["id"],
            {"after_seq": after_seq, "order": "oldestFirst", "limit": limit},
        )
        lane_rows = read_lane_move_rows(self._db, self._metadata["id"], {"after_seq": after_seq, "limit": limit})
        fact_rows = read_fact_rows(self._db, self._metadata["id"], {"after_seq": after_seq, "limit": limit})

        log_rows: list[dict[str, Any]] = [
            *[
                {
                    "seq": row["seq"],
                    "decode": lambda current=row: {"kind": "entry", "seq": current["seq"], "entry": decode_entry(current)},
                }
                for row in entry_rows
            ],
            *[
                {
                    "seq": row["seq"],
                    "decode": lambda current=row: {
                        "kind": "record",
                        "seq": current["seq"],
                        "record": decode_record(current),
                    },
                }
                for row in record_rows
            ],
            *[
                {
                    "seq": row["seq"],
                    "decode": lambda current=row: {
                        "kind": "lane",
                        "seq": current["seq"],
                        "lane": current["lane"],
                        "leaf_id": current["leaf_id"],
                    },
                }
                for row in lane_rows
            ],
            *[
                {
                    "seq": row["seq"],
                    "decode": lambda current=row: (
                        {
                            "kind": "fact",
                            "seq": current["seq"],
                            "fact": "name",
                            "name": None if current["value"] is None else json.loads(current["value"]),
                        }
                        if current["kind"] == "name"
                        else {
                            "kind": "fact",
                            "seq": current["seq"],
                            "fact": "label",
                            "target_id": current["key"] or "",
                            "label": None if current["value"] is None else json.loads(current["value"]),
                        }
                    ),
                }
                for row in fact_rows
            ],
        ]
        log_rows.sort(key=lambda item: item["seq"])
        selected = log_rows if limit is None else log_rows[:limit]
        return [row["decode"]() for row in selected]

    async def get_name(self) -> str | None:
        row = read_latest_fact(self._db, self._metadata["id"], "name", None)
        if row is None or row.get("value") is None:
            return None
        return json.loads(row["value"])

    async def set_name(self, name: str | None) -> None:
        def _write() -> None:
            seq = get_next_sequence(self._db, self._metadata["id"])
            append_fact(
                self._db,
                self._metadata["id"],
                seq,
                "name",
                None,
                None if name is None else _json_dumps(name),
            )
            advance_sequence(self._db, self._metadata["id"], seq)

        await self._enqueue_write(_write)

    async def get_label(self, entry_id: str) -> str | None:
        row = read_latest_fact(self._db, self._metadata["id"], "label", entry_id)
        if row is None or row.get("value") is None:
            return None
        return json.loads(row["value"])

    async def set_label(self, entry_id: str, label: str | None) -> None:
        def _write() -> None:
            if not read_entry_row(self._db, self._metadata["id"], entry_id):
                raise SessionError("not_found", f"Entry not found: {entry_id}")
            seq = get_next_sequence(self._db, self._metadata["id"])
            append_fact(
                self._db,
                self._metadata["id"],
                seq,
                "label",
                entry_id,
                None if label is None else _json_dumps(label),
            )
            advance_sequence(self._db, self._metadata["id"], seq)

        await self._enqueue_write(_write)

    async def get_stats(self) -> SessionStats:
        return read_stats(self._db, self._metadata["id"])


def _claim_storage(
    db: SqliteDatabase,
    metadata: SqliteSessionMetadata,
    lease_options: _ResolvedWriterLeaseOptions,
    on_release: Callable[[], None],
) -> SqliteSessionStorage:
    _require_session_row(db, metadata["id"])

    def _claim() -> dict[str, Any]:
        lease = _claim_writer_lease(db, metadata["id"], lease_options)
        row = _require_session_row(db, metadata["id"])
        read_lanes(db, metadata["id"])
        return {"lease": lease, "row": row}

    claimed = db.transaction(_claim)
    return SqliteSessionStorage(
        db,
        decode_session_metadata(claimed["row"], metadata["path"]),
        claimed["lease"],
        lease_options,
        on_release,
    )


class SqliteSessionRepository:
    def __init__(self, options: SqliteSessionRepositoryOptions | dict[str, Any]) -> None:
        self._options = options
        self._lease_options = _resolve_writer_lease_options(
            _option(options, "writer_lease", "writerLease")
        )
        self._database_path: str | None = None
        self._database: SqliteDatabase | None = None
        self._database_promise: asyncio.Future[SqliteDatabase] | None = None
        self._operations = SerialOperationQueue()
        self._active_storages: set[SqliteSessionStorage] = set()

    async def _release_storages_for_session(self, session_id: str) -> None:
        for storage in list(self._active_storages):
            if storage.is_for_session(session_id):
                await storage.release()

    def _session_from_lease(
        self,
        db: SqliteDatabase,
        metadata: SqliteSessionMetadata,
        lease: WriterLease,
    ) -> Session:
        storage_holder: dict[str, SqliteSessionStorage] = {}

        def _on_release() -> None:
            self._active_storages.discard(storage_holder["storage"])

        storage = SqliteSessionStorage(db, metadata, lease, self._lease_options, _on_release)
        storage_holder["storage"] = storage
        self._active_storages.add(storage)
        return Session(storage)

    def _claim_session(self, db: SqliteDatabase, metadata: SqliteSessionMetadata) -> Session:
        active = next((storage for storage in self._active_storages if storage.is_for_session(metadata["id"])), None)
        if active:
            read_lanes(db, metadata["id"])
            return Session(active)
        storage_holder: dict[str, SqliteSessionStorage] = {}

        def _on_release() -> None:
            self._active_storages.discard(storage_holder["storage"])

        storage = _claim_storage(db, metadata, self._lease_options, _on_release)
        storage_holder["storage"] = storage
        self._active_storages.add(storage)
        return Session(storage)

    async def create(self, options: SqliteSessionCreateOptions | dict[str, Any]) -> Session:
        async def _create() -> Session:
            db = await self._get_database()
            path = await self._get_database_path()
            session_id = options.get("id") or uuidv7()
            if session_exists(db, session_id):
                raise SessionError("already_exists", f"Session already exists: {session_id}")
            created_at = int(time.time() * 1000)

            def _txn() -> WriterLease:
                insert_session_row(
                    db,
                    {
                        "id": session_id,
                        "created_at": created_at,
                        "cwd": options["cwd"],
                        "parent_session_id": _option(options, "parent_session_id", "parentSessionId"),
                        "metadata": options.get("metadata"),
                    },
                )
                create_sequence(db, session_id)
                create_stats(db, session_id)
                create_initial_lane(db, session_id)
                return _claim_writer_lease(db, session_id, self._lease_options)

            lease = db.transaction(_txn)
            row = _require_session_row(db, session_id)
            return self._session_from_lease(db, decode_session_metadata(row, path), lease)

        return await self._operations.enqueue(_create)

    async def open(self, metadata: SqliteSessionMetadata) -> Session:
        async def _open() -> Session:
            return self._claim_session(await self._get_database(), metadata)

        return await self._operations.enqueue(_open)

    async def repair_branch_cache(self, metadata: SqliteSessionMetadata) -> None:
        async def _repair() -> None:
            await self._release_storages_for_session(metadata["id"])
            db = await self._get_database()

            def _txn() -> None:
                lease = _claim_writer_lease(db, metadata["id"], self._lease_options)
                _require_session_row(db, metadata["id"])
                rebuild_branch_cache(db, metadata["id"])
                release_writer_lease(db, metadata["id"], lease)

            db.transaction(_txn)

        await self._operations.enqueue(_repair)

    async def list(self, options: SqliteSessionListOptions | dict[str, Any] | None = None) -> list[SqliteSessionMetadata]:
        options = options or {}

        async def _list() -> list[SqliteSessionMetadata]:
            path = await self._get_database_path()
            if not _result_or_throw(await self._options["env"].exists(path), f"Failed to check database {path}"):
                return []
            db = await self._get_database()
            rows = read_session_rows(db, options)
            return [decode_session_metadata(row, path) for row in rows]

        return await self._operations.enqueue(_list)

    async def delete(self, metadata: SqliteSessionMetadata) -> None:
        async def _delete() -> None:
            await self._release_storages_for_session(metadata["id"])
            db = await self._get_database()

            def _txn() -> None:
                if not session_exists(db, metadata["id"]):
                    delete_writer_lease(db, metadata["id"])
                    return
                _claim_writer_lease(db, metadata["id"], self._lease_options)
                delete_branch_cache(db, metadata["id"])
                delete_fact_rows(db, metadata["id"])
                delete_lane_rows(db, metadata["id"])
                delete_record_rows(db, metadata["id"])
                delete_entry_rows(db, metadata["id"])
                delete_writer_lease(db, metadata["id"])
                delete_stats(db, metadata["id"])
                delete_sequence(db, metadata["id"])
                delete_session_row(db, metadata["id"])

            db.transaction(_txn)

        await self._operations.enqueue(_delete)

    async def fork(
        self,
        source: SqliteSessionMetadata,
        options: ForkOptions | dict[str, Any],
    ) -> Session:
        async def _fork() -> Session:
            db = await self._get_database()
            path = await self._get_database_path()
            source_metadata = decode_session_metadata(_require_session_row(db, source["id"]), path)
            session_id = options.get("id") or uuidv7()
            if session_exists(db, session_id):
                raise SessionError("already_exists", f"Session already exists: {session_id}")

            entries: list[dict[str, Any]] = []
            lanes: list[dict[str, Any]] = []
            branch_tips: list[str] = []
            branch_fork_target_id: str | None = None

            if options.get("scope") == "tree":
                entries.extend(read_entry_rows(db, source["id"], {"order": "oldestFirst"}))
                lanes.extend(
                    {"lane": row["lane"], "leaf_id": row["leaf_id"]} for row in read_lanes(db, source["id"])
                )
                branch_tips.extend(read_branch_tip_ids(db, source["id"]))
            else:
                main = read_lane(db, source["id"], "main")
                if not main:
                    raise SessionError("invalid_lane", "Lane not found: main")
                selected_entry_id = _option(options, "entry_id", "entryId", main["leaf_id"])
                if selected_entry_id is not None:
                    target = read_entry_row(db, source["id"], selected_entry_id)
                    if not target or target["type"] != "message":
                        raise SessionError(
                            "invalid_fork_target",
                            f"Fork target is not a message entry: {selected_entry_id}",
                        )
                    position = options.get("position") or ("at" if _option(options, "entry_id", "entryId") is None else "before")
                    branch_fork_target_id = target["id"] if position == "at" else target["parent_id"]
                lanes.append({"lane": "main", "leaf_id": branch_fork_target_id})
                if branch_fork_target_id is not None:
                    cached = read_cached_branch(db, source["id"], branch_fork_target_id)
                    if not cached:
                        raise SessionError(
                            "invalid_fork_target",
                            f"Fork target is not on a cached branch: {branch_fork_target_id}",
                        )
                    rows = query_cached_branch_rows(db, source["id"], cached, {"order": "oldestFirst"})
                    entries.extend(_entry_row_from_cached(row) for row in rows)
                    branch_tips.append(branch_fork_target_id)

            copied_ids = {entry["id"] for entry in entries}
            latest_name = read_latest_fact(db, source["id"], "name", None)
            latest_labels = read_latest_label_facts(db, source["id"])
            labels_to_copy = [
                row
                for row in latest_labels
                if options.get("scope") == "tree" or (row.get("key") is not None and row["key"] in copied_ids)
            ]
            created_at = int(time.time() * 1000)
            metadata = options.get("metadata", source_metadata.get("metadata"))

            try:

                def _txn() -> WriterLease:
                    insert_session_row(
                        db,
                        {
                            "id": session_id,
                            "created_at": created_at,
                            "cwd": options["cwd"],
                            "parent_session_id": _option(options, "parent_session_id", "parentSessionId") or source["id"],
                            "metadata": metadata,
                        },
                    )
                    create_sequence(db, session_id)
                    create_stats(db, session_id, len([entry for entry in entries if entry["type"] == "message"]))

                    next_seq = 1

                    def allocate_seq() -> int:
                        nonlocal next_seq
                        current = next_seq
                        next_seq += 1
                        return current

                    for entry in entries:
                        insert_entry_row(
                            db,
                            session_id,
                            {
                                "seq": allocate_seq(),
                                "id": entry["id"],
                                "parent_id": entry["parent_id"],
                                "type": entry["type"],
                                "timestamp": entry["timestamp"],
                                "payload": entry["payload"],
                            },
                        )

                    if options.get("scope") == "tree":
                        for lane in lanes:
                            insert_lane(db, session_id, allocate_seq(), lane["lane"], lane["leaf_id"])
                    else:
                        create_initial_lane(db, session_id, "main", branch_fork_target_id)

                    if latest_name is not None and latest_name.get("value") is not None:
                        append_fact(db, session_id, allocate_seq(), "name", None, latest_name["value"])
                    for label in labels_to_copy:
                        append_fact(db, session_id, allocate_seq(), "label", label["key"], label["value"])

                    set_next_sequence(db, session_id, next_seq)
                    for tip in branch_tips:
                        build_cached_branch(db, session_id, tip)
                    return _claim_writer_lease(db, session_id, self._lease_options)

                lease = db.transaction(_txn)
            except SessionError:
                raise
            except Exception as error:
                raise SessionError(
                    "storage",
                    f"Failed to fork SQLite session {session_id}",
                    error if isinstance(error, Exception) else None,
                ) from error

            row = _require_session_row(db, session_id)
            return self._session_from_lease(db, decode_session_metadata(row, path), lease)

        return await self._operations.enqueue(_fork)

    async def close(self) -> None:
        await self._operations.drain()
        for storage in list(self._active_storages):
            await storage.release()
        if self._database is not None:
            self._database.close()
        self._database = None
        self._database_promise = None

    async def __aenter__(self) -> SqliteSessionRepository:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def _get_database_path(self) -> str:
        if self._database_path is None:
            database_path = _option(self._options, "database_path", "databasePath")
            self._database_path = _result_or_throw(
                await self._options["env"].absolute_path(database_path),
                f"Failed to resolve SQLite sessions database {database_path}",
            )
        return self._database_path

    async def _get_database(self) -> SqliteDatabase:
        if self._database_promise is None:
            self._database_promise = asyncio.ensure_future(self._open_database())
        self._database = await self._database_promise
        return self._database

    async def _open_database(self) -> SqliteDatabase:
        path = await self._get_database_path()
        _result_or_throw(
            await self._options["env"].create_dir(_get_parent_path(path), {"recursive": True}),
            f"Failed to create SQLite sessions directory {path}",
        )
        db = await self._options["sqlite"].open(path)
        try:
            _configure_sqlite_database(db)
            await apply_migrations(db)
            return db
        except Exception:
            db.close()
            raise
