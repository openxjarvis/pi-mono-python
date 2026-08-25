"""Record storage — mirrors sqlite/storage/records.ts."""
from __future__ import annotations

from typing import Any, TypedDict

from pi_agent.harness.session.types import SessionError

from pi_session_backend_sqlite.sqlite.sql import join_sql_fragments, sql, sql_template
from pi_session_backend_sqlite.sqlite.types import SqliteDatabase


class RecordRow(TypedDict):
    session_id: str
    seq: int
    id: str
    lane: str
    run_id: str | None
    type: str
    op_kind: str | None
    timestamp: int
    payload: str


class NewRecordRow(TypedDict, total=False):
    seq: int
    id: str
    lane: str
    run_id: str
    type: str
    op_kind: str
    timestamp: int
    payload: str


def append_record_row(db: SqliteDatabase, session_id: str, record: NewRecordRow | dict[str, Any]) -> None:
    run_id = record.get("run_id", record.get("runId"))
    op_kind = record.get("op_kind", record.get("opKind"))
    sql(
        """INSERT INTO records
			(session_id, seq, id, lane, run_id, type, op_kind, timestamp, payload)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        session_id,
        record["seq"],
        record["id"],
        record["lane"],
        run_id,
        record["type"],
        op_kind,
        record["timestamp"],
        record["payload"],
    ).run(db)


def id_exists_in_records(db: SqliteDatabase, session_id: str, record_id: str) -> bool:
    return bool(
        sql(
            "SELECT 1 AS found FROM records WHERE session_id = ? AND id = ? LIMIT 1",
            session_id,
            record_id,
        ).get(db)
    )


def delete_record_rows(db: SqliteDatabase, session_id: str) -> None:
    sql("DELETE FROM records WHERE session_id = ?", session_id).run(db)


def read_record_rows(
    db: SqliteDatabase,
    session_id: str,
    query: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    query = query or {}
    predicates = [sql("session_id = ?", session_id)]
    if query.get("lane") is not None:
        predicates.append(sql("lane = ?", query["lane"]))
    if query.get("type") is not None:
        predicates.append(sql("type = ?", query["type"]))
    run_id = query.get("run_id", query.get("runId"))
    if run_id is not None:
        predicates.append(sql("run_id = ?", run_id))
    operation_kind = query.get("operation_kind", query.get("operationKind"))
    if operation_kind is not None:
        predicates.append(sql("op_kind = ?", operation_kind))
    after_seq = query.get("after_seq", query.get("afterSeq"))
    if after_seq is not None:
        predicates.append(sql("seq > ?", after_seq))
    direction = sql("ASC") if query.get("order") == "oldestFirst" else sql("DESC")
    limit = sql("") if query.get("limit") is None else sql(" LIMIT ?", query["limit"])
    return sql_template(
        "SELECT session_id, seq, id, lane, run_id, type, op_kind, timestamp, payload FROM records WHERE ",
        join_sql_fragments(predicates, " AND "),
        " ORDER BY seq ",
        direction,
        limit,
    ).all(db)


def read_open_operation_rows(
    db: SqliteDatabase,
    session_id: str,
    lane: str,
    _options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    lane_row = sql(
        "SELECT open_operation_id FROM lanes WHERE session_id = ? AND lane = ?",
        session_id,
        lane,
    ).get(db)
    if not lane_row or not lane_row.get("open_operation_id"):
        return []

    record = sql(
        """SELECT session_id, seq, id, lane, run_id, type, op_kind, timestamp, payload
		FROM records
		WHERE session_id = ?
			AND id = ?""",
        session_id,
        lane_row["open_operation_id"],
    ).get(db)
    if not record:
        raise SessionError(
            "storage",
            f"Lane {lane} points at missing open operation {lane_row['open_operation_id']}",
        )
    if record["lane"] != lane or record["type"] != "operation_started":
        raise SessionError(
            "storage",
            f"Lane {lane} points at invalid open operation {lane_row['open_operation_id']}",
        )
    return [record]
