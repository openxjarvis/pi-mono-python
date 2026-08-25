"""Lane storage — mirrors sqlite/storage/lanes.ts."""
from __future__ import annotations

from typing import Any, TypedDict

from pi_agent.harness.session.types import SessionError

from pi_session_backend_sqlite.sqlite.sql import param, sql, sql_template
from pi_session_backend_sqlite.sqlite.types import SqliteDatabase


class LaneRow(TypedDict):
    session_id: str
    lane: str
    leaf_id: str | None
    open_operation_id: str | None


class LaneMoveRow(TypedDict):
    session_id: str
    seq: int
    lane: str
    leaf_id: str | None


def create_initial_lane(
    db: SqliteDatabase,
    session_id: str,
    lane: str = "main",
    leaf_id: str | None = None,
) -> None:
    sql(
        """INSERT INTO lanes (session_id, lane, leaf_id, open_operation_id)
		VALUES (?, ?, ?, NULL)""",
        session_id,
        lane,
        leaf_id,
    ).run(db)


def read_lanes(db: SqliteDatabase, session_id: str) -> list[dict[str, Any]]:
    rows = sql(
        """SELECT
			l.session_id,
			l.lane,
			l.leaf_id,
			l.open_operation_id,
			(l.leaf_id IS NULL OR EXISTS (
				SELECT 1 FROM entries AS e WHERE e.session_id = l.session_id AND e.id = l.leaf_id
			)) AS leaf_exists
		FROM lanes AS l
		WHERE l.session_id = ?
		ORDER BY l.lane""",
        session_id,
    ).all(db)
    for row in rows:
        if int(row["leaf_exists"] or 0) == 0:
            raise SessionError("storage", f"Lane {row['lane']} points at missing entry {row['leaf_id']}")
    return [
        {
            "session_id": row["session_id"],
            "lane": row["lane"],
            "leaf_id": row["leaf_id"],
            "open_operation_id": row["open_operation_id"],
        }
        for row in rows
    ]


def read_lane(db: SqliteDatabase, session_id: str, lane: str) -> dict[str, Any] | None:
    return sql(
        """SELECT session_id, lane, leaf_id, open_operation_id
		FROM lanes
		WHERE session_id = ? AND lane = ?""",
        session_id,
        lane,
    ).get(db)


def read_lane_head(db: SqliteDatabase, session_id: str, lane: str) -> dict[str, Any]:
    row = sql(
        """SELECT
			l.leaf_id,
			(l.leaf_id IS NULL OR EXISTS (
				SELECT 1 FROM entries AS e WHERE e.session_id = l.session_id AND e.id = l.leaf_id
			)) AS leaf_exists
		FROM lanes AS l
		WHERE l.session_id = ? AND l.lane = ?""",
        session_id,
        lane,
    ).get(db)
    if not row:
        raise SessionError("invalid_lane", f"Lane not found: {lane}")
    if int(row["leaf_exists"] or 0) == 0:
        raise SessionError("storage", f"Entry {row['leaf_id']} not found")
    return {"leaf_id": row["leaf_id"]}


def create_lane(db: SqliteDatabase, session_id: str, seq: int, lane: str, leaf_id: str | None) -> None:
    sql(
        """INSERT INTO lanes (session_id, lane, leaf_id, open_operation_id)
		VALUES (?, ?, ?, NULL)""",
        session_id,
        lane,
        leaf_id,
    ).run(db)
    _append_lane_move(db, session_id, seq, lane, leaf_id)


def move_lane(db: SqliteDatabase, session_id: str, seq: int, lane: str, leaf_id: str | None) -> None:
    result = sql(
        "UPDATE lanes SET leaf_id = ? WHERE session_id = ? AND lane = ?",
        leaf_id,
        session_id,
        lane,
    ).run(db)
    if result.changes != 1:
        raise SessionError("invalid_lane", f"Lane not found: {lane}")
    _append_lane_move(db, session_id, seq, lane, leaf_id)


def set_lane_leaf(db: SqliteDatabase, session_id: str, lane: str, leaf_id: str | None) -> None:
    result = sql(
        "UPDATE lanes SET leaf_id = ? WHERE session_id = ? AND lane = ?",
        leaf_id,
        session_id,
        lane,
    ).run(db)
    if result.changes != 1:
        raise SessionError("invalid_lane", f"Lane not found: {lane}")


def start_lane_operation(db: SqliteDatabase, session_id: str, lane: str, run_id: str) -> None:
    result = sql(
        """UPDATE lanes SET open_operation_id = ?
		WHERE session_id = ? AND lane = ? AND open_operation_id IS NULL""",
        run_id,
        session_id,
        lane,
    ).run(db)
    if result.changes == 1:
        return
    current = read_lane(db, session_id, lane)
    if not current:
        raise SessionError("invalid_lane", f"Lane not found: {lane}")
    raise SessionError("storage", f"Lane {lane} already has an open operation {current['open_operation_id']}")


def finish_lane_operation(db: SqliteDatabase, session_id: str, lane: str, run_id: str) -> None:
    sql(
        """UPDATE lanes SET open_operation_id = NULL
		WHERE session_id = ? AND lane = ? AND open_operation_id = ?""",
        session_id,
        lane,
        run_id,
    ).run(db)


def read_lane_move_rows(
    db: SqliteDatabase,
    session_id: str,
    options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    options = options or {}
    after_seq = options.get("after_seq", options.get("afterSeq"))
    after = sql("") if after_seq is None else sql(" AND seq > ?", after_seq)
    limit = sql("") if options.get("limit") is None else sql(" LIMIT ?", options["limit"])
    return sql_template(
        "SELECT session_id, seq, lane, leaf_id FROM lane_moves WHERE session_id = ",
        param(session_id),
        after,
        " ORDER BY seq",
        limit,
    ).all(db)


def delete_lane_rows(db: SqliteDatabase, session_id: str) -> None:
    sql("DELETE FROM lane_moves WHERE session_id = ?", session_id).run(db)
    sql("DELETE FROM lanes WHERE session_id = ?", session_id).run(db)


def _append_lane_move(
    db: SqliteDatabase,
    session_id: str,
    seq: int,
    lane: str,
    leaf_id: str | None,
) -> None:
    sql(
        "INSERT INTO lane_moves (session_id, seq, lane, leaf_id) VALUES (?, ?, ?, ?)",
        session_id,
        seq,
        lane,
        leaf_id,
    ).run(db)
