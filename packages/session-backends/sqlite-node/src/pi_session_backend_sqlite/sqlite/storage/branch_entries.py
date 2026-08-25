"""Derived branch cache storage — mirrors sqlite/storage/branch-entries.ts."""
from __future__ import annotations

import json
from typing import Any, TypedDict

from pi_agent.harness.session.types import SessionError

from pi_session_backend_sqlite.sqlite.sql import join_sql_fragments, param, sql, sql_template
from pi_session_backend_sqlite.sqlite.types import SqliteDatabase


class CachedBranch(TypedDict):
    branch_id: str
    leaf_seq: int


class CachedBranchEntryRow(TypedDict):
    session_id: str
    id: str
    entry_seq: int
    parent_id: str | None
    type: str
    timestamp: int
    payload: str


def read_cached_branch(db: SqliteDatabase, session_id: str, leaf_id: str) -> CachedBranch | None:
    membership = sql(
        """SELECT branch_id, entry_seq
		FROM branch_entries
		WHERE session_id = ? AND entry_id = ?
		ORDER BY branch_id
		LIMIT 1""",
        session_id,
        leaf_id,
    ).get(db)
    if not membership:
        return None
    return {"branch_id": membership["branch_id"], "leaf_seq": membership["entry_seq"]}


def query_cached_branch_rows(
    db: SqliteDatabase,
    session_id: str,
    branch: CachedBranch | dict[str, Any],
    query: dict[str, Any],
) -> list[dict[str, Any]]:
    oldest_first = query.get("order") == "oldestFirst"
    stop_predicates = []
    stop_at_type = query.get("stop_at_type", query.get("stopAtType"))
    stop_at_id = query.get("stop_at_id", query.get("stopAtId"))
    if stop_at_type is not None:
        stop_predicates.append(sql("stop.entry_type = ?", stop_at_type))
    if stop_at_id is not None:
        stop_predicates.append(sql("stop.entry_id = ?", stop_at_id))

    aggregate = sql("MIN") if oldest_first else sql("MAX")
    boundary_comparison = sql("<=") if oldest_first else sql(">=")
    cursor_comparison = sql(">") if oldest_first else sql("<")
    direction = sql("ASC") if oldest_first else sql("DESC")
    branch_id = branch.get("branch_id", branch.get("branchId"))
    leaf_seq = branch.get("leaf_seq", branch.get("leafSeq"))
    if stop_predicates:
        boundary = sql_template(
            "SELECT ",
            aggregate,
            "(stop.entry_seq) FROM branch_entries AS stop WHERE stop.session_id = ",
            param(session_id),
            " AND stop.branch_id = ",
            param(branch_id),
            " AND stop.entry_seq <= ",
            param(leaf_seq),
            " AND (",
            join_sql_fragments(stop_predicates, " OR "),
            ")",
        )
    else:
        boundary = sql("")

    predicates = [
        sql("b.session_id = ?", session_id),
        sql("b.branch_id = ?", branch_id),
        sql("b.entry_seq <= ?", leaf_seq),
    ]
    if stop_predicates:
        predicates.append(
            sql_template(
                "b.entry_seq ",
                boundary_comparison,
                " COALESCE((",
                boundary,
                "), ",
                leaf_seq if oldest_first else 0,
                ")",
            )
        )
    cursor = query.get("cursor")
    if cursor is not None:
        predicates.append(
            sql_template("b.entry_seq ", cursor_comparison, " ", cursor.get("after_seq", cursor.get("afterSeq")))
        )
    if query.get("type") is not None:
        predicates.append(sql("b.entry_type = ?", query["type"]))
    custom_type = query.get("custom_type", query.get("customType"))
    if custom_type is not None:
        predicates.append(sql("b.custom_type = ?", custom_type))
    limit = sql("") if query.get("limit") is None else sql(" LIMIT ?", query["limit"])

    return sql_template(
        """SELECT e.session_id, e.id, e.seq AS entry_seq, e.parent_id, e.type, e.timestamp, e.payload
		FROM branch_entries AS b
		JOIN entries AS e ON e.session_id = b.session_id AND e.id = b.entry_id
		WHERE """,
        join_sql_fragments(predicates, " AND "),
        " ORDER BY b.entry_seq ",
        direction,
        limit,
    ).all(db)


def delete_branch_entries(db: SqliteDatabase, session_id: str) -> None:
    sql("DELETE FROM branch_entries WHERE session_id = ?", session_id).run(db)


def insert_branch_entry(
    db: SqliteDatabase,
    session_id: str,
    branch_id: str,
    entry_id: str,
    entry_seq: int,
    entry_type: str,
    custom_type: str | None,
) -> None:
    sql(
        """INSERT INTO branch_entries
			(session_id, branch_id, entry_id, entry_seq, entry_type, custom_type)
			VALUES (?, ?, ?, ?, ?, ?)""",
        session_id,
        branch_id,
        entry_id,
        entry_seq,
        entry_type,
        custom_type,
    ).run(db)


def _custom_type_from_payload(row: dict[str, Any]) -> str | None:
    if row["type"] != "custom":
        return None
    try:
        payload = json.loads(row["payload"])
        if not isinstance(payload, dict):
            raise ValueError("Payload is not an object")
        custom_type = payload.get("custom_type", payload.get("customType"))
        if not isinstance(custom_type, str):
            raise ValueError("Invalid custom payload")
        return custom_type
    except Exception as error:
        raise SessionError(
            "invalid_entry",
            f"Invalid SQLite session entry {row['id']}: failed to decode entry {row['id']}",
            error if isinstance(error, Exception) else None,
        ) from error


def insert_branch_entries_for_path(db: SqliteDatabase, session_id: str, branch_id: str, leaf_id: str) -> None:
    path: list[dict[str, Any]] = []
    seen: set[str] = set()
    entry_id: str | None = leaf_id

    while entry_id is not None:
        if entry_id in seen:
            raise SessionError("invalid_entry", f"Entry parent cycle at {entry_id}")
        seen.add(entry_id)
        row = sql(
            """SELECT id, seq, parent_id, type, payload
			FROM entries
			WHERE session_id = ? AND id = ?""",
            session_id,
            entry_id,
        ).get(db)
        if not row:
            raise SessionError("invalid_entry", f"Entry {entry_id} not found")
        path.append(row)
        entry_id = row["parent_id"]

    for row in reversed(path):
        insert_branch_entry(db, session_id, branch_id, row["id"], row["seq"], row["type"], _custom_type_from_payload(row))


def read_branch_containing_entry(db: SqliteDatabase, session_id: str, entry_id: str) -> dict[str, Any] | None:
    row = sql(
        """SELECT b.branch_id, b.entry_seq
		FROM branch_entries AS b
		WHERE b.session_id = ? AND b.entry_id = ?
		ORDER BY b.branch_id
		LIMIT 1""",
        session_id,
        entry_id,
    ).get(db)
    if row is None:
        return None
    return {"branch_id": row["branch_id"], "entry_seq": row["entry_seq"]}


def copy_branch_entries_through_seq(
    db: SqliteDatabase,
    session_id: str,
    target_branch_id: str,
    source_branch_id: str,
    through_seq: int,
) -> None:
    sql(
        """INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq, entry_type, custom_type)
		SELECT session_id, ?, entry_id, entry_seq, entry_type, custom_type
		FROM branch_entries
		WHERE session_id = ? AND branch_id = ? AND entry_seq <= ?""",
        target_branch_id,
        session_id,
        source_branch_id,
        through_seq,
    ).run(db)
