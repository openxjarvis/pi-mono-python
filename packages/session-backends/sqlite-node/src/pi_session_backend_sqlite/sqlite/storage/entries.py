"""Entry storage — mirrors sqlite/storage/entries.ts."""
from __future__ import annotations

from typing import Any, TypedDict

from pi_session_backend_sqlite.sqlite.sql import param, sql, sql_template
from pi_session_backend_sqlite.sqlite.types import SqliteDatabase

_ENTRY_META_KEYS = {"type", "id", "seq", "parent_id", "parentId", "timestamp"}


class EntryRow(TypedDict):
    session_id: str
    seq: int
    id: str
    parent_id: str | None
    type: str
    timestamp: int
    payload: str


class NewEntryRow(TypedDict):
    seq: int
    id: str
    parent_id: str | None
    type: str
    timestamp: int
    payload: str


def entry_payload(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if key not in _ENTRY_META_KEYS}


def insert_entry_row(db: SqliteDatabase, session_id: str, entry: NewEntryRow | dict[str, Any]) -> None:
    parent_id = entry.get("parent_id", entry.get("parentId"))
    sql(
        """INSERT INTO entries (session_id, id, seq, parent_id, type, timestamp, payload)
		VALUES (?, ?, ?, ?, ?, ?, ?)""",
        session_id,
        entry["id"],
        entry["seq"],
        parent_id,
        entry["type"],
        entry["timestamp"],
        entry["payload"],
    ).run(db)


def read_entry_row(db: SqliteDatabase, session_id: str, entry_id: str) -> dict[str, Any] | None:
    return sql(
        """SELECT session_id, seq, id, parent_id, type, timestamp, payload
		FROM entries
		WHERE session_id = ? AND id = ?""",
        session_id,
        entry_id,
    ).get(db)


def read_entry_rows(
    db: SqliteDatabase,
    session_id: str,
    options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    options = options or {}
    oldest_first = options.get("order") == "oldestFirst"
    after_seq = options.get("after_seq", options.get("afterSeq"))
    after = sql("") if after_seq is None else sql(" AND seq > ?", after_seq)
    cursor = options.get("cursor")
    if cursor is None:
        cursor_sql = sql("")
    else:
        cursor_after = cursor.get("after_seq", cursor.get("afterSeq"))
        cursor_sql = sql(" AND seq > ?", cursor_after) if oldest_first else sql(" AND seq < ?", cursor_after)
    type_sql = sql("") if options.get("type") is None else sql(" AND type = ?", options["type"])
    direction = sql("ASC") if oldest_first else sql("DESC")
    limit = sql("") if options.get("limit") is None else sql(" LIMIT ?", options["limit"])
    return sql_template(
        "SELECT session_id, seq, id, parent_id, type, timestamp, payload FROM entries WHERE session_id = ",
        param(session_id),
        after,
        cursor_sql,
        type_sql,
        " ORDER BY seq ",
        direction,
        limit,
    ).all(db)


def id_exists_in_entries(db: SqliteDatabase, session_id: str, entry_id: str) -> bool:
    return bool(
        sql(
            "SELECT 1 AS found FROM entries WHERE session_id = ? AND id = ? LIMIT 1",
            session_id,
            entry_id,
        ).get(db)
    )


def delete_entry_rows(db: SqliteDatabase, session_id: str) -> None:
    sql("DELETE FROM entries WHERE session_id = ?", session_id).run(db)
