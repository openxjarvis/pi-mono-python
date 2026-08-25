"""Session catalog storage — mirrors sqlite/storage/sessions.ts."""
from __future__ import annotations

import json
from typing import Any, TypedDict

from pi_agent.harness.session.session import assert_json_serializable
from pi_agent.harness.session.types import SessionError

from pi_session_backend_sqlite.sqlite.sql import sql, sql_template
from pi_session_backend_sqlite.sqlite.types import SqliteDatabase, SqliteSessionMetadata


class SessionRow(TypedDict):
    id: str
    created_at: int
    metadata: str | None
    cwd: str
    parent_session_id: str | None
    has_session_name: int
    session_name: str | None


class NewSessionRow(TypedDict, total=False):
    id: str
    created_at: int
    cwd: str
    parent_session_id: str
    metadata: dict[str, Any]


def _parse_metadata(metadata: str | None, session_id: str) -> dict[str, Any] | None:
    if metadata is None:
        return None
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError as error:
        raise SessionError("storage", f"Invalid SQLite session {session_id}: metadata is not valid JSON", error) from error
    if not isinstance(parsed, dict):
        raise SessionError("storage", f"Invalid SQLite session {session_id}: metadata must be an object")
    return parsed


def session_exists(db: SqliteDatabase, session_id: str) -> bool:
    return bool(sql("SELECT 1 AS found FROM sessions WHERE id = ?", session_id).get(db))


def _serialize_metadata(metadata: dict[str, Any] | None) -> str | None:
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise SessionError("invalid_payload", "SQLite session metadata must be an object")
    assert_json_serializable(metadata)
    return json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))


def insert_session_row(db: SqliteDatabase, session: NewSessionRow | dict[str, Any]) -> None:
    created_at = session.get("created_at", session.get("createdAt"))
    parent_session_id = session.get("parent_session_id", session.get("parentSessionId"))
    sql(
        """INSERT INTO sessions (id, created_at, metadata, cwd, parent_session_id)
		VALUES (?, ?, ?, ?, ?)""",
        session["id"],
        created_at,
        _serialize_metadata(session.get("metadata")),
        session["cwd"],
        parent_session_id,
    ).run(db)


_SESSION_SELECT = """SELECT s.id, s.created_at, s.metadata, s.cwd, s.parent_session_id,
			name_fact.seq IS NOT NULL AS has_session_name,
			name_fact.value AS session_name
		FROM sessions AS s
		LEFT JOIN facts AS name_fact
			ON name_fact.session_id = s.id
			AND name_fact.kind = 'name'
			AND name_fact.key IS NULL
			AND name_fact.seq = (
				SELECT MAX(f.seq)
				FROM facts AS f
				WHERE f.session_id = s.id AND f.kind = 'name' AND f.key IS NULL
			)"""


def read_session_row(db: SqliteDatabase, session_id: str) -> dict[str, Any] | None:
    return sql(f"{_SESSION_SELECT} WHERE s.id = ?", session_id).get(db)


def read_session_rows(db: SqliteDatabase, options: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    options = options or {}
    where = sql("") if options.get("cwd") is None else sql("WHERE s.cwd = ?", options["cwd"])
    return sql_template(_SESSION_SELECT, " ", where, " ORDER BY s.created_at DESC").all(db)


def delete_session_row(db: SqliteDatabase, session_id: str) -> None:
    sql("DELETE FROM sessions WHERE id = ?", session_id).run(db)


def _parse_session_name(value: str | None, session_id: str) -> str | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise SessionError("storage", f"Invalid SQLite session {session_id}: name is not valid JSON", error) from error
    if not isinstance(parsed, str):
        raise SessionError("storage", f"Invalid SQLite session {session_id}: name must be a string")
    return parsed


def decode_session_metadata(row: dict[str, Any], path: str) -> SqliteSessionMetadata:
    metadata = _parse_metadata(row.get("metadata"), row["id"])
    name = None if int(row.get("has_session_name") or 0) == 0 else _parse_session_name(row.get("session_name"), row["id"])
    decoded: SqliteSessionMetadata = {
        "id": row["id"],
        "created_at": row["created_at"],
        "cwd": row["cwd"],
        "path": path,
    }
    if name is not None:
        decoded["name"] = name
    parent_session_id = row.get("parent_session_id")
    if parent_session_id is not None:
        decoded["parent_session_id"] = parent_session_id
    if metadata is not None:
        decoded["metadata"] = metadata
    return decoded
