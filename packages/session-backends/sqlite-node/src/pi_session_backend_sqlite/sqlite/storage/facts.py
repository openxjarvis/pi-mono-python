"""Fact storage — mirrors sqlite/storage/facts.ts."""
from __future__ import annotations

from typing import Any

from pi_session_backend_sqlite.sqlite.sql import param, sql, sql_template
from pi_session_backend_sqlite.sqlite.types import SqliteDatabase


def append_fact(
    db: SqliteDatabase,
    session_id: str,
    seq: int,
    kind: str,
    key: str | None,
    value: str | None,
) -> None:
    sql(
        "INSERT INTO facts (session_id, seq, kind, key, value) VALUES (?, ?, ?, ?, ?)",
        session_id,
        seq,
        kind,
        key,
        value,
    ).run(db)


def read_latest_fact(db: SqliteDatabase, session_id: str, kind: str, key: str | None) -> dict[str, Any] | None:
    return sql(
        """SELECT session_id, seq, kind, key, value
		FROM facts INDEXED BY idx_facts_session_kind_key_seq
		WHERE session_id = ? AND kind = ? AND key IS ?
		ORDER BY seq DESC
		LIMIT 1""",
        session_id,
        kind,
        key,
    ).get(db)


def read_latest_label_facts(db: SqliteDatabase, session_id: str) -> list[dict[str, Any]]:
    return sql(
        """SELECT f.key, f.value
		FROM facts AS f INDEXED BY idx_facts_session_kind_key_seq
		WHERE f.session_id = ?
			AND f.kind = 'label'
			AND f.value IS NOT NULL
			AND f.seq = (
				SELECT MAX(candidate.seq)
				FROM facts AS candidate INDEXED BY idx_facts_session_kind_key_seq
				WHERE candidate.session_id = f.session_id
					AND candidate.kind = f.kind
					AND candidate.key IS f.key
			)
		ORDER BY f.key""",
        session_id,
    ).all(db)


def read_fact_rows(
    db: SqliteDatabase,
    session_id: str,
    options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    options = options or {}
    after_seq = options.get("after_seq", options.get("afterSeq"))
    after = sql("") if after_seq is None else sql(" AND seq > ?", after_seq)
    limit = sql("") if options.get("limit") is None else sql(" LIMIT ?", options["limit"])
    return sql_template(
        "SELECT session_id, seq, kind, key, value FROM facts WHERE session_id = ",
        param(session_id),
        after,
        " ORDER BY seq",
        limit,
    ).all(db)


def delete_fact_rows(db: SqliteDatabase, session_id: str) -> None:
    sql("DELETE FROM facts WHERE session_id = ?", session_id).run(db)
