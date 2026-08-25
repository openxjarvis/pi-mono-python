"""Session sequence storage — mirrors sqlite/storage/session-sequences.ts."""
from __future__ import annotations

from pi_agent.harness.session.types import SessionError

from pi_session_backend_sqlite.sqlite.sql import sql
from pi_session_backend_sqlite.sqlite.types import SqliteDatabase


def create_sequence(db: SqliteDatabase, session_id: str, next_seq: int = 1) -> None:
    sql("INSERT INTO session_sequences (session_id, next_seq) VALUES (?, ?)", session_id, next_seq).run(db)


def get_next_sequence(db: SqliteDatabase, session_id: str) -> int:
    sequence_row = sql("SELECT next_seq FROM session_sequences WHERE session_id = ?", session_id).get(db)
    if not sequence_row:
        raise SessionError("storage", f"Missing sequence row for session {session_id}")
    return sequence_row["next_seq"]


def set_next_sequence(db: SqliteDatabase, session_id: str, next_seq: int) -> None:
    sql("UPDATE session_sequences SET next_seq = ? WHERE session_id = ?", next_seq, session_id).run(db)


def advance_sequence(db: SqliteDatabase, session_id: str, seq: int) -> None:
    set_next_sequence(db, session_id, seq + 1)


def delete_sequence(db: SqliteDatabase, session_id: str) -> None:
    sql("DELETE FROM session_sequences WHERE session_id = ?", session_id).run(db)
