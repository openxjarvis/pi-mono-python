"""Writer lease storage — mirrors sqlite/storage/writer-leases.ts."""
from __future__ import annotations

from typing import TypedDict

from pi_session_backend_sqlite.sqlite.sql import sql
from pi_session_backend_sqlite.sqlite.types import SqliteDatabase


class WriterLease(TypedDict):
    owner_id: str
    fence: int
    expires_at_ms: int


def acquire_writer_lease(
    db: SqliteDatabase,
    session_id: str,
    owner_id: str,
    now: int,
    expires_at_ms: int,
) -> WriterLease | None:
    row = sql(
        """INSERT INTO writer_leases (session_id, owner_id, fence, expires_at_ms)
		VALUES (?, ?, 1, ?)
		ON CONFLICT(session_id) DO UPDATE SET
			owner_id = excluded.owner_id,
			fence = writer_leases.fence + 1,
			expires_at_ms = excluded.expires_at_ms
		WHERE writer_leases.expires_at_ms <= ?
		RETURNING owner_id, fence, expires_at_ms""",
        session_id,
        owner_id,
        expires_at_ms,
        now,
    ).get(db)
    if row is None:
        return None
    return {
        "owner_id": row["owner_id"],
        "fence": row["fence"],
        "expires_at_ms": row["expires_at_ms"],
    }


def renew_writer_lease(
    db: SqliteDatabase,
    session_id: str,
    lease: WriterLease,
    now: int,
    expires_at_ms: int,
) -> bool:
    result = sql(
        """UPDATE writer_leases
		SET expires_at_ms = ?
		WHERE session_id = ?
			AND owner_id = ?
			AND fence = ?
			AND expires_at_ms > ?""",
        expires_at_ms,
        session_id,
        lease["owner_id"],
        lease["fence"],
        now,
    ).run(db)
    if result.changes == 1:
        lease["expires_at_ms"] = expires_at_ms
    return result.changes == 1


def release_writer_lease(db: SqliteDatabase, session_id: str, lease: WriterLease) -> None:
    sql(
        """DELETE FROM writer_leases
		WHERE session_id = ? AND owner_id = ? AND fence = ?""",
        session_id,
        lease["owner_id"],
        lease["fence"],
    ).run(db)


def delete_writer_lease(db: SqliteDatabase, session_id: str) -> None:
    sql("DELETE FROM writer_leases WHERE session_id = ?", session_id).run(db)
