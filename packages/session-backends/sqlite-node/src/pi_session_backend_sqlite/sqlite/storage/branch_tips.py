"""Branch tip storage — mirrors sqlite/storage/branch-tips.ts."""
from __future__ import annotations

from pi_session_backend_sqlite.sqlite.sql import sql
from pi_session_backend_sqlite.sqlite.types import SqliteDatabase


def read_branch_tip_ids(db: SqliteDatabase, session_id: str) -> list[str]:
    return [
        row["tip_id"]
        for row in sql("SELECT tip_id FROM branch_tips WHERE session_id = ? ORDER BY tip_id", session_id).all(db)
    ]


def read_branch_tip_branch_id(db: SqliteDatabase, session_id: str, tip_id: str) -> str | None:
    tip = sql(
        "SELECT branch_id FROM branch_tips WHERE session_id = ? AND tip_id = ?",
        session_id,
        tip_id,
    ).get(db)
    if tip is None:
        return None
    return tip["branch_id"]


def insert_branch_tip(db: SqliteDatabase, session_id: str, tip_id: str, branch_id: str) -> None:
    sql(
        "INSERT INTO branch_tips (session_id, tip_id, branch_id) VALUES (?, ?, ?)",
        session_id,
        tip_id,
        branch_id,
    ).run(db)


def update_branch_tip(
    db: SqliteDatabase,
    session_id: str,
    branch_id: str,
    old_tip_id: str,
    new_tip_id: str,
) -> bool:
    result = sql(
        """UPDATE branch_tips SET tip_id = ?
		WHERE session_id = ? AND branch_id = ? AND tip_id = ?""",
        new_tip_id,
        session_id,
        branch_id,
        old_tip_id,
    ).run(db)
    return result.changes == 1


def delete_branch_tips(db: SqliteDatabase, session_id: str) -> None:
    sql("DELETE FROM branch_tips WHERE session_id = ?", session_id).run(db)
