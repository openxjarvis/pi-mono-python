"""Derived branch-read cache — mirrors sqlite/branch-cache.ts."""
from __future__ import annotations

from pi_agent.harness.session.types import SessionError
from pi_ai.utils.uuid import uuidv7

from pi_session_backend_sqlite.sqlite.sql import sql
from pi_session_backend_sqlite.sqlite.storage.branch_entries import (
    copy_branch_entries_through_seq,
    delete_branch_entries,
    insert_branch_entries_for_path,
    insert_branch_entry,
    read_branch_containing_entry,
)
from pi_session_backend_sqlite.sqlite.storage.branch_tips import (
    delete_branch_tips,
    insert_branch_tip,
    read_branch_tip_branch_id,
    update_branch_tip,
)
from pi_session_backend_sqlite.sqlite.types import SqliteDatabase


def delete_branch_cache(db: SqliteDatabase, session_id: str) -> None:
    delete_branch_tips(db, session_id)
    delete_branch_entries(db, session_id)


def rebuild_branch_cache(db: SqliteDatabase, session_id: str) -> None:
    tips = sql(
        """SELECT leaf.id
		FROM entries AS leaf
		WHERE leaf.session_id = ?
			AND NOT EXISTS (
				SELECT 1 FROM entries AS child WHERE child.session_id = leaf.session_id AND child.parent_id = leaf.id
			)
		ORDER BY leaf.seq""",
        session_id,
    ).all(db)
    delete_branch_cache(db, session_id)
    for tip in tips:
        build_cached_branch(db, session_id, tip["id"])


def build_cached_branch(db: SqliteDatabase, session_id: str, leaf_id: str) -> None:
    sql("SAVEPOINT build_branch_cache").exec(db)
    try:
        branch_id = uuidv7()
        insert_branch_entries_for_path(db, session_id, branch_id, leaf_id)
        insert_branch_tip(db, session_id, leaf_id, branch_id)
        sql("RELEASE SAVEPOINT build_branch_cache").exec(db)
    except Exception as error:
        try:
            sql("ROLLBACK TO SAVEPOINT build_branch_cache").exec(db)
            sql("RELEASE SAVEPOINT build_branch_cache").exec(db)
        except Exception:
            pass
        if isinstance(error, SessionError):
            raise
        raise SessionError(
            "storage",
            f"Failed to build SQLite branch cache at entry {leaf_id}",
            error if isinstance(error, Exception) else None,
        ) from error


def _extend_branch(
    db: SqliteDatabase,
    session_id: str,
    branch_id: str,
    parent_id: str,
    entry_id: str,
    entry_seq: int,
    entry_type: str,
    custom_type: str | None,
) -> None:
    insert_branch_entry(db, session_id, branch_id, entry_id, entry_seq, entry_type, custom_type)
    if not update_branch_tip(db, session_id, branch_id, parent_id, entry_id):
        raise SessionError("invalid_entry", f"Branch tip {parent_id} changed during append")


def append_entry_to_branch_cache(
    db: SqliteDatabase,
    session_id: str,
    entry_id: str,
    entry_seq: int,
    entry_type: str,
    custom_type: str | None,
    parent_id: str | None,
) -> None:
    if parent_id is None:
        branch_id = uuidv7()
        insert_branch_entry(db, session_id, branch_id, entry_id, entry_seq, entry_type, custom_type)
        insert_branch_tip(db, session_id, entry_id, branch_id)
        return

    tip_branch_id = read_branch_tip_branch_id(db, session_id, parent_id)
    if tip_branch_id is not None:
        _extend_branch(db, session_id, tip_branch_id, parent_id, entry_id, entry_seq, entry_type, custom_type)
        return

    source = read_branch_containing_entry(db, session_id, parent_id)
    if not source:
        raise SessionError("invalid_entry", f"Branch cache has no branch containing parent entry {parent_id}")

    branch_id = uuidv7()
    copy_branch_entries_through_seq(db, session_id, branch_id, source["branch_id"], source["entry_seq"])
    insert_branch_entry(db, session_id, branch_id, entry_id, entry_seq, entry_type, custom_type)
    insert_branch_tip(db, session_id, entry_id, branch_id)
