"""Session stats storage — mirrors sqlite/storage/session-stats.ts."""
from __future__ import annotations

from typing import Any

from pi_agent.harness.session.types import SessionError, SessionStats

from pi_session_backend_sqlite.sqlite.sql import sql
from pi_session_backend_sqlite.sqlite.types import SqliteDatabase


def _usage_field(usage: Any, snake: str, camel: str, default: Any = 0) -> Any:
    if usage is None:
        return default
    if isinstance(usage, dict):
        if snake in usage:
            return usage[snake]
        if camel in usage:
            return usage[camel]
        return default
    if hasattr(usage, snake):
        return getattr(usage, snake)
    return getattr(usage, camel, default)


def create_stats(db: SqliteDatabase, session_id: str, message_count: int = 0) -> None:
    sql(
        """INSERT INTO session_stats
			(session_id, message_count, cached_tokens, uncached_tokens, total_tokens, cost_total)
			VALUES (?, ?, 0, 0, 0, 0)""",
        session_id,
        message_count,
    ).run(db)


def read_stats(db: SqliteDatabase, session_id: str) -> SessionStats:
    row = sql(
        """SELECT session_id, message_count, cached_tokens, uncached_tokens, total_tokens, cost_total
		FROM session_stats
		WHERE session_id = ?""",
        session_id,
    ).get(db)
    if not row:
        raise SessionError("storage", f"Missing stats row for session {session_id}")
    return {
        "message_count": row["message_count"],
        "cached_tokens": row["cached_tokens"],
        "uncached_tokens": row["uncached_tokens"],
        "total_tokens": row["total_tokens"],
        "cost_total": row["cost_total"],
    }


def increment_message_count(db: SqliteDatabase, session_id: str) -> None:
    result = sql(
        "UPDATE session_stats SET message_count = message_count + 1 WHERE session_id = ?",
        session_id,
    ).run(db)
    if result.changes != 1:
        raise SessionError("storage", f"Missing stats row for session {session_id}")


def add_usage_to_stats(db: SqliteDatabase, session_id: str, usage: Any) -> None:
    cost = _usage_field(usage, "cost", "cost", {})
    if hasattr(cost, "total"):
        cost_total = cost.total
    elif isinstance(cost, dict):
        cost_total = cost.get("total", 0)
    else:
        cost_total = 0
    result = sql(
        """UPDATE session_stats
		SET cached_tokens = cached_tokens + ?,
			uncached_tokens = uncached_tokens + ?,
			total_tokens = total_tokens + ?,
			cost_total = cost_total + ?
		WHERE session_id = ?""",
        _usage_field(usage, "cache_read", "cacheRead"),
        _usage_field(usage, "input", "input") + _usage_field(usage, "cache_write", "cacheWrite"),
        _usage_field(usage, "total_tokens", "totalTokens"),
        cost_total,
        session_id,
    ).run(db)
    if result.changes != 1:
        raise SessionError("storage", f"Missing stats row for session {session_id}")


def delete_stats(db: SqliteDatabase, session_id: str) -> None:
    sql("DELETE FROM session_stats WHERE session_id = ?", session_id).run(db)
