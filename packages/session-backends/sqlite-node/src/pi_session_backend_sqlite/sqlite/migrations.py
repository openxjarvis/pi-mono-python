"""SQLite schema migrations — mirrors sqlite/migrations.ts."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from pi_session_backend_sqlite.sqlite.sql import sql
from pi_session_backend_sqlite.sqlite.types import SqliteDatabase

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


class SqliteMigration(TypedDict):
    id: str
    order: int
    sql: str


def _load_migration_sql(relative_path: str) -> str:
    return (_MIGRATIONS_DIR / relative_path).read_text(encoding="utf-8")


async def load_migrations() -> list[SqliteMigration]:
    return [
        {
            "id": "001_initial.sql",
            "order": 1,
            "sql": _load_migration_sql("001_initial.sql"),
        },
    ]


def _ensure_migrations_table(db: SqliteDatabase) -> None:
    sql(
        """
CREATE TABLE IF NOT EXISTS migrations (
	id TEXT PRIMARY KEY,
	applied_at TEXT NOT NULL
);
"""
    ).exec(db)


def _iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


async def apply_migrations(db: SqliteDatabase) -> None:
    _ensure_migrations_table(db)
    migrations = await load_migrations()
    applied_rows = sql("SELECT id FROM migrations ORDER BY applied_at, id").all(db)
    applied = {row["id"] for row in applied_rows}

    for migration in migrations:
        if migration["id"] in applied:
            continue

        def _apply(current: SqliteMigration = migration) -> None:
            db.exec(current["sql"])
            sql("INSERT INTO migrations (id, applied_at) VALUES (?, ?)", current["id"], _iso_now()).run(db)

        db.transaction(_apply)
        applied.add(migration["id"])
