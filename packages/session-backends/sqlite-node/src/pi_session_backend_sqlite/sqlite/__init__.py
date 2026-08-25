"""SQLite session backend — mirrors sqlite/index.ts."""
from pi_session_backend_sqlite.sqlite.migrations import SqliteMigration, apply_migrations, load_migrations
from pi_session_backend_sqlite.sqlite.repo import (
    SqliteSessionRepository,
    SqliteSessionRepositoryOptions,
    SqliteWriterLeaseOptions,
)
from pi_session_backend_sqlite.sqlite.search_backend import (
    SqliteSessionSearchHit,
    SqliteSessionSearchOptions,
    create_sqlite_session_search,
)
from pi_session_backend_sqlite.sqlite.sql import SqlQuery, join_sql_fragments, param, sql, sql_template
from pi_session_backend_sqlite.sqlite.types import (
    SqliteDatabase,
    SqliteDatabaseFactory,
    SqliteRunResult,
    SqliteSessionCreateOptions,
    SqliteSessionListOptions,
    SqliteSessionMetadata,
    SqliteSessionRepositoryEnv,
    SqliteStatement,
)

__all__ = [
    "SqliteMigration",
    "apply_migrations",
    "load_migrations",
    "SqliteSessionRepository",
    "SqliteSessionRepositoryOptions",
    "SqliteWriterLeaseOptions",
    "SqliteSessionSearchHit",
    "SqliteSessionSearchOptions",
    "create_sqlite_session_search",
    "SqlQuery",
    "join_sql_fragments",
    "param",
    "sql",
    "sql_template",
    "SqliteDatabase",
    "SqliteDatabaseFactory",
    "SqliteRunResult",
    "SqliteSessionCreateOptions",
    "SqliteSessionListOptions",
    "SqliteSessionMetadata",
    "SqliteSessionRepositoryEnv",
    "SqliteStatement",
]
