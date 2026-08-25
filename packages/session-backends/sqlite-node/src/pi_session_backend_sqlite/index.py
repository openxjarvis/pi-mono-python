"""Python sqlite3 factory — mirrors src/index.ts (node:sqlite)."""
from __future__ import annotations

import inspect
import sqlite3
from collections.abc import Iterable
from typing import Any

from pi_session_backend_sqlite.sqlite import (
    SqliteDatabase,
    SqliteDatabaseFactory,
    SqliteMigration,
    SqliteRunResult,
    SqliteSessionCreateOptions,
    SqliteSessionListOptions,
    SqliteSessionMetadata,
    SqliteSessionRepository,
    SqliteSessionRepositoryEnv,
    SqliteSessionRepositoryOptions,
    SqliteSessionSearchHit,
    SqliteSessionSearchOptions,
    SqliteStatement,
    SqliteWriterLeaseOptions,
    SqlQuery,
    apply_migrations,
    create_sqlite_session_search,
    join_sql_fragments,
    load_migrations,
    param,
    sql,
    sql_template,
)
from pi_session_backend_sqlite.sqlite.sql import sql as _sql


def _is_named_parameters(value: object) -> bool:
    return isinstance(value, dict)


def _is_async_result(value: object) -> bool:
    return inspect.isawaitable(value)


def _row_to_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    if isinstance(row, sqlite3.Row):
        return {key: row[key] for key in row.keys()}
    return dict(row)


class PythonSqliteStatement:
    def __init__(self, connection: sqlite3.Connection, query: str) -> None:
        self._connection = connection
        self._query = query

    def _execute(self, params: tuple[Any, ...]) -> sqlite3.Cursor:
        if params and _is_named_parameters(params[0]):
            return self._connection.execute(self._query, params[0])
        return self._connection.execute(self._query, params)

    def run(self, *params: Any) -> SqliteRunResult:
        cursor = self._execute(params)
        last_insert_rowid = cursor.lastrowid
        changes = cursor.rowcount if cursor.rowcount is not None and cursor.rowcount >= 0 else 0
        return SqliteRunResult(
            changes=changes,
            last_insert_rowid=None if last_insert_rowid is None else int(last_insert_rowid),
        )

    def get(self, *params: Any) -> dict[str, Any] | None:
        row = self._execute(params).fetchone()
        return None if row is None else _row_to_dict(row)

    def all(self, *params: Any) -> list[dict[str, Any]]:
        return [_row_to_dict(row) for row in self._execute(params).fetchall()]

    def iterate(self, *params: Any) -> Iterable[dict[str, Any]]:
        cursor = self._execute(params)
        for row in cursor:
            yield _row_to_dict(row)


def _execute_script(connection: sqlite3.Connection, script: str) -> None:
    """Run one or more SQL statements without the implicit COMMIT of executescript()."""
    statement = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            stripped = statement.strip()
            if stripped:
                connection.execute(stripped)
            statement = ""
    leftover = statement.strip()
    if leftover:
        connection.execute(leftover)


class PythonSqliteDatabase:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def exec(self, sql_text: str) -> None:
        _execute_script(self._connection, sql_text)

    def prepare(self, sql_text: str) -> SqliteStatement:
        return PythonSqliteStatement(self._connection, sql_text)

    def transaction(self, fn: Any) -> Any:
        _sql("BEGIN IMMEDIATE").exec(self)
        try:
            result = fn()
            if _is_async_result(result):
                raise TypeError("SQLite transaction callbacks must be synchronous")
            _sql("COMMIT").exec(self)
            return result
        except Exception:
            try:
                _sql("ROLLBACK").exec(self)
            except Exception:
                pass
            raise

    def close(self) -> None:
        self._connection.close()


def wrap_python_sqlite_database(conn: sqlite3.Connection) -> SqliteDatabase:
    if conn.row_factory is None:
        conn.row_factory = sqlite3.Row
    conn.isolation_level = None
    return PythonSqliteDatabase(conn)


class PythonSqliteDatabaseFactory:
    async def open(self, path: str) -> SqliteDatabase:
        connection = sqlite3.connect(path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.isolation_level = None
        return PythonSqliteDatabase(connection)


def create_python_sqlite_factory() -> SqliteDatabaseFactory:
    return PythonSqliteDatabaseFactory()


create_node_sqlite_factory = create_python_sqlite_factory

__all__ = [
    "SqlQuery",
    "SqliteDatabase",
    "SqliteDatabaseFactory",
    "SqliteMigration",
    "SqliteRunResult",
    "SqliteSessionCreateOptions",
    "SqliteSessionListOptions",
    "SqliteSessionMetadata",
    "SqliteSessionRepository",
    "SqliteSessionRepositoryEnv",
    "SqliteSessionRepositoryOptions",
    "SqliteSessionSearchHit",
    "SqliteSessionSearchOptions",
    "SqliteStatement",
    "SqliteWriterLeaseOptions",
    "apply_migrations",
    "create_node_sqlite_factory",
    "create_python_sqlite_factory",
    "create_sqlite_session_search",
    "join_sql_fragments",
    "load_migrations",
    "param",
    "sql",
    "sql_template",
    "wrap_python_sqlite_database",
]
