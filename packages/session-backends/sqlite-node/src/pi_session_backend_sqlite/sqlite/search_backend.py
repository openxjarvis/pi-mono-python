"""SQLite FTS session search — mirrors sqlite/search-backend.ts."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, TypedDict

from pi_agent.harness.session.types import SessionError
from pi_agent.harness.types import Result
from pi_agent.search.types import SessionSearch, SessionSearchHit, SessionSearchOptions

from pi_session_backend_sqlite.sqlite.migrations import apply_migrations
from pi_session_backend_sqlite.sqlite.sql import sql
from pi_session_backend_sqlite.sqlite.storage.sessions import decode_session_metadata
from pi_session_backend_sqlite.sqlite.types import (
    SqliteDatabase,
    SqliteDatabaseFactory,
    SqliteSessionMetadata,
    SqliteSessionRepositoryEnv,
)


def _file_system_result_or_throw(result: Result, message: str) -> Any:
    if not result.get("ok"):
        error = result["error"]
        code = "not_found" if getattr(error, "code", None) == "not_found" else "storage"
        raise SessionError(code, f"{message}: {error}", error if isinstance(error, Exception) else None)
    return result["value"]


def _get_parent_path(path: str) -> str:
    normalized = path.rstrip("\\/")
    last_slash = max(normalized.rfind("/"), normalized.rfind("\\"))
    if last_slash < 0:
        return "."
    if last_slash == 0:
        return normalized[:1]
    return normalized[:last_slash]


def _throw_if_aborted(signal: Any | None) -> None:
    if signal is None:
        return
    aborted = getattr(signal, "aborted", None)
    if aborted:
        reason = getattr(signal, "reason", None)
        if isinstance(reason, Exception):
            raise reason
        error = Exception("The operation was aborted")
        error.name = "AbortError"  # type: ignore[attr-defined]
        raise error
    if callable(getattr(signal, "is_set", None)) and signal.is_set():
        error = Exception("The operation was aborted")
        error.name = "AbortError"  # type: ignore[attr-defined]
        raise error


def _configure_sqlite_database(db: SqliteDatabase) -> None:
    sql("PRAGMA journal_mode=WAL").exec(db)
    sql("PRAGMA synchronous=FULL").exec(db)
    sql("PRAGMA busy_timeout=5000").exec(db)


class SqliteSessionSearchOptions(TypedDict):
    env: SqliteSessionRepositoryEnv
    sqlite: SqliteDatabaseFactory
    database_path: str


def _table_exists(db: SqliteDatabase, name: str) -> bool:
    return bool(
        sql(
            "SELECT 1 AS found FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            name,
        ).get(db)
    )


def _rebuild_search_index(db: SqliteDatabase) -> None:
    sql("INSERT INTO session_search_fts(session_search_fts) VALUES('rebuild')").run(db)


def _ensure_search_schema(db: SqliteDatabase) -> None:
    fts_exists = _table_exists(db, "session_search_fts")
    entries_exist = _table_exists(db, "entries")

    def _create() -> None:
        sql(
            """
CREATE VIRTUAL TABLE IF NOT EXISTS session_search_fts USING fts5(
  payload,
  content = 'entries',
  content_rowid = 'rowid',
  tokenize = 'trigram remove_diacritics 1'
);
CREATE TRIGGER IF NOT EXISTS session_search_fts_ai AFTER INSERT ON entries BEGIN
  INSERT INTO session_search_fts(rowid, payload) VALUES (new.rowid, new.payload);
END;
CREATE TRIGGER IF NOT EXISTS session_search_fts_ad AFTER DELETE ON entries BEGIN
  INSERT INTO session_search_fts(session_search_fts, rowid, payload) VALUES('delete', old.rowid, old.payload);
END;
CREATE TRIGGER IF NOT EXISTS session_search_fts_au AFTER UPDATE OF payload ON entries BEGIN
  INSERT INTO session_search_fts(session_search_fts, rowid, payload) VALUES('delete', old.rowid, old.payload);
  INSERT INTO session_search_fts(rowid, payload) VALUES (new.rowid, new.payload);
END;
"""
        ).exec(db)
        if not fts_exists and entries_exist:
            _rebuild_search_index(db)

    db.transaction(_create)


class SqliteSessionSearchHit(SessionSearchHit, total=False):
    metadata: SqliteSessionMetadata
    timestamp: int
    score: float


class SqliteSessionSearch:
    def __init__(self, options: SqliteSessionSearchOptions | dict[str, Any]) -> None:
        self._options = options
        self._database_path: str | None = None

    def _database_path_option(self) -> str:
        return self._options.get("database_path") or self._options["databasePath"]  # type: ignore[index]

    async def _get_database_path(self) -> str:
        if not self._database_path:
            self._database_path = _file_system_result_or_throw(
                await self._options["env"].absolute_path(self._database_path_option()),
                f"Failed to resolve SQLite search database {self._database_path_option()}",
            )
        return self._database_path

    async def _open_database(self) -> SqliteDatabase:
        path = await self._get_database_path()
        directory = _get_parent_path(path)
        _file_system_result_or_throw(
            await self._options["env"].create_dir(directory, {"recursive": True}),
            f"Failed to create SQLite search directory {directory}",
        )
        db = await self._options["sqlite"].open(path)
        try:
            _configure_sqlite_database(db)
            await apply_migrations(db)
            _ensure_search_schema(db)
            return db
        except Exception:
            db.close()
            raise

    async def search(
        self,
        text: str,
        options: SessionSearchOptions | dict[str, Any] | None = None,
    ) -> AsyncIterator[SqliteSessionSearchHit]:
        options = options or {}
        query_text = text.strip()
        limit = options.get("limit")
        if not query_text or (limit is not None and limit <= 0):
            return
        entry_types = options.get("entry_types", options.get("entryTypes"))
        if entry_types is not None and len(entry_types) == 0:
            return
        signal = options.get("signal", options.get("abort"))
        _throw_if_aborted(signal)
        db = await self._open_database()
        try:
            query = '"' + query_text.replace('"', '""') + '"'
            predicates = ["session_search_fts MATCH ?"]
            params: list[Any] = [query]
            if entry_types is not None:
                predicates.append(f"se.type IN ({', '.join('?' for _ in entry_types)})")
                params.extend(entry_types)
            rows = db.prepare(
                f"""SELECT s.id, s.created_at, s.metadata, s.cwd, s.parent_session_id,
						name_fact.seq IS NOT NULL AS has_session_name,
						name_fact.value AS session_name,
						se.id AS entry_id, se.timestamp, bm25(session_search_fts) AS score
					FROM session_search_fts
					JOIN entries AS se ON se.rowid = session_search_fts.rowid
					JOIN sessions AS s ON s.id = se.session_id
					LEFT JOIN facts AS name_fact
						ON name_fact.session_id = s.id
						AND name_fact.kind = 'name'
						AND name_fact.key IS NULL
						AND name_fact.seq = (
							SELECT MAX(f.seq)
							FROM facts AS f
							WHERE f.session_id = s.id AND f.kind = 'name' AND f.key IS NULL
						)
					WHERE {' AND '.join(predicates)}
					ORDER BY score
					LIMIT ?"""
            ).iterate(*params, limit if limit is not None else -1)
            path = await self._get_database_path()
            for row in rows:
                _throw_if_aborted(signal)
                yield {
                    "session_id": row["id"],
                    "metadata": decode_session_metadata(row, path),
                    "entry_id": row["entry_id"],
                    "timestamp": row["timestamp"],
                    "score": row["score"],
                }
        finally:
            db.close()


def create_sqlite_session_search(
    options: SqliteSessionSearchOptions | dict[str, Any],
) -> SessionSearch:
    return SqliteSessionSearch(options)
