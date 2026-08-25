"""SQLite session backend types — mirrors sqlite/types.ts."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from pi_agent.harness.session.types import SessionCreateOptions, SessionMetadata
from pi_agent.harness.types import Result


@dataclass
class SqliteRunResult:
    """Result of a prepared SQLite statement execution."""

    changes: int
    last_insert_rowid: int | None = None


class SqliteStatement(Protocol):
    """Prepared SQLite statement capability used by the SQLite session backend."""

    def run(self, *params: Any) -> SqliteRunResult: ...
    def get(self, *params: Any) -> dict[str, Any] | None: ...
    def all(self, *params: Any) -> list[dict[str, Any]]: ...
    def iterate(self, *params: Any) -> Iterable[dict[str, Any]]: ...


class SqliteDatabase(Protocol):
    """SQLite database capability used by the SQLite session backend."""

    def exec(self, sql: str) -> None: ...
    def prepare(self, sql: str) -> SqliteStatement: ...
    def transaction(self, fn: Any) -> Any: ...
    def close(self) -> None: ...


class SqliteDatabaseFactory(Protocol):
    async def open(self, path: str) -> SqliteDatabase: ...


class SqliteSessionMetadata(SessionMetadata, total=False):
    cwd: str
    path: str
    parent_session_id: str
    name: str
    metadata: dict[str, Any]


class SqliteSessionCreateOptions(SessionCreateOptions, total=False):
    cwd: str
    parent_session_id: str
    metadata: dict[str, Any]


class SqliteSessionListOptions(TypedDict, total=False):
    cwd: str


class SqliteSessionRepositoryEnv(Protocol):
    async def absolute_path(self, path: str, abort: Any = None) -> Result: ...
    async def create_dir(self, path: str, options: dict[str, Any] | None = None) -> Result: ...
    async def exists(self, path: str, abort: Any = None) -> Result: ...
