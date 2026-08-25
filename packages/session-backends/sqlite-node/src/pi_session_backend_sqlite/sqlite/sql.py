"""Parameterized SQLite query helper — mirrors sqlite/sql.ts.

TypeScript uses a tagged template. Python 3.11 does not have PEP 750
t-strings, so this module provides:

- ``sql(query, *params)`` — ``?`` placeholders; nested ``SqlQuery`` values
  are inlined. When ``query`` has no ``?`` and extra values are passed, the
  call is treated as a template-like concatenation (strings are trusted SQL,
  ``SqlQuery`` is inlined, other values become ``?`` parameters).
- ``sql_template(*parts)`` — explicit template-like helper.
- ``join_sql_fragments(fragments, separator)`` — join trusted fragments while
  preserving parameter order.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pi_session_backend_sqlite.sqlite.types import SqliteDatabase, SqliteRunResult


class SqlParam:
    """Marks a value as a bound parameter for :func:`sql_template`."""

    def __init__(self, value: Any) -> None:
        self.value = value


def param(value: Any) -> SqlParam:
    """Wrap a value so :func:`sql_template` binds it as ``?`` instead of inlining it."""
    return SqlParam(value)


class SqlQuery:
    """A parameterized SQLite query produced by :func:`sql`."""

    def __init__(self, query_text: str, params: Sequence[Any] | None = None) -> None:
        self.query_text = query_text
        self.params: tuple[Any, ...] = tuple(params or ())

    def exec(self, db: SqliteDatabase) -> None:
        if len(self.params) > 0:
            raise TypeError("SQLite exec queries cannot have parameters")
        db.exec(self.query_text)

    def run(self, db: SqliteDatabase) -> SqliteRunResult:
        return db.prepare(self.query_text).run(*self.params)

    def get(self, db: SqliteDatabase) -> dict[str, Any] | None:
        return db.prepare(self.query_text).get(*self.params)

    def all(self, db: SqliteDatabase) -> list[dict[str, Any]]:
        return db.prepare(self.query_text).all(*self.params)

    def iterate(self, db: SqliteDatabase) -> Any:
        return db.prepare(self.query_text).iterate(*self.params)


def sql_template(*parts: Any) -> SqlQuery:
    """Concatenate trusted strings / nested queries; other values become ``?`` params.

    Plain strings are trusted SQL (like tagged-template literals). Bind string
    values with :func:`param` so they are not inlined.
    """
    query_text: list[str] = []
    params: list[Any] = []
    for part in parts:
        if isinstance(part, SqlQuery):
            query_text.append(part.query_text)
            params.extend(part.params)
        elif isinstance(part, SqlParam):
            query_text.append("?")
            params.append(part.value)
        elif isinstance(part, str):
            query_text.append(part)
        else:
            query_text.append("?")
            params.append(part)
    return SqlQuery("".join(query_text), params)


def _sql_placeholders(query: str, values: Sequence[Any]) -> SqlQuery:
    parts = query.split("?")
    if len(parts) - 1 != len(values):
        raise TypeError(
            f"sql() query has {len(parts) - 1} placeholders but {len(values)} parameters were provided"
        )
    query_text: list[str] = []
    params: list[Any] = []
    for index, value in enumerate(values):
        query_text.append(parts[index])
        if isinstance(value, SqlQuery):
            query_text.append(value.query_text)
            params.extend(value.params)
        else:
            query_text.append("?")
            params.append(value)
    query_text.append(parts[-1])
    return SqlQuery("".join(query_text), params)


def sql(query: str, *params: Any) -> SqlQuery:
    """Build a parameterized query.

    Interpolations become ``?`` parameters. Nested ``SqlQuery`` values are
    inlined (their SQL is substituted and their params flattened).

    Examples::

        sql("SELECT id FROM t WHERE id = ?", session_id)
        sql("SELECT id FROM t WHERE ", join_sql_fragments(predicates, " AND "))
        sql("ASC")
    """
    if not params:
        return SqlQuery(query, ())
    if "?" in query:
        return _sql_placeholders(query, params)
    return sql_template(query, *params)


def join_sql_fragments(fragments: Sequence[SqlQuery], separator: str) -> SqlQuery:
    """Join trusted query fragments while preserving their parameter order."""
    query_text = ""
    params: list[Any] = []
    for index, fragment in enumerate(fragments):
        if index > 0:
            query_text += separator
        query_text += fragment.query_text
        params.extend(fragment.params)
    return SqlQuery(query_text, params)
