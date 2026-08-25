"""Scanning session search — mirrors packages/agent/src/search/scanning.ts."""
from __future__ import annotations

import json
from collections.abc import AsyncIterable, Callable, Sequence
from typing import Any

from pi_agent.harness.session.types import Entry, SessionMetadata
from pi_agent.search.types import SessionSearch, SessionSearchHit, SessionSearchOptions


class AbortError(Exception):
    def __init__(self, message: str = "The operation was aborted") -> None:
        super().__init__(message)
        self.name = "AbortError"


class SessionSearchCandidate(dict):
    """Candidate produced while scanning a readable session."""

    @property
    def entry_id(self) -> str:
        return self["entry_id"]

    @property
    def seq(self) -> int:
        return self["seq"]

    @property
    def type(self) -> str:
        return self["type"]

    @property
    def timestamp(self) -> int:
        return self["timestamp"]

    @property
    def text(self) -> str:
        return self["text"]

    @property
    def fields(self) -> dict[str, Any] | None:
        return self.get("fields")


class ScanningSessionSearchHit(dict):
    @property
    def session_id(self) -> str:
        return self["session_id"]

    @property
    def entry_id(self) -> str:
        return self["entry_id"]

    @property
    def timestamp(self) -> int:
        return self["timestamp"]

    @property
    def snippet(self) -> str:
        return self["snippet"]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        camel = "".join(part[:1].upper() + part[1:] if i else part for i, part in enumerate(key.split("_")))
        if key in obj:
            return obj[key]
        if camel != key and camel in obj:
            return obj[camel]
        return default
    return getattr(obj, key, default)


def _default_search_text(_metadata: SessionMetadata, entry: Entry, label: str | None) -> str:
    serialized = json.dumps(entry, default=lambda value: getattr(value, "__dict__", str(value)), ensure_ascii=False)
    return serialized if label is None else f"{serialized} {label}"


def _throw_if_aborted(abort: Any) -> None:
    if abort is None:
        return
    is_set = getattr(abort, "is_set", None)
    if callable(is_set) and is_set():
        reason = getattr(abort, "reason", None)
        if isinstance(reason, BaseException):
            raise reason
        raise AbortError()
    if getattr(abort, "aborted", False):
        raise AbortError()


async def _scan_readable_entries(
    readable: Any,
    metadata: Any,
    options: dict[str, Any],
    query: dict[str, Any] | None = None,
) -> AsyncIterable[SessionSearchCandidate]:
    query = query or {}
    project_text = options.get("project_text") or _default_search_text
    page_size = query.get("limit") or options.get("page_size") or 100
    after_seq = query.get("after_seq") or 0
    raw_types = query.get("entry_types")
    entry_types = set(raw_types) if raw_types is not None else None
    while True:
        entries = await readable.find_entries(
            {
                "order": "oldestFirst",
                "limit": page_size,
                "cursor": {"after_seq": after_seq, "afterSeq": after_seq},
                "type": raw_types[0] if raw_types is not None and len(raw_types) == 1 else None,
            }
        )
        if not entries:
            break
        for entry in entries:
            entry_type = _get(entry, "type")
            if entry_types is not None and entry_type not in entry_types:
                continue
            label = await readable.get_label(_get(entry, "id"))
            yield SessionSearchCandidate(
                entry_id=_get(entry, "id"),
                seq=_get(entry, "seq"),
                type=entry_type,
                timestamp=_get(entry, "timestamp"),
                text=project_text(metadata, entry, label),
                fields=None if label is None else {"label": label},
            )
        after_seq = _get(entries[-1], "seq", after_seq)
        if len(entries) < page_size:
            break


async def scanning_entries(readable: Any, options: dict[str, Any] | None = None) -> AsyncIterable[SessionSearchCandidate]:
    options = options or {}
    metadata = await readable.get_metadata()
    async for candidate in _scan_readable_entries(readable, metadata, options):
        yield candidate


async def _array_source(readables: Sequence[Any]) -> AsyncIterable[Any]:
    for readable in readables:
        yield readable


def _readables_for(source: Any, options: Any) -> AsyncIterable[Any]:
    if callable(source):
        result = source(options)
        if hasattr(result, "__aiter__"):
            return result

        async def _wrap_iterable() -> AsyncIterable[Any]:
            for item in result:
                yield item

        return _wrap_iterable()
    return _array_source(source)


def _default_match(query_text: str, candidate: SessionSearchCandidate) -> bool:
    return query_text in candidate.text.lower()


def _create_default_hit(metadata: Any, candidate: SessionSearchCandidate) -> ScanningSessionSearchHit:
    return ScanningSessionSearchHit(
        session_id=_get(metadata, "id", ""),
        entry_id=candidate.entry_id,
        timestamp=candidate.timestamp,
        snippet=candidate.text,
    )


def create_scanning_session_search(source: Any, options: dict[str, Any] | None = None) -> SessionSearch:
    options = options or {}
    create_hit = options.get("create_hit") or (
        lambda metadata, candidate: _create_default_hit(metadata, candidate)
    )

    class ScanningSessionSearch:
        async def search(self, text: str, search_options: SessionSearchOptions | None = None):
            search_options = search_options or {}
            normalized_text = text.strip().lower()
            limit = search_options.get("limit")
            if not normalized_text or (limit is not None and limit <= 0):
                return
            entry_types = search_options.get("entry_types")
            if entry_types is not None and len(entry_types) == 0:
                return
            type_set = None if entry_types is None else set(entry_types)
            abort = search_options.get("abort") or search_options.get("signal")
            source_options_fn = options.get("source_options")
            source_options = source_options_fn(normalized_text, search_options) if callable(source_options_fn) else None
            hit_count = 0
            seen_session_ids: set[str] = set()
            async for readable in _readables_for(source, source_options):
                _throw_if_aborted(abort)
                metadata = await readable.get_metadata()
                session_id = _get(metadata, "id")
                if session_id in seen_session_ids:
                    raise ValueError(f"Duplicate sessionId: {session_id}")
                seen_session_ids.add(session_id)
                async for candidate in _scan_readable_entries(
                    readable,
                    metadata,
                    options,
                    {"entry_types": entry_types},
                ):
                    _throw_if_aborted(abort)
                    if type_set is not None and candidate.type not in type_set:
                        continue
                    match = options.get("match")
                    matches = (
                        match(normalized_text, candidate, metadata)
                        if match is not None
                        else _default_match(normalized_text, candidate)
                    )
                    if not matches:
                        continue
                    yield create_hit(metadata, candidate)
                    hit_count += 1
                    if limit is not None and hit_count >= limit:
                        return

    return ScanningSessionSearch()
