"""Session search/sort — mirrors session-selector-search.ts"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pi_tui.fuzzy import fuzzy_match

SortMode = Literal["threaded", "recent", "relevance"]
NameFilter = Literal["all", "named"]


@dataclass
class ParsedSearchQuery:
    mode: Literal["tokens", "regex"]
    tokens: list[dict[str, str]]
    regex: re.Pattern[str] | None = None
    error: str | None = None


@dataclass
class MatchResult:
    matches: bool
    score: float


def _normalize_whitespace_lower(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _session_search_text(session: Any) -> str:
    return " ".join(
        str(part)
        for part in (
            getattr(session, "id", "") or getattr(session, "session_id", ""),
            getattr(session, "name", "") or "",
            getattr(session, "all_messages_text", "") or "",
            getattr(session, "cwd", "") or "",
        )
    )


def has_session_name(session: Any) -> bool:
    name = getattr(session, "name", None)
    return bool(name and str(name).strip())


def parse_search_query(query: str) -> ParsedSearchQuery:
    trimmed = query.strip()
    if not trimmed:
        return ParsedSearchQuery("tokens", [])
    if trimmed.startswith("re:"):
        pattern = trimmed[3:].strip()
        if not pattern:
            return ParsedSearchQuery("regex", [], error="Empty regex")
        try:
            return ParsedSearchQuery("regex", [], regex=re.compile(pattern, re.I))
        except re.error as exc:
            return ParsedSearchQuery("regex", [], error=str(exc))

    tokens: list[dict[str, str]] = []
    buf = ""
    in_quote = False
    had_unclosed = False

    def flush(kind: str) -> None:
        nonlocal buf
        value = buf.strip()
        buf = ""
        if value:
            tokens.append({"kind": kind, "value": value})

    for ch in trimmed:
        if ch == '"':
            if in_quote:
                flush("phrase")
                in_quote = False
            else:
                flush("fuzzy")
                in_quote = True
            continue
        if not in_quote and ch.isspace():
            flush("fuzzy")
            continue
        buf += ch
    if in_quote:
        had_unclosed = True
    if had_unclosed:
        return ParsedSearchQuery(
            "tokens",
            [{"kind": "fuzzy", "value": part} for part in trimmed.split() if part],
        )
    flush("phrase" if in_quote else "fuzzy")
    return ParsedSearchQuery("tokens", tokens)


def match_session(session: Any, parsed: ParsedSearchQuery) -> MatchResult:
    text = _session_search_text(session)
    if parsed.mode == "regex":
        if parsed.regex is None:
            return MatchResult(False, 0)
        match = parsed.regex.search(text)
        if match is None:
            return MatchResult(False, 0)
        return MatchResult(True, match.start() * 0.1)
    if not parsed.tokens:
        return MatchResult(True, 0)
    total = 0.0
    normalized = None
    for token in parsed.tokens:
        if token["kind"] == "phrase":
            if normalized is None:
                normalized = _normalize_whitespace_lower(text)
            phrase = _normalize_whitespace_lower(token["value"])
            if not phrase:
                continue
            idx = normalized.find(phrase)
            if idx < 0:
                return MatchResult(False, 0)
            total += idx * 0.1
            continue
        result = fuzzy_match(token["value"], text)
        if not result.matches:
            return MatchResult(False, 0)
        total += result.score
    return MatchResult(True, total)


def filter_and_sort_sessions(
    sessions: list[Any],
    query: str,
    sort_mode: SortMode,
    name_filter: NameFilter = "all",
) -> list[Any]:
    named = sessions if name_filter == "all" else [item for item in sessions if has_session_name(item)]
    if not query.strip():
        return named
    parsed = parse_search_query(query)
    if parsed.error:
        return []
    if sort_mode == "recent":
        return [item for item in named if match_session(item, parsed).matches]
    scored = []
    for item in named:
        result = match_session(item, parsed)
        if result.matches:
            scored.append((result.score, item))
    scored.sort(key=lambda pair: (pair[0], -_modified_ts(pair[1])))
    return [item for _, item in scored]


def _modified_ts(session: Any) -> float:
    modified = getattr(session, "modified", None) or getattr(session, "mtime", None)
    if hasattr(modified, "timestamp"):
        return float(modified.timestamp())
    try:
        return float(modified or 0)
    except (TypeError, ValueError):
        return 0.0
