"""Alternate-screen transcript search — mirrors alt-screen-search.ts"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .components.input import Input
from .utils import _segment_graphemes, strip_terminal_sequences, truncate_to_width, visible_width


@dataclass
class _SearchSourceSpan:
    row: int
    start_col: int
    end_col: int


@dataclass
class AltScreenSearchSegment:
    row: int
    start_col: int
    end_col: int


@dataclass
class AltScreenSearchMatch:
    segments: list[AltScreenSearchSegment] = field(default_factory=list)


@dataclass
class _SearchCorpus:
    text: str
    source: list[_SearchSourceSpan | None]


def _append_mapped_text(text: str, span: _SearchSourceSpan | None, corpus: _SearchCorpus) -> None:
    corpus.text += text
    corpus.source.extend([span] * len(text))


def _build_search_corpus(lines: list[str] | tuple[str, ...]) -> _SearchCorpus:
    corpus = _SearchCorpus(text="", source=[])
    pending_separator = False

    for row, raw in enumerate(lines):
        line = strip_terminal_sequences(raw or "")
        column = 0
        for text in _segment_graphemes(line):
            width = visible_width(text)
            if re.fullmatch(r"\s+", text):
                if corpus.text:
                    pending_separator = True
                column += width
                continue
            if pending_separator:
                _append_mapped_text(" ", None, corpus)
                pending_separator = False
            _append_mapped_text(text, _SearchSourceSpan(row, column, column + width), corpus)
            column += width
        if corpus.text:
            pending_separator = True

    return corpus


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def _escape_reg_exp(text: str) -> str:
    return re.escape(text)


def find_alt_screen_search_matches(lines: list[str] | tuple[str, ...], query: str) -> list[AltScreenSearchMatch]:
    normalized_query = _normalize_query(query)
    if not normalized_query:
        return []

    corpus = _build_search_corpus(lines)
    expression = re.compile(_escape_reg_exp(normalized_query), re.IGNORECASE | re.UNICODE)
    matches: list[AltScreenSearchMatch] = []

    for match in expression.finditer(corpus.text):
        start = match.start()
        end = match.end()
        segments: list[AltScreenSearchSegment] = []
        for index in range(start, end):
            span = corpus.source[index] if index < len(corpus.source) else None
            if span is None:
                continue
            previous = segments[-1] if segments else None
            if previous and previous.row == span.row and span.start_col <= previous.end_col:
                previous.end_col = max(previous.end_col, span.end_col)
            else:
                segments.append(
                    AltScreenSearchSegment(row=span.row, start_col=span.start_col, end_col=span.end_col)
                )
        if segments:
            matches.append(AltScreenSearchMatch(segments=segments))

    return matches


def get_alt_screen_search_match_key(match: AltScreenSearchMatch) -> str:
    if not match.segments:
        return ""
    first = match.segments[0]
    last = match.segments[-1]
    return f"{first.row}:{first.start_col}:{last.row}:{last.end_col}"


class AltScreenSearchComponent:
    def __init__(self, on_query_change: Callable[[str], None]) -> None:
        self._input = Input()
        self._on_query_change = on_query_change
        self._result_count = 0
        self._result_index = -1
        self._focused = False

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._input.focused = value

    def set_result(self, index: int, count: int) -> None:
        self._result_index = index
        self._result_count = count

    def handle_input(self, data: str) -> None:
        previous = self._input.get_value()
        self._input.handle_input(data)
        query = self._input.get_value()
        if query != previous:
            self._on_query_change(query)

    def invalidate(self) -> None:
        self._input.invalidate()

    def render(self, width: int) -> list[str]:
        safe_width = max(1, width)
        label = " Find transcript"
        query = self._input.get_value()
        if not query:
            status = ""
        elif self._result_count == 0:
            status = "No matches "
        else:
            status = f"{self._result_index + 1}/{self._result_count} "
        label_width = visible_width(label)
        status_width = visible_width(status)
        gap = " " * max(1, safe_width - label_width - status_width)
        title = truncate_to_width(f"{label}{gap}{status}", safe_width, "")
        padding = " " * max(0, safe_width - visible_width(title))
        return [f"\x1b[7m{title}{padding}\x1b[27m", *self._input.render(safe_width)]
