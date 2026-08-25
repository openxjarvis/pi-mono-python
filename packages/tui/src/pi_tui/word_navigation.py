"""
Word navigation — mirrors packages/tui/src/word-navigation.ts

Unicode word-boundary movement that still preserves ASCII punctuation
boundaries inside word-like segments.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import TypedDict

from .utils import PUNCTUATION_REGEX, is_whitespace_char

_WORD_TOKEN_RE = re.compile(r"\w+|[^\w\s]+|\s+", re.UNICODE)


class _Segment(TypedDict):
    segment: str
    is_word_like: bool


class WordNavigationOptions(TypedDict, total=False):
    segment: Callable[[str], Iterable[_Segment]]
    is_atomic_segment: Callable[[str], bool]


def _default_segments(text: str) -> list[_Segment]:
    return [
        {"segment": m.group(0), "is_word_like": bool(re.match(r"\w", m.group(0), re.UNICODE))}
        for m in _WORD_TOKEN_RE.finditer(text)
    ]


def find_word_backward(text: str, cursor: int, options: WordNavigationOptions | None = None) -> int:
    """Return the cursor position after moving one word backward from ``cursor``."""
    if cursor <= 0:
        return 0

    text_before = text[:cursor]
    segment_fn = options.get("segment") if options else None
    is_atomic = options.get("is_atomic_segment") if options else None
    segments: list[_Segment] = list(segment_fn(text_before)) if segment_fn else _default_segments(text_before)
    new_cursor = cursor

    while (
        segments
        and not (is_atomic and is_atomic(segments[-1]["segment"]))
        and is_whitespace_char(segments[-1]["segment"])
    ):
        new_cursor -= len(segments.pop()["segment"])

    if not segments:
        return new_cursor

    last = segments[-1]
    if is_atomic and is_atomic(last["segment"]):
        new_cursor -= len(last["segment"])
    elif last["is_word_like"]:
        segment = last["segment"]
        matches = list(PUNCTUATION_REGEX.finditer(segment))
        if not matches:
            new_cursor -= len(segment)
        else:
            last_match = matches[-1]
            new_cursor -= len(segment) - (last_match.end())
    else:
        while (
            segments
            and not (is_atomic and is_atomic(segments[-1]["segment"]))
            and not segments[-1]["is_word_like"]
            and not is_whitespace_char(segments[-1]["segment"])
        ):
            new_cursor -= len(segments.pop()["segment"])

    return new_cursor


def find_word_forward(text: str, cursor: int, options: WordNavigationOptions | None = None) -> int:
    """Return the cursor position after moving one word forward from ``cursor``."""
    if cursor >= len(text):
        return len(text)

    text_after = text[cursor:]
    segment_fn = options.get("segment") if options else None
    is_atomic = options.get("is_atomic_segment") if options else None
    segments = iter(segment_fn(text_after) if segment_fn else _default_segments(text_after))
    new_cursor = cursor
    nxt = next(segments, None)

    while nxt is not None and not (is_atomic and is_atomic(nxt["segment"])) and is_whitespace_char(nxt["segment"]):
        new_cursor += len(nxt["segment"])
        nxt = next(segments, None)

    if nxt is None:
        return new_cursor

    if is_atomic and is_atomic(nxt["segment"]):
        new_cursor += len(nxt["segment"])
    elif nxt["is_word_like"]:
        match = PUNCTUATION_REGEX.search(nxt["segment"])
        new_cursor += match.start() if match else len(nxt["segment"])
    else:
        while (
            nxt is not None
            and not (is_atomic and is_atomic(nxt["segment"]))
            and not nxt["is_word_like"]
            and not is_whitespace_char(nxt["segment"])
        ):
            new_cursor += len(nxt["segment"])
            nxt = next(segments, None)

    return new_cursor
