"""
Fuzzy matching utilities — mirrors packages/tui/src/fuzzy.ts

Matches if all query characters appear in order (not necessarily consecutive).
Lower score = better match.
"""
from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class FuzzyMatch:
    __slots__ = ("matches", "score")

    def __init__(self, matches: bool, score: float) -> None:
        self.matches = matches
        self.score = score


def fuzzy_match(query: str, text: str) -> FuzzyMatch:
    """
    Check whether query fuzzy-matches text and compute a quality score.
    Lower score = better match (consecutive matches rewarded, gaps penalised).
    Mirrors fuzzyMatch() in fuzzy.ts.
    """
    query_lower = query.lower()
    text_lower = text.lower()

    def _match_query(normalized_query: str) -> FuzzyMatch:
        if not normalized_query:
            return FuzzyMatch(True, 0)
        if len(normalized_query) > len(text_lower):
            return FuzzyMatch(False, 0)

        query_index = 0
        score: float = 0
        last_match_index = -1
        consecutive = 0

        for i, ch in enumerate(text_lower):
            if query_index >= len(normalized_query):
                break
            if ch == normalized_query[query_index]:
                is_word_boundary = (
                    i == 0 or text_lower[i - 1] in " \t-_./:"
                )
                if last_match_index == i - 1:
                    consecutive += 1
                    score -= consecutive * 5
                else:
                    consecutive = 0
                    if last_match_index >= 0:
                        score += (i - last_match_index - 1) * 2
                if is_word_boundary:
                    score -= 10
                score += i * 0.1
                last_match_index = i
                query_index += 1

        if query_index < len(normalized_query):
            return FuzzyMatch(False, 0)
        return FuzzyMatch(True, score)

    primary = _match_query(query_lower)
    if primary.matches:
        return primary

    # Try swapping alpha-numeric order (e.g. "gpt4" ↔ "4gpt")
    import re
    alpha_num = re.match(r"^(?P<letters>[a-z]+)(?P<digits>[0-9]+)$", query_lower)
    num_alpha = re.match(r"^(?P<digits>[0-9]+)(?P<letters>[a-z]+)$", query_lower)
    if alpha_num:
        swapped = alpha_num.group("digits") + alpha_num.group("letters")
    elif num_alpha:
        swapped = num_alpha.group("letters") + num_alpha.group("digits")
    else:
        return primary

    swapped_match = _match_query(swapped)
    if not swapped_match.matches:
        return primary
    return FuzzyMatch(True, swapped_match.score + 5)


def fuzzy_filter(items: list[T], query: str, get_text: Callable[[T], str]) -> list[T]:
    """
    Filter and sort items by fuzzy match quality (best matches first).
    Supports space-separated tokens: all tokens must match.
    Mirrors fuzzyFilter() in fuzzy.ts.
    """
    if not query.strip():
        return items

    tokens = [t for t in query.strip().split() if t]
    if not tokens:
        return items

    results: list[tuple[T, float]] = []
    for item in items:
        item_text = get_text(item)
        total_score: float = 0
        all_match = True
        for token in tokens:
            m = fuzzy_match(token, item_text)
            if m.matches:
                total_score += m.score
            else:
                all_match = False
                break
        if all_match:
            results.append((item, total_score))

    results.sort(key=lambda x: x[1])
    return [r[0] for r in results]
