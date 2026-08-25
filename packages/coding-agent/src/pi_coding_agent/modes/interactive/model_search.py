"""Model search. Mirrors packages/coding-agent/src/modes/interactive/model-search.ts"""
from __future__ import annotations

from typing import Any


def search_models(models: list[Any], query: str) -> list[Any]:
    if not query:
        return list(models)
    q = query.lower()
    hits = []
    for model in models:
        text = " ".join(
            str(getattr(model, key, "") if not isinstance(model, dict) else model.get(key, ""))
            for key in ("id", "name", "provider")
        ).lower()
        if q in text:
            hits.append(model)
    return hits
