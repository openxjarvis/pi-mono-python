"""
List available models with optional fuzzy search.

Mirrors packages/coding-agent/src/cli/list-models.ts
"""
from __future__ import annotations

from typing import Any


def _format_token_count(count: int) -> str:
    """Format a number as human-readable (e.g., 200000 -> '200K', 1000000 -> '1M')."""
    if count >= 1_000_000:
        millions = count / 1_000_000
        return f"{int(millions)}M" if millions == int(millions) else f"{millions:.1f}M"
    if count >= 1_000:
        thousands = count / 1_000
        return f"{int(thousands)}K" if thousands == int(thousands) else f"{thousands:.1f}K"
    return str(count)


def _fuzzy_filter(models: list[Any], pattern: str) -> list[Any]:
    """Simple fuzzy filter — checks if all chars of pattern appear in order in the target."""
    pattern_lower = pattern.lower()

    def matches(model: Any) -> bool:
        target = f"{getattr(model, 'provider', '')} {getattr(model, 'id', '')}".lower()
        it = iter(target)
        return all(c in it for c in pattern_lower)

    return [m for m in models if matches(m)]


async def list_models(model_registry: Any, search_pattern: str | None = None) -> None:
    """
    List available models, optionally filtered by search pattern.

    model_registry should have an async or sync get_available() method.
    """
    import asyncio
    import inspect

    if hasattr(model_registry, "get_available"):
        result = model_registry.get_available()
        if inspect.isawaitable(result):
            models = await result
        else:
            models = result
    else:
        models = []

    if not models:
        print("No models available. Set API keys in environment variables.")
        return

    filtered = models
    if search_pattern:
        filtered = _fuzzy_filter(models, search_pattern)

    if not filtered:
        print(f'No models matching "{search_pattern}"')
        return

    # Sort by provider then model id
    def sort_key(m: Any) -> tuple[str, str]:
        provider = m.get("provider", "") if isinstance(m, dict) else getattr(m, "provider", "")
        mid = m.get("id", "") if isinstance(m, dict) else getattr(m, "id", "")
        return (provider, mid)

    filtered.sort(key=sort_key)

    def get_attr(m: Any, key: str, default: Any = "") -> Any:
        return m.get(key, default) if isinstance(m, dict) else getattr(m, key, default)

    rows = []
    for m in filtered:
        context = get_attr(m, "contextWindow", 0) or get_attr(m, "context_window", 0)
        max_tokens = get_attr(m, "maxTokens", 0) or get_attr(m, "max_tokens", 0)
        reasoning = get_attr(m, "reasoning", False)
        inputs = get_attr(m, "input", []) or []

        rows.append({
            "provider": str(get_attr(m, "provider", "")),
            "model": str(get_attr(m, "id", "")),
            "context": _format_token_count(int(context)),
            "maxOut": _format_token_count(int(max_tokens)),
            "thinking": "yes" if reasoning else "no",
            "images": "yes" if "image" in inputs else "no",
        })

    headers = {
        "provider": "provider",
        "model": "model",
        "context": "context",
        "maxOut": "max-out",
        "thinking": "thinking",
        "images": "images",
    }

    widths = {k: max(len(headers[k]), max((len(r[k]) for r in rows), default=0)) for k in headers}

    header_line = "  ".join(headers[k].ljust(widths[k]) for k in headers)
    print(header_line)

    for row in rows:
        line = "  ".join(row[k].ljust(widths[k]) for k in headers)
        print(line)
