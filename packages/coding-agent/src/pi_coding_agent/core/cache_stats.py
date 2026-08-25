"""
Prompt-cache waste analysis — mirrors packages/coding-agent/src/core/cache-stats.ts
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

CACHE_TTL_MS = 5 * 60 * 1000
NOISE_FLOOR_TOKENS = 1024


@dataclass
class CacheMiss:
    missed_tokens: int
    missed_cost: float
    idle_ms: int
    model_changed: bool


@dataclass
class CacheWasteTotals:
    missed_tokens: int = 0
    missed_cost: float = 0.0
    miss_count: int = 0


class ModelPriceSource(Protocol):
    def get_model(self, provider: str, model_id: str) -> Any: ...


@dataclass
class _PreviousRequest:
    prompt_tokens: int
    model_key: str
    timestamp: int
    reported_cache: bool


def _usage_field(usage: Any, *names: str) -> int:
    if usage is None:
        return 0
    for name in names:
        if isinstance(usage, dict) and name in usage:
            return int(usage.get(name) or 0)
        value = getattr(usage, name, None)
        if value is not None:
            return int(value or 0)
    return 0


def _usage_cost_field(usage: Any, field: str) -> float:
    cost = usage.get("cost") if isinstance(usage, dict) else getattr(usage, "cost", None)
    if cost is None:
        return 0.0
    if isinstance(cost, dict):
        return float(cost.get(field) or 0)
    return float(getattr(cost, field, 0) or 0)


def detect_miss(
    prev: _PreviousRequest | None,
    message: Any,
    models: ModelPriceSource | None = None,
) -> CacheMiss | None:
    usage = getattr(message, "usage", None) if not isinstance(message, dict) else message.get("usage")
    if usage is None:
        return None
    input_tokens = _usage_field(usage, "input")
    cache_read = _usage_field(usage, "cache_read", "cacheRead")
    cache_write = _usage_field(usage, "cache_write", "cacheWrite")
    prompt_tokens = input_tokens + cache_read + cache_write
    if not prev or prompt_tokens <= 0 or (cache_read + cache_write == 0 and not prev.reported_cache):
        return None

    missed_tokens = min(prev.prompt_tokens, prompt_tokens) - cache_read
    if missed_tokens <= NOISE_FLOOR_TOKENS:
        return None

    paid_tokens = input_tokens + cache_write
    paid_per_token = (
        (_usage_cost_field(usage, "input") + _usage_cost_field(usage, "cacheWrite") + _usage_cost_field(usage, "cache_write"))
        / paid_tokens
        if paid_tokens > 0
        else 0.0
    )
    if cache_read > 0:
        read_per_token = _usage_cost_field(usage, "cacheRead") / cache_read if _usage_cost_field(usage, "cacheRead") else (
            _usage_cost_field(usage, "cache_read") / cache_read
        )
    else:
        read_per_token = 0.0
        if models is not None:
            provider = message.get("provider") if isinstance(message, dict) else getattr(message, "provider", "")
            model_id = message.get("model") if isinstance(message, dict) else getattr(message, "model", "")
            model = models.get_model(provider, model_id)
            cost = getattr(model, "cost", None) if model is not None else None
            cache_read_price = (
                cost.get("cacheRead") if isinstance(cost, dict) else getattr(cost, "cache_read", 0) if cost else 0
            )
            read_per_token = float(cache_read_price or 0) / 1_000_000

    timestamp = message.get("timestamp") if isinstance(message, dict) else getattr(message, "timestamp", 0)
    provider = message.get("provider") if isinstance(message, dict) else getattr(message, "provider", "")
    model_id = message.get("model") if isinstance(message, dict) else getattr(message, "model", "")
    return CacheMiss(
        missed_tokens=missed_tokens,
        missed_cost=missed_tokens * max(0.0, paid_per_token - read_per_token),
        idle_ms=max(0, int(timestamp or 0) - prev.timestamp),
        model_changed=f"{provider}/{model_id}" != prev.model_key,
    )


def _as_previous_request(message: Any, reported_cache: bool) -> _PreviousRequest | None:
    usage = getattr(message, "usage", None) if not isinstance(message, dict) else message.get("usage")
    if usage is None:
        return None
    input_tokens = _usage_field(usage, "input")
    cache_read = _usage_field(usage, "cache_read", "cacheRead")
    cache_write = _usage_field(usage, "cache_write", "cacheWrite")
    prompt_tokens = input_tokens + cache_read + cache_write
    if prompt_tokens <= 0:
        return None
    provider = message.get("provider") if isinstance(message, dict) else getattr(message, "provider", "")
    model_id = message.get("model") if isinstance(message, dict) else getattr(message, "model", "")
    timestamp = message.get("timestamp") if isinstance(message, dict) else getattr(message, "timestamp", 0)
    return _PreviousRequest(
        prompt_tokens=prompt_tokens,
        model_key=f"{provider}/{model_id}",
        timestamp=int(timestamp or 0),
        reported_cache=reported_cache or cache_read + cache_write > 0,
    )


def _entry_message(entry: Any) -> Any:
    if isinstance(entry, dict):
        return entry.get("message") or (entry.get("data") or {}).get("message")
    data = getattr(entry, "data", None)
    if isinstance(data, dict):
        return data.get("message")
    return getattr(entry, "message", None)


def _entry_type(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("type") or "")
    return str(getattr(entry, "type", "") or "")


def _scan(
    entries: list[Any],
    models: ModelPriceSource | None,
) -> tuple[_PreviousRequest | None, CacheWasteTotals, dict[int, CacheMiss]]:
    prev: _PreviousRequest | None = None
    totals = CacheWasteTotals()
    misses: dict[int, CacheMiss] = {}
    for index, entry in enumerate(entries):
        etype = _entry_type(entry)
        if etype in ("compaction", "branch_summary"):
            prev = None
            continue
        message = _entry_message(entry)
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if etype == "message" and role == "assistant":
            miss = detect_miss(prev, message, models)
            if miss:
                totals.missed_tokens += miss.missed_tokens
                totals.missed_cost += miss.missed_cost
                totals.miss_count += 1
                misses[index] = miss
            prev = _as_previous_request(message, prev.reported_cache if prev else False) or prev
    return prev, totals, misses


def compute_cache_waste(entries: list[Any], models: ModelPriceSource | None = None) -> CacheWasteTotals:
    return _scan(entries, models)[1]


def collect_cache_misses(entries: list[Any], models: ModelPriceSource | None = None) -> dict[int, CacheMiss]:
    return _scan(entries, models)[2]


def detect_cache_miss(
    entries: list[Any],
    message: Any,
    models: ModelPriceSource | None = None,
) -> CacheMiss | None:
    prev, _, _ = _scan(entries, models)
    return detect_miss(prev, message, models)
