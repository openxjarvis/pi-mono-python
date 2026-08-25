"""
Usage aggregation — mirrors packages/coding-agent/src/core/usage-totals.ts
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UsageTotals:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float = 0.0


def create_usage_totals() -> UsageTotals:
    return UsageTotals()


def _usage_value(usage: Any, *names: str) -> int:
    for name in names:
        if isinstance(usage, dict) and name in usage:
            return int(usage.get(name) or 0)
        value = getattr(usage, name, None)
        if value is not None:
            return int(value or 0)
    return 0


def _usage_cost(usage: Any) -> float:
    cost = usage.get("cost") if isinstance(usage, dict) else getattr(usage, "cost", None)
    if cost is None:
        return 0.0
    if isinstance(cost, (int, float)):
        return float(cost)
    total = cost.get("total") if isinstance(cost, dict) else getattr(cost, "total", 0)
    return float(total or 0)


def add_usage_to_totals(totals: UsageTotals, usage: Any) -> None:
    if usage is None:
        return
    totals.input += _usage_value(usage, "input")
    totals.output += _usage_value(usage, "output")
    totals.cache_read += _usage_value(usage, "cache_read", "cacheRead")
    totals.cache_write += _usage_value(usage, "cache_write", "cacheWrite")
    totals.cost += _usage_cost(usage)


@dataclass
class UsageCostBreakdownEntry:
    key: str
    cost: float
    tokens: int


def get_usage_cost_breakdown(entries: list[Any]) -> list[UsageCostBreakdownEntry]:
    totals_by_key: dict[str, UsageTotals] = {}

    for entry in entries:
        key: str | None = None
        usage: Any = None
        entry_type = getattr(entry, "type", None) or (entry.get("type") if isinstance(entry, dict) else None)
        data = getattr(entry, "data", None) if not isinstance(entry, dict) else entry
        message = None
        if data is not None:
            message = data.get("message") if isinstance(data, dict) else getattr(data, "message", None)
        if isinstance(entry, dict):
            message = entry.get("message") or message

        role = None
        if isinstance(message, dict):
            role = message.get("role")
            usage = message.get("usage")
        elif message is not None:
            role = getattr(message, "role", None)
            usage = getattr(message, "usage", None)

        if entry_type == "message" and role == "assistant":
            provider = (
                message.get("provider") if isinstance(message, dict) else getattr(message, "provider", "")
            ) or ""
            model = (
                (message.get("responseModel") or message.get("model"))
                if isinstance(message, dict)
                else (getattr(message, "response_model", None) or getattr(message, "model", ""))
            ) or ""
            key = f"{provider}/{model}"
        elif entry_type == "message" and role == "toolResult" and usage:
            key = "Tools/summaries"
        elif entry_type in ("branch_summary", "compaction"):
            usage = (data.get("usage") if isinstance(data, dict) else getattr(data, "usage", None)) or (
                entry.get("usage") if isinstance(entry, dict) else None
            )
            if usage:
                key = "Tools/summaries"

        if not key or not usage:
            continue
        totals = totals_by_key.setdefault(key, create_usage_totals())
        add_usage_to_totals(totals, usage)

    breakdown = [
        UsageCostBreakdownEntry(
            key=key,
            cost=totals.cost,
            tokens=totals.input + totals.output + totals.cache_read + totals.cache_write,
        )
        for key, totals in totals_by_key.items()
        if totals.cost > 0 or (totals.input + totals.output + totals.cache_read + totals.cache_write) > 0
    ]
    breakdown.sort(key=lambda item: item.cost, reverse=True)
    return breakdown
