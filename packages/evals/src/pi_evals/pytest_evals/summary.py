from __future__ import annotations

from typing import Any, Literal, TypedDict


class PairedMetricSummary(TypedDict):
    totalPairs: int
    eligiblePairs: int
    baselineMean: float | None
    candidateMean: float | None
    meanDelta: float | None


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _precise_difference(left: float, right: float) -> float:
    return float(f"{left - right:.15g}")


def summarize_harness_comparisons(observations: list[dict[str, Any]]) -> dict[str, Any]:
    eval_sets: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        eval_sets.setdefault(observation["evalSet"], []).append(observation)
    reports = []
    diagnostics = []
    for eval_set, items in sorted(eval_sets.items()):
        baseline = items[0]["baseline"] if items else ""
        candidates = sorted({name for item in items for name in item.get("candidates", [])})
        comparisons = []
        for candidate in candidates:
            pairs = [
                (left, right)
                for left in items
                if left["harness"] == baseline
                for right in items
                if right["harness"] == candidate
                and left["groupKey"] == right["groupKey"]
                and left["repetition"] == right["repetition"]
            ]
            eligible = [
                pair
                for pair in pairs
                if pair[0].get("outcome") == "scored" and pair[1].get("outcome") == "scored"
            ]
            baseline_pass = sum(1 for left, _ in eligible if left.get("score", 0) >= 1)
            candidate_pass = sum(1 for _, right in eligible if right.get("score", 0) >= 1)
            baseline_rate = None if not eligible else baseline_pass / len(eligible)
            candidate_rate = None if not eligible else candidate_pass / len(eligible)
            comparisons.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "correctness": {
                        "totalPairs": len(pairs),
                        "eligiblePairs": len(eligible),
                        "baselinePassRate": baseline_rate,
                        "candidatePassRate": candidate_rate,
                        "lift": None
                        if baseline_rate is None or candidate_rate is None
                        else _precise_difference(candidate_rate, baseline_rate),
                        "baselineWins": sum(1 for left, right in eligible if left.get("score", 0) >= 1 > right.get("score", 0)),
                        "candidateWins": sum(1 for left, right in eligible if right.get("score", 0) >= 1 > left.get("score", 0)),
                        "ties": sum(1 for left, right in eligible if (left.get("score", 0) >= 1) == (right.get("score", 0) >= 1)),
                    },
                }
            )
        reports.append({"evalSet": eval_set, "comparisons": comparisons})
    return {"schemaVersion": 1, "evalSets": reports, "diagnostics": diagnostics}


def format_harness_comparison_report(report: dict[str, Any]) -> str:
    if all(not eval_set["comparisons"] for eval_set in report.get("evalSets", [])):
        return ""
    lines = ["Eval Comparisons"]
    for eval_set in report.get("evalSets", []):
        lines.append(f"  {eval_set['evalSet']}")
        for comparison in eval_set["comparisons"]:
            correctness = comparison["correctness"]
            lines.append(f"    Baseline   {comparison['baseline']}")
            lines.append(f"    Candidate  {comparison['candidate']}")
            lift = correctness.get("lift")
            lines.append(f"    Pass rate  {lift if lift is not None else 'unavailable'}")
    return "\n".join(lines)
