from __future__ import annotations

from typing import Any

EVAL_HARNESS_ITERATION_ARTIFACT = "vitestEvalsHarnessIteration"


def parse_eval_harness_iteration_artifact(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    required = ("schemaVersion", "evalSet", "groupKey", "harness", "baseline", "candidates", "repetition")
    if value.get("schemaVersion") != 1 or any(key not in value for key in required):
        return None
    if not isinstance(value["candidates"], list) or not all(isinstance(name, str) for name in value["candidates"]):
        return None
    return value


def build_eval_harness_table(options: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = options["baseline"]
    candidates = options.get("candidates") or ([options["candidate"]] if "candidate" in options else [])
    repetitions = int(options.get("repetitions") or 1)
    rows = []
    for repetition in range(repetitions):
        rows.append({"harness": baseline, "name": getattr(baseline, "name", "baseline"), "repetition": repetition})
        for candidate in candidates:
            rows.append({"harness": candidate, "name": getattr(candidate, "name", "candidate"), "repetition": repetition})
    return rows
