from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .artifacts import PI_SESSION_SNAPSHOT_ARTIFACT, persist_eval_artifact_references
from .harness_table import EVAL_HARNESS_ITERATION_ARTIFACT, parse_eval_harness_iteration_artifact
from .summary import format_harness_comparison_report, summarize_harness_comparisons


async def append_harness_run_report(test: dict[str, Any], run: dict[str, Any]) -> None:
    artifact_directory = os.environ.get("PI_EVAL_ARTIFACT_DIR", "").strip()
    if not artifact_directory:
        return
    run_id = run.get("artifacts", {}).get("runId") or test.get("id") or "run"
    Path(artifact_directory).mkdir(parents=True, exist_ok=True)
    record = {
        "schemaVersion": 1,
        "runId": run_id,
        "test": {"name": test.get("name"), "status": test.get("status")},
        "harness": run.get("name"),
        "usage": run.get("usage"),
        "artifacts": persist_eval_artifact_references(test.get("artifacts") or [], run_id, artifact_directory),
    }
    with (Path(artifact_directory) / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def collect_harness_observations(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations = []
    for run in runs:
        iteration = parse_eval_harness_iteration_artifact((run.get("artifacts") or {}).get(EVAL_HARNESS_ITERATION_ARTIFACT))
        if not iteration:
            continue
        observation = {
            **iteration,
            "testName": run.get("testName"),
            "file": run.get("file"),
            "totalTokens": (run.get("usage") or {}).get("totalTokens"),
            "totalMs": (run.get("timings") or {}).get("totalMs"),
            "outcome": "errored" if run.get("errors") else "scored" if run.get("score") is not None else "unscored",
        }
        if run.get("score") is not None:
            observation["score"] = run["score"]
        observations.append(observation)
    return observations


def render_comparison_report(runs: list[dict[str, Any]]) -> str:
    return format_harness_comparison_report(summarize_harness_comparisons(collect_harness_observations(runs)))
