from __future__ import annotations

import hashlib
from pathlib import Path

PI_SESSION_SNAPSHOT_ARTIFACT = "piSessionJsonl"


def persist_eval_artifact_references(
    artifacts: list[dict],
    run_id: str,
    artifact_directory: str,
) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    for artifact in artifacts:
        if artifact.get("type") not in {"@earendil-works/pi-evals:session", "@earendil-works/pi-evals:source"}:
            continue
        if artifact.get("runId") != run_id:
            continue
        category = "sessions" if artifact["type"].endswith(":session") else "sources"
        for attachment in artifact.get("attachments", []):
            name = Path(attachment["name"]).name
            if name != attachment["name"]:
                raise TypeError(f"Invalid eval artifact name: {attachment['name']}")
            directory = Path(artifact_directory) / category / hashlib.sha256(run_id.encode()).hexdigest()
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / name
            path.write_text(attachment["body"], encoding="utf-8")
            references.append({"name": name, "path": str(path.relative_to(artifact_directory))})
    return references


async def record_eval_session_artifact(task: object, run: dict) -> None:
    return None


async def record_eval_source_artifact(task: object, run_id: str, attachment: dict) -> None:
    return None
