from __future__ import annotations

from .artifacts import record_eval_session_artifact


async def after_each(task: object, run: dict | None = None) -> None:
    if run:
        await record_eval_session_artifact(task, run)
