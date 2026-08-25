"""
JSONL session export — mirrors packages/coding-agent/src/core/session-export.ts
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable

from .session_manager import CURRENT_SESSION_VERSION, SessionManager


def export_session_to_jsonl(
    session_manager: SessionManager,
    output_path: str | None = None,
    create_trailing_entries: Callable[[str | None, str], list[object]] | None = None,
) -> str:
    file_path = os.path.abspath(
        output_path
        or f"session-{datetime.now(timezone.utc).isoformat().replace(':', '-').replace('.', '-')}.jsonl"
    )
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    header = {
        "type": "session",
        "version": CURRENT_SESSION_VERSION,
        "id": session_manager.get_session_id(),
        "timestamp": timestamp,
        "cwd": session_manager.get_cwd(),
    }
    lines = [json.dumps(header, ensure_ascii=False)]

    parent_id: str | None = None
    for entry in session_manager.get_branch():
        payload = dict(entry.data) if isinstance(entry.data, dict) else {
            "id": entry.id,
            "type": entry.type,
            "timestamp": entry.timestamp,
        }
        payload["parentId"] = parent_id
        lines.append(json.dumps(payload, ensure_ascii=False))
        parent_id = entry.id

    if create_trailing_entries:
        for extra in create_trailing_entries(parent_id, timestamp):
            lines.append(json.dumps(extra, ensure_ascii=False))

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return file_path
