from __future__ import annotations

from typing import Any, Literal, TypedDict

from pi_agent.harness.session.types import JsonValue, SessionCreateOptions, SessionMetadata


class JsonlSessionRepoOptions(TypedDict):
    fs: Any
    sessions_root: str


class JsonlSessionMetadata(SessionMetadata, total=False):
    cwd: str
    path: str
    modified_at: int
    source_format: Literal[3, 4]
    legacy_parent_session_path: str
    metadata: dict[str, JsonValue]


class JsonlSessionCreateOptions(SessionCreateOptions, total=False):
    cwd: str
    metadata: dict[str, JsonValue]


class JsonlSessionListOptions(TypedDict, total=False):
    cwd: str


class JsonlV4Header(TypedDict, total=False):
    kind: Literal["header"]
    version: Literal[4]
    id: str
    created_at: int
    cwd: str
    parent_session_id: str
    legacy_parent_session_path: str
    metadata: dict[str, JsonValue]
