"""JSONL session repository — mirrors harness/session/jsonl/repo.ts."""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any

from pi_ai.utils.uuid import uuidv7

from pi_agent.harness.session.jsonl.codec import metadata_from_header, parse_header
from pi_agent.harness.session.jsonl.errors import file_result
from pi_agent.harness.session.jsonl.storage import JsonlSessionStorage
from pi_agent.harness.session.jsonl.types import (
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionMetadata,
    JsonlSessionRepoOptions,
    JsonlV4Header,
)
from pi_agent.harness.session.session import Session, assert_json_serializable
from pi_agent.harness.session.types import ForkOptions, SessionError

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def validate_session_id(session_id: str) -> None:
    if not SESSION_ID_PATTERN.match(session_id):
        raise SessionError(
            "invalid_payload",
            "Session id must be non-empty, contain only alphanumeric characters, '-', '_', and '.', and start and end with an alphanumeric character",
        )


def jsonl_session_directory_name(cwd: str) -> str:
    stripped = re.sub(r"^[/\\]", "", cwd)
    encoded = re.sub(r"[/\\:]", "-", stripped)
    return f"--{encoded}--"


async def jsonl_sessions_root(options: JsonlSessionRepoOptions) -> str:
    return file_result(
        await options["fs"].absolute_path(options["sessions_root"]),
        f"Failed to resolve sessions root {options['sessions_root']}",
    )


async def jsonl_session_directory(fs: Any, sessions_root: str, cwd: str) -> str:
    return file_result(
        await fs.join_path([sessions_root, jsonl_session_directory_name(cwd)]),
        f"Failed to resolve sessions directory for {cwd}",
    )


async def jsonl_session_directories(options: JsonlSessionRepoOptions, cwd: str | None = None) -> list[str]:
    sessions_root = await jsonl_sessions_root(options)
    if cwd is not None:
        resolved_cwd = file_result(await options["fs"].absolute_path(cwd), f"Failed to resolve session cwd {cwd}")
        directory = await jsonl_session_directory(options["fs"], sessions_root, resolved_cwd)
        exists = file_result(await options["fs"].exists(directory), f"Failed to check sessions directory {directory}")
        return [directory] if exists else []
    if not file_result(await options["fs"].exists(sessions_root), f"Failed to check sessions directory {sessions_root}"):
        return []
    entries = file_result(await options["fs"].list_dir(sessions_root), f"Failed to list sessions directory {sessions_root}")
    return [entry["path"] for entry in entries if entry["kind"] in ("directory", "symlink")]


async def list_jsonl_session_metadata(
    options: JsonlSessionRepoOptions,
    query: JsonlSessionListOptions | None = None,
) -> list[JsonlSessionMetadata]:
    query = query or {}
    metadata: list[JsonlSessionMetadata] = []
    for directory in await jsonl_session_directories(options, query.get("cwd")):
        files = [
            entry
            for entry in file_result(await options["fs"].list_dir(directory), f"Failed to list sessions directory {directory}")
            if entry["kind"] != "directory" and entry["name"].endswith(".jsonl")
        ]
        for file in files:
            lines = file_result(
                await options["fs"].read_text_lines(file["path"], {"max_lines": 1}),
                f"Failed to read session header {file['path']}",
            )
            if not lines:
                continue
            header_result = parse_header(lines[0])
            if not header_result["ok"]:
                continue
            metadata.append(metadata_from_header(header_result["value"], file["path"], file["mtime_ms"]))
    metadata.sort(key=lambda item: item["modified_at"], reverse=True)
    return metadata


async def load_jsonl_session_storage(
    options: JsonlSessionRepoOptions,
    metadata: JsonlSessionMetadata,
) -> JsonlSessionStorage:
    if not file_result(await options["fs"].exists(metadata["path"]), f"Failed to check session {metadata['path']}"):
        raise SessionError("not_found", f"Session not found: {metadata['id']}")
    storage = await JsonlSessionStorage.load(options["fs"], metadata["path"])
    loaded = await storage.get_metadata()
    if loaded["id"] != metadata["id"]:
        raise SessionError("invalid_entry", f"Session id does not match header: {metadata['id']}")
    return storage


def session_file_name(created_at: int, session_id: str) -> str:
    timestamp = datetime.fromtimestamp(created_at / 1000, tz=timezone.utc).isoformat().replace(":", "-").replace(".", "-")
    return f"{timestamp}_{session_id}.jsonl"


class JsonlSessionRepo:
    def __init__(self, options: JsonlSessionRepoOptions) -> None:
        self._fs = options["fs"]
        self._sessions_root_input = options["sessions_root"]
        self._active_create_destinations: set[str] = set()
        self._root: str | None = None

    async def create(self, options: JsonlSessionCreateOptions) -> Session:
        destination = await self._resolve_create_destination(options)
        return await self._claim_create_destination(
            destination,
            lambda: self._create_session(destination, options),
        )

    async def _create_session(self, destination: dict[str, str], options: JsonlSessionCreateOptions) -> Session:
        prepared = await self._prepare_create(destination, options)
        return Session(await JsonlSessionStorage.create(self._fs, prepared["path"], prepared["header"]))

    async def open(self, metadata: JsonlSessionMetadata) -> Session:
        return Session(await self._load_storage(metadata))

    async def list(self, options: JsonlSessionListOptions | None = None) -> list[JsonlSessionMetadata]:
        return await list_jsonl_session_metadata(
            {"fs": self._fs, "sessions_root": self._sessions_root_input},
            options or {},
        )

    async def delete(self, metadata: JsonlSessionMetadata) -> None:
        file_result(await self._fs.remove(metadata["path"], {"force": True}), f"Failed to delete session {metadata['path']}")

    async def fork(self, source: JsonlSessionMetadata, options: ForkOptions) -> Session:
        source_storage = await self._load_storage(source)
        create_options = {**options, "parent_session_id": options.get("parent_session_id") or source["id"]}
        destination = await self._resolve_create_destination(create_options)

        async def _fork() -> Session:
            prepared = await self._prepare_create(destination, create_options)
            return Session(await source_storage.fork(prepared["path"], prepared["header"], options))

        return await self._claim_create_destination(destination, _fork)

    async def _load_storage(self, metadata: JsonlSessionMetadata) -> JsonlSessionStorage:
        return await load_jsonl_session_storage(
            {"fs": self._fs, "sessions_root": self._sessions_root_input},
            metadata,
        )

    async def _resolve_create_destination(self, options: JsonlSessionCreateOptions) -> dict[str, str]:
        session_id = options.get("id") or uuidv7()
        validate_session_id(session_id)
        cwd = file_result(await self._fs.absolute_path(options["cwd"]), f"Failed to resolve session cwd {options['cwd']}")
        return {"id": session_id, "cwd": cwd}

    async def _claim_create_destination(self, destination: dict[str, str], operation: Any) -> Session:
        key = f"{destination['cwd']}\0{destination['id']}"
        if key in self._active_create_destinations:
            raise SessionError("already_exists", f"Session already exists: {destination['id']}")
        self._active_create_destinations.add(key)
        try:
            return await operation()
        finally:
            self._active_create_destinations.discard(key)

    async def _prepare_create(
        self,
        destination: dict[str, str],
        options: JsonlSessionCreateOptions,
    ) -> dict[str, Any]:
        session_id = destination["id"]
        cwd = destination["cwd"]
        if await self._session_id_exists(session_id, cwd):
            raise SessionError("already_exists", f"Session already exists: {session_id}")
        created_at = int(time.time() * 1000)
        session_directory = await self._session_directory(cwd)
        path = file_result(
            await self._fs.join_path([session_directory, session_file_name(created_at, session_id)]),
            f"Failed to resolve path for session {session_id}",
        )
        if options.get("metadata") is not None:
            assert_json_serializable(options["metadata"])
        header: JsonlV4Header = {
            "kind": "header",
            "version": 4,
            "id": session_id,
            "created_at": created_at,
            "cwd": cwd,
        }
        if options.get("parent_session_id") is not None:
            header["parent_session_id"] = options["parent_session_id"]
        if options.get("metadata") is not None:
            header["metadata"] = options["metadata"]
        file_result(await self._fs.create_dir(session_directory, {"recursive": True}), "Failed to create sessions directory")
        return {"header": header, "path": path}

    async def _session_id_exists(self, session_id: str, cwd: str) -> bool:
        suffix = f"_{session_id}.jsonl"
        directory = await self._session_directory(cwd)
        if not file_result(await self._fs.exists(directory), f"Failed to check sessions directory {directory}"):
            return False
        files = file_result(await self._fs.list_dir(directory), f"Failed to list sessions directory {directory}")
        return any(entry["kind"] != "directory" and entry["name"].endswith(suffix) for entry in files)

    async def _session_directory(self, cwd: str) -> str:
        return file_result(
            await self._fs.join_path([await self._root_path(), jsonl_session_directory_name(cwd)]),
            f"Failed to resolve sessions directory for {cwd}",
        )

    async def _root_path(self) -> str:
        if self._root is None:
            self._root = file_result(
                await self._fs.absolute_path(self._sessions_root_input),
                f"Failed to resolve sessions root {self._sessions_root_input}",
            )
        return self._root
