"""Harness types. Mirrors packages/agent/src/harness/types.ts"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Protocol

from pi_agent.types import AgentTool, AgentToolResult
from pi_agent.harness.result import (
    Err,
    Ok,
    Result,
    err,
    get_or_throw,
    get_or_undefined,
    ok,
    to_error,
)

FileKind = Literal["file", "directory", "symlink"]
FileErrorCode = Literal[
    "aborted",
    "not_found",
    "permission_denied",
    "not_directory",
    "is_directory",
    "invalid",
    "not_supported",
    "unknown",
]
ExecutionErrorCode = Literal[
    "aborted",
    "timeout",
    "shell_unavailable",
    "spawn_error",
    "callback_error",
    "unknown",
]
CompactionErrorCode = Literal["aborted", "summarization_failed"]
BranchSummaryErrorCode = Literal["aborted", "summarization_failed"]


class FileError(Exception):
    def __init__(self, code: FileErrorCode, message: str, path: str | None = None) -> None:
        super().__init__(message)
        self.name = "FileError"
        self.code = code
        self.path = path


class ExecutionError(Exception):
    def __init__(self, code: ExecutionErrorCode, message: str) -> None:
        super().__init__(message)
        self.name = "ExecutionError"
        self.code = code


class CompactionError(Exception):
    def __init__(self, code: CompactionErrorCode, message: str) -> None:
        super().__init__(message)
        self.name = "CompactionError"
        self.code = code


class BranchSummaryError(Exception):
    def __init__(self, code: BranchSummaryErrorCode, message: str) -> None:
        super().__init__(message)
        self.name = "BranchSummaryError"
        self.code = code


@dataclass
class FileInfo:
    name: str
    path: str
    kind: FileKind
    size: int
    mtime_ms: float

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class Skill:
    name: str
    description: str
    content: str
    file_path: str
    disable_model_invocation: bool = False


@dataclass
class PromptTemplate:
    name: str
    content: str
    description: str | None = None


@dataclass
class AgentHarnessResources:
    prompt_templates: list[PromptTemplate] | None = None
    skills: list[Skill] | None = None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class AgentHarnessStreamOptions:
    transport: Any = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    headers: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None
    cache_retention: Any = None


@dataclass
class ShellExecOptions:
    cwd: str | None = None
    env: dict[str, str] | None = None
    inherit_env: bool = True
    timeout: float | None = None
    abort: asyncio.Event | None = None
    on_stdout: Callable[[str], None] | None = None
    on_stderr: Callable[[str], None] | None = None


class FileSystem(Protocol):
    cwd: str

    async def absolute_path(self, path: str, abort: asyncio.Event | None = None) -> Result[str, FileError]: ...
    async def join_path(self, parts: list[str], abort: asyncio.Event | None = None) -> Result[str, FileError]: ...
    async def read_text_file(self, path: str, abort: asyncio.Event | None = None) -> Result[str, FileError]: ...
    async def read_text_lines(
        self, path: str, options: dict[str, Any] | None = None
    ) -> Result[list[str], FileError]: ...
    async def read_binary_file(self, path: str, abort: asyncio.Event | None = None) -> Result[bytes, FileError]: ...
    async def write_file(
        self, path: str, content: str | bytes, abort: asyncio.Event | None = None
    ) -> Result[None, FileError]: ...
    async def append_file(
        self, path: str, content: str | bytes, abort: asyncio.Event | None = None
    ) -> Result[None, FileError]: ...
    async def rename_file(
        self, source_path: str, destination_path: str, abort: asyncio.Event | None = None
    ) -> Result[None, FileError]: ...
    async def file_info(self, path: str, abort: asyncio.Event | None = None) -> Result[FileInfo, FileError]: ...
    async def list_dir(self, path: str, abort: asyncio.Event | None = None) -> Result[list[FileInfo], FileError]: ...
    async def canonical_path(self, path: str, abort: asyncio.Event | None = None) -> Result[str, FileError]: ...
    async def exists(self, path: str, abort: asyncio.Event | None = None) -> Result[bool, FileError]: ...
    async def create_dir(self, path: str, options: dict[str, Any] | None = None) -> Result[None, FileError]: ...
    async def remove(self, path: str, options: dict[str, Any] | None = None) -> Result[None, FileError]: ...
    async def create_temp_dir(self, prefix: str = "tmp-", abort: asyncio.Event | None = None) -> Result[str, FileError]: ...
    async def create_temp_file(self, options: dict[str, Any] | None = None) -> Result[str, FileError]: ...
    async def cleanup(self) -> None: ...


class Shell(Protocol):
    async def exec(
        self, command: str, options: ShellExecOptions | None = None
    ) -> Result[dict[str, Any], ExecutionError]: ...
    async def cleanup(self) -> None: ...


class ExecutionEnv(FileSystem, Shell, Protocol):
    pass


AgentHarnessTool = AgentTool
AgentHarnessToolContextSource = Any | Callable[[], Any | Awaitable[Any]]
