"""
Session replacement host — mirrors packages/coding-agent/src/core/agent-session-runtime.ts
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .agent_session import AgentSession
from .agent_session_services import AgentSessionRuntimeDiagnostic, AgentSessionServices
from .session_cwd import MissingSessionCwdError, assert_session_cwd_exists
from .session_manager import SessionManager


class SessionImportFileNotFoundError(FileNotFoundError):
    def __init__(self, file_path: str) -> None:
        super().__init__(f"File not found: {file_path}")
        self.file_path = file_path
        self.name = "SessionImportFileNotFoundError"


@dataclass
class CreateAgentSessionRuntimeResult:
    session: AgentSession
    services: AgentSessionServices
    diagnostics: list[AgentSessionRuntimeDiagnostic]


class AgentSessionRuntime:
    """Owns the current AgentSession plus its cwd-bound services."""

    def __init__(
        self,
        session: AgentSession,
        services: AgentSessionServices,
        create_runtime: Callable[..., Awaitable[CreateAgentSessionRuntimeResult]] | None = None,
        diagnostics: list[AgentSessionRuntimeDiagnostic] | None = None,
    ) -> None:
        self._session = session
        self._services = services
        self.create_runtime = create_runtime
        self._diagnostics = diagnostics or []

    @property
    def session(self) -> AgentSession:
        return self._session

    @property
    def services(self) -> AgentSessionServices:
        return self._services

    @property
    def cwd(self) -> str:
        return self._session.cwd

    def apply(self, result: CreateAgentSessionRuntimeResult) -> None:
        self._session = result.session
        self._services = result.services
        self._diagnostics = result.diagnostics

    async def new_session(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        created = await self._session.new_session(options)
        return {"cancelled": not created}

    async def switch_session(
        self,
        session_path: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ok = await self._session.switch_session(session_path)
        return {"cancelled": not ok}

    async def fork(
        self,
        entry_id: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        position = (options or {}).get("position", "before")
        if position == "at":
            new_sm = self._session.session_manager.branch(entry_id, self._session.cwd)
            self._session._session_manager = new_sm
            self._session.session_id = new_sm.get_session_id()
            context = new_sm.build_context()
            self._session._agent.replace_messages(context.messages)
            return {"cancelled": False}
        forked = await self._session.fork(entry_id)
        return {"cancelled": False, "selectedText": None, "session": forked}

    async def import_from_jsonl(self, input_path: str, cwd_override: str | None = None) -> dict[str, Any]:
        resolved = os.path.abspath(os.path.expanduser(input_path))
        if not os.path.exists(resolved):
            raise SessionImportFileNotFoundError(resolved)
        session_dir = self._session.session_manager.get_session_dir()
        os.makedirs(session_dir, exist_ok=True)
        destination = os.path.join(session_dir, os.path.basename(resolved))
        if os.path.abspath(destination) != resolved:
            shutil.copy2(resolved, destination)
        session_manager = SessionManager.open(destination)
        assert_session_cwd_exists(session_manager, cwd_override or self._session.cwd)
        return await self._session.import_from_jsonl(input_path, cwd_override)

    async def dispose(self) -> None:
        self._session.dispose()
