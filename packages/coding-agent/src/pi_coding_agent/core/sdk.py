"""
SDK factory — mirrors packages/coding-agent/src/core/sdk.ts

Public API for creating AgentSession instances programmatically.
"""
from __future__ import annotations

import os
from typing import Any

from pi_ai.types import Model

from .agent_session import AgentSession
from .auth_storage import AuthStorage
from .model_registry import ModelRegistry
from .session_manager import SessionManager
from .settings_manager import Settings


class AgentSessionOptions:
    """Options for creating an AgentSession."""

    def __init__(
        self,
        cwd: str | None = None,
        model: Model | None = None,
        model_id: str | None = None,
        provider: str | None = None,
        session_id: str | None = None,
        api_key: str | None = None,
        system_prompt: str | None = None,
        thinking_level: str = "off",
        auto_compact: bool = True,
        sessions_dir: str | None = None,
        session_manager: SessionManager | None = None,
        auth_storage: AuthStorage | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self.cwd = cwd
        self.model = model
        self.model_id = model_id
        self.provider = provider
        self.session_id = session_id
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.thinking_level = thinking_level
        self.auto_compact = auto_compact
        self.sessions_dir = sessions_dir
        self.session_manager = session_manager
        self.auth_storage = auth_storage
        self.model_registry = model_registry


def create_agent_session(opts: AgentSessionOptions | None = None) -> AgentSession:
    """
    Create an AgentSession with the given options.
    Mirrors createAgentSession() in TypeScript.
    """
    if opts is None:
        opts = AgentSessionOptions()

    settings = Settings(
        thinking_level=opts.thinking_level,
        auto_compact=opts.auto_compact,
        model_id=opts.model_id,
        provider=opts.provider,
    )

    auth = opts.auth_storage or AuthStorage()
    if opts.api_key and opts.provider:
        auth.set_api_key(opts.provider, opts.api_key)

    session_manager = opts.session_manager or SessionManager(sessions_dir=opts.sessions_dir)
    model_registry = opts.model_registry or ModelRegistry()

    session = AgentSession(
        cwd=opts.cwd,
        model=opts.model,
        settings=settings,
        session_id=opts.session_id,
        session_manager=session_manager,
        auth_storage=auth,
        model_registry=model_registry,
    )

    return session
