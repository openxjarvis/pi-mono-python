"""
SDK factory — mirrors packages/coding-agent/src/core/sdk.ts

Public API for creating AgentSession instances programmatically.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from pi_agent.types import AgentTool, ThinkingLevel
from pi_ai import clamp_thinking_level
from pi_ai.types import Model

from pi_coding_agent.core.defaults import DEFAULT_THINKING_LEVEL

from .agent_session import AgentSession
from .auth_storage import AuthStorage
from .extensions import ToolDefinition
from .model_registry import ModelRegistry
from .model_resolver import find_initial_model, restore_model_from_session
from .session_manager import SessionManager
from .settings_manager import Settings, SettingsManager


class ResourceLoader(Protocol):
    """Protocol for resource loaders."""
    def get_extensions(self) -> dict[str, Any] | None: ...
    async def reload(self) -> None: ...


@dataclass
class CreateAgentSessionOptions:
    """
    Options for creating an AgentSession.
    Mirrors CreateAgentSessionOptions in TypeScript.
    """
    # Working directory for project-local discovery. Default: os.getcwd()
    cwd: str | None = None
    # Global config directory. Default: ~/.pi/agent
    agent_dir: str | None = None
    
    # Auth storage for credentials
    auth_storage: AuthStorage | None = None
    # Model registry
    model_registry: ModelRegistry | None = None
    
    # Model to use. Default: from settings, else first available
    model: Model | None = None
    # Thinking level. Default: from settings, else 'medium' (clamped to model capabilities)
    thinking_level: ThinkingLevel | None = None
    # Models available for cycling (Ctrl+P in interactive mode)
    scoped_models: list[dict[str, Model | ThinkingLevel | None]] | None = None
    
    # Built-in tools to use. Default: codingTools [read, bash, edit, write]
    tools: list[AgentTool] | None = None
    # Custom tools to register (in addition to built-in tools)
    custom_tools: list[ToolDefinition] | None = None
    
    # Resource loader. When omitted, DefaultResourceLoader is used
    resource_loader: ResourceLoader | None = None
    
    # Session manager
    session_manager: SessionManager | None = None
    
    # Settings manager
    settings_manager: SettingsManager | None = None


@dataclass
class CreateAgentSessionResult:
    """
    Result from create_agent_session.
    Mirrors CreateAgentSessionResult in TypeScript.
    """
    # The created session
    session: AgentSession
    # Extensions result (for UI context setup in interactive mode)
    extensions_result: dict[str, Any] | None = None
    # Warning if session was restored with a different model than saved
    model_fallback_message: str | None = None


async def create_agent_session(
    options: CreateAgentSessionOptions | None = None
) -> CreateAgentSessionResult:
    """
    Create an AgentSession with the specified options.
    Mirrors createAgentSession() in TypeScript.
    
    Example:
        # Minimal - uses defaults
        result = await create_agent_session()
        session = result.session
        
        # With explicit model
        from pi_ai import get_model
        result = await create_agent_session(
            CreateAgentSessionOptions(
                model=get_model('anthropic', 'claude-opus-4-5'),
                thinking_level='high',
            )
        )
    """
    if options is None:
        options = CreateAgentSessionOptions()
    
    cwd = options.cwd or os.getcwd()
    agent_dir = options.agent_dir  # Will use default in components if None
    
    # Use provided or create AuthStorage and ModelRegistry
    auth_storage = options.auth_storage or AuthStorage()
    model_registry = options.model_registry or ModelRegistry()
    
    settings_manager = options.settings_manager or SettingsManager.create(cwd=cwd, agent_dir=agent_dir)
    session_manager = options.session_manager or SessionManager.create(cwd=cwd)
    
    resource_loader = options.resource_loader
    if not resource_loader:
        # Import here to avoid circular dependency
        from .resource_loader import DefaultResourceLoader, DefaultResourceLoaderOptions
        loader_opts = DefaultResourceLoaderOptions(
            cwd=cwd,
            agent_dir=agent_dir,
            settings_manager=settings_manager,
        )
        resource_loader = DefaultResourceLoader(loader_opts)
        await resource_loader.reload()
    
    # Check if session has existing data to restore
    existing_session = session_manager.build_context()
    has_existing_session = len(existing_session.messages) > 0
    
    model = options.model
    model_fallback_message: str | None = None
    
    # If session has data, try to restore model from it
    if not model and has_existing_session:
        saved_model = existing_session.model
        if saved_model:
            provider = saved_model.get("provider") if isinstance(saved_model, dict) else None
            model_id = saved_model.get("model_id") if isinstance(saved_model, dict) else None
            if provider and model_id:
                model, model_fallback_message = await restore_model_from_session(
                    provider, model_id, None, model_registry
                )

    # If still no model, use find_initial_model (settings default, then provider defaults)
    if not model:
        initial = await find_initial_model(
            is_continuing=has_existing_session,
            default_provider=settings_manager.get_default_provider(),
            default_model_id=settings_manager.get_default_model(),
            default_thinking_level=settings_manager.get_default_thinking_level(),
            model_thinking_levels=settings_manager.get_all_model_thinking_levels(),
            model_registry=model_registry,
        )
        model = initial.model
        if initial.fallback_message:
            if model_fallback_message:
                model_fallback_message = f"{model_fallback_message}. {initial.fallback_message}"
            else:
                model_fallback_message = initial.fallback_message
        # Tests and unauthenticated local use still need a built-in model.
        if not model:
            try:
                model = model_registry.resolve_model(
                    model_id=settings_manager.get_default_model(),
                    provider=settings_manager.get_default_provider(),
                )
            except Exception:
                pass
    
    has_thinking_entry = any(
        getattr(e, "type", None) == "thinking_level_change"
        for e in session_manager.get_branch()
    )

    thinking_level = options.thinking_level
    if thinking_level is None and has_existing_session:
        thinking_level = (
            existing_session.thinking_level
            if has_thinking_entry
            else (settings_manager.get_default_thinking_level() or DEFAULT_THINKING_LEVEL)
        )
    if thinking_level is None and model is not None:
        per_model = settings_manager.get_model_thinking_level(model.provider, model.id)
        if per_model:
            thinking_level = per_model
    if thinking_level is None:
        thinking_level = settings_manager.get_default_thinking_level() or DEFAULT_THINKING_LEVEL

    # Clamp to model capabilities (mirrors clampThinkingLevel in TypeScript)
    if model:
        thinking_level = clamp_thinking_level(model, thinking_level)
    else:
        thinking_level = "off"
    
    settings = Settings(
        thinking_level=thinking_level,
        model_id=model.id if model else None,
        provider=model.provider if model else None,
    )
    
    # Create session with all options
    session = AgentSession(
        cwd=cwd,
        model=model,
        settings=settings,
        session_manager=session_manager,
        auth_storage=auth_storage,
        model_registry=model_registry,
        settings_manager=settings_manager,
    )
    
    # Apply scoped models if provided
    if options.scoped_models:
        session.set_scoped_models(options.scoped_models)
    
    # Apply custom tools if provided
    if options.tools is not None:
        tool_names = [t.name for t in options.tools]
        session.set_active_tools_by_name(tool_names)
    
    extensions_result = resource_loader.get_extensions() if hasattr(resource_loader, "get_extensions") else None
    
    return CreateAgentSessionResult(
        session=session,
        extensions_result=extensions_result,
        model_fallback_message=model_fallback_message,
    )
