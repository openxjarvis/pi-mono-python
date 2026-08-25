"""
Cwd-bound runtime services — mirrors packages/coding-agent/src/core/agent-session-services.ts
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from pi_coding_agent.config import get_agent_dir

from .model_runtime import CreateModelRuntimeOptions, ModelRuntime
from .resource_loader import DefaultResourceLoader, DefaultResourceLoaderOptions
from .sdk import CreateAgentSessionOptions, CreateAgentSessionResult, create_agent_session
from .session_manager import SessionManager
from .settings_manager import SettingsManager


@dataclass
class AgentSessionRuntimeDiagnostic:
    type: Literal["info", "warning", "error"]
    message: str


@dataclass
class CreateAgentSessionServicesOptions:
    cwd: str
    agent_dir: str | None = None
    settings_manager: SettingsManager | None = None
    model_runtime: ModelRuntime | None = None
    extension_flag_values: dict[str, bool | str] | None = None
    resource_loader_options: dict[str, Any] | None = None


@dataclass
class AgentSessionServices:
    cwd: str
    agent_dir: str
    model_runtime: ModelRuntime
    settings_manager: SettingsManager
    resource_loader: DefaultResourceLoader | None
    diagnostics: list[AgentSessionRuntimeDiagnostic] = field(default_factory=list)


@dataclass
class CreateAgentSessionFromServicesOptions:
    services: AgentSessionServices
    session_manager: SessionManager
    session_start_event: Any | None = None
    model: Any | None = None
    thinking_level: str | None = None
    scoped_models: list[Any] | None = None
    tools: list[str] | None = None
    custom_tools: list[Any] | None = None


def apply_extension_flag_values(
    resource_loader: Any,
    extension_flag_values: dict[str, bool | str] | None,
) -> list[AgentSessionRuntimeDiagnostic]:
    if not extension_flag_values:
        return []
    diagnostics: list[AgentSessionRuntimeDiagnostic] = []
    get_extensions = getattr(resource_loader, "get_extensions", None)
    if not callable(get_extensions):
        return diagnostics
    extensions_result = get_extensions()
    registered: dict[str, str] = {}
    extensions = []
    if isinstance(extensions_result, dict):
        extensions = extensions_result.get("extensions") or []
    else:
        extensions = getattr(extensions_result, "extensions", []) or []
    for extension in extensions:
        flags = getattr(extension, "flags", {}) or {}
        if isinstance(flags, dict):
            for name, flag in flags.items():
                registered[name] = getattr(flag, "type", "boolean")
    unknown: list[str] = []
    runtime = getattr(extensions_result, "runtime", None)
    flag_values = getattr(runtime, "flag_values", None) if runtime is not None else None
    for name, value in extension_flag_values.items():
        flag_type = registered.get(name)
        if not flag_type:
            unknown.append(name)
            continue
        if flag_values is not None and hasattr(flag_values, "set"):
            flag_values.set(name, True if flag_type == "boolean" else value)
        elif isinstance(flag_values, dict):
            flag_values[name] = True if flag_type == "boolean" else value
    if unknown:
        prefix = "Unknown option" if len(unknown) == 1 else "Unknown options"
        diagnostics.append(
            AgentSessionRuntimeDiagnostic(
                type="error",
                message=f"{prefix}: {', '.join(f'--{name}' for name in unknown)}",
            )
        )
    return diagnostics


async def create_agent_session_services(
    options: CreateAgentSessionServicesOptions,
) -> AgentSessionServices:
    cwd = os.path.abspath(options.cwd)
    agent_dir = os.path.abspath(options.agent_dir or get_agent_dir())
    model_runtime = options.model_runtime or await ModelRuntime.create(
        CreateModelRuntimeOptions(
            auth_path=os.path.join(agent_dir, "auth.json"),
            models_path=os.path.join(agent_dir, "models.json"),
            allow_model_network=False,
        )
    )
    settings_manager = options.settings_manager or SettingsManager.create(cwd=cwd, agent_dir=agent_dir)
    resource_loader: DefaultResourceLoader | None = None
    try:
        loader_opts = DefaultResourceLoaderOptions(
            cwd=cwd,
            agent_dir=agent_dir,
            settings_manager=settings_manager,
            **(options.resource_loader_options or {}),
        )
        resource_loader = DefaultResourceLoader(loader_opts)
        if hasattr(resource_loader, "reload"):
            await resource_loader.reload()
    except Exception:
        resource_loader = None

    diagnostics: list[AgentSessionRuntimeDiagnostic] = []
    if resource_loader is not None:
        diagnostics.extend(apply_extension_flag_values(resource_loader, options.extension_flag_values))
    return AgentSessionServices(
        cwd=cwd,
        agent_dir=agent_dir,
        model_runtime=model_runtime,
        settings_manager=settings_manager,
        resource_loader=resource_loader,
        diagnostics=diagnostics,
    )


async def create_agent_session_from_services(
    options: CreateAgentSessionFromServicesOptions,
) -> CreateAgentSessionResult:
    return await create_agent_session(
        CreateAgentSessionOptions(
            cwd=options.services.cwd,
            agent_dir=options.services.agent_dir,
            settings_manager=options.services.settings_manager,
            session_manager=options.session_manager,
            model=options.model,
            thinking_level=options.thinking_level,
            scoped_models=options.scoped_models,
            custom_tools=options.custom_tools,
            resource_loader=options.services.resource_loader,
        )
    )
