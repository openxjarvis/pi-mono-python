"""
Project trust resolution — mirrors packages/coding-agent/src/core/project-trust.ts
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Protocol

from pi_coding_agent.config import APP_NAME, CONFIG_DIR_NAME

from .trust_manager import (
    ProjectTrustOption,
    ProjectTrustStore,
    get_project_trust_options,
    has_trust_requiring_project_resources,
)

AppMode = Literal["interactive", "print", "json", "rpc"]
DefaultProjectTrust = Literal["ask", "always", "never"]


class ProjectTrustUI(Protocol):
    async def select(self, title: str, options: list[str]) -> str | None: ...


class ProjectTrustContext(Protocol):
    has_ui: bool
    ui: ProjectTrustUI


@dataclass
class ResolveProjectTrustedOptions:
    cwd: str
    trust_store: ProjectTrustStore
    project_trust_context: ProjectTrustContext
    trust_override: bool | None = None
    default_project_trust: DefaultProjectTrust = "ask"
    extensions_result: Any = None
    on_extension_error: Callable[[str], None] | None = None


def format_project_trust_prompt(cwd: str) -> str:
    return (
        f"Trust project folder?\n{cwd}\n\n"
        f"This allows {APP_NAME} to load {CONFIG_DIR_NAME} settings and resources, "
        "install missing project packages, and execute project extensions."
    )


async def select_project_trust_option(
    cwd: str,
    ctx: ProjectTrustContext,
) -> ProjectTrustOption | None:
    options = get_project_trust_options(cwd, include_session_only=True)
    selected = await ctx.ui.select(
        format_project_trust_prompt(cwd),
        [option.label for option in options],
    )
    return next((option for option in options if option.label == selected), None)


def save_project_trust_prompt_result(trust_store: ProjectTrustStore, result: ProjectTrustOption) -> None:
    if result.updates:
        trust_store.set_many(result.updates)


async def resolve_project_trusted(options: ResolveProjectTrustedOptions) -> bool:
    if options.trust_override is not None:
        return options.trust_override
    if not has_trust_requiring_project_resources(options.cwd):
        return True

    extensions_result = options.extensions_result
    if extensions_result is not None:
        emit = getattr(extensions_result, "emit_project_trust", None)
        if callable(emit):
            result = emit({"type": "project_trust", "cwd": options.cwd})
            if isinstance(result, Awaitable):
                result = await result
            if isinstance(result, dict):
                trusted = result.get("trusted") == "yes" or result.get("trusted") is True
                if result.get("remember") is True:
                    options.trust_store.set(options.cwd, trusted)
                return trusted

    decision = options.trust_store.get(options.cwd)
    if decision is not None:
        return decision

    default = options.default_project_trust or "ask"
    if default == "always":
        return True
    if default == "never":
        return False

    if not getattr(options.project_trust_context, "has_ui", False):
        return False

    selected = await select_project_trust_option(options.cwd, options.project_trust_context)
    if selected is not None:
        save_project_trust_prompt_result(options.trust_store, selected)
        return selected.trusted
    return False
