"""
ToolDefinition wrappers — mirrors packages/coding-agent/src/core/tools/tool-definition-wrapper.ts
"""
from __future__ import annotations

from typing import Any, Callable

from pi_coding_agent.core.extensions.types import ToolDefinition


def wrap_tool_definition(
    definition: ToolDefinition,
    ctx_factory: Callable[[], Any] | None = None,
) -> Any:
    """Wrap a ToolDefinition into an AgentTool-like mapping for the core runtime."""
    async def execute(tool_call_id: str, params: Any, signal: Any = None, on_update: Any = None, ctx: Any = None) -> Any:
        if definition.execute is None:
            raise RuntimeError(f"Tool '{definition.name}' has no execute handler")
        return await definition.execute(
            tool_call_id,
            params,
            signal,
            on_update,
            ctx if ctx is not None else (ctx_factory() if ctx_factory else None),
        )

    return {
        "name": definition.name,
        "label": definition.label,
        "description": definition.description,
        "parameters": definition.parameters,
        "execute": execute,
        "prompt_snippet": definition.prompt_snippet,
        "prompt_guidelines": definition.prompt_guidelines,
    }


def wrap_tool_definitions(
    definitions: list[ToolDefinition],
    ctx_factory: Callable[[], Any] | None = None,
) -> list[Any]:
    return [wrap_tool_definition(definition, ctx_factory) for definition in definitions]


def create_tool_definition_from_agent_tool(tool: Any) -> ToolDefinition:
    async def execute(tool_call_id: str, params: Any, signal: Any = None, on_update: Any = None, ctx: Any = None) -> Any:
        return await tool.execute(tool_call_id, params, signal, on_update)

    return ToolDefinition(
        name=getattr(tool, "name", ""),
        label=getattr(tool, "label", "") or "",
        description=getattr(tool, "description", "") or "",
        parameters=getattr(tool, "parameters", {}) or {},
        execute=execute,
    )
