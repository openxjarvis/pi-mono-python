"""
Tool wrappers for extensions.

Wraps agent tools with extension event callbacks, allowing extensions to
intercept tool calls and results.

Mirrors core/extensions/wrapper.ts
"""

from __future__ import annotations

from typing import Any, Callable


def wrap_registered_tool(registered_tool: Any, runner: Any) -> dict[str, Any]:
    """Wrap a RegisteredTool definition into an AgentTool-compatible dict."""
    definition = registered_tool if not hasattr(registered_tool, "definition") else registered_tool.definition

    async def _execute(tool_call_id: str, params: dict, cancel_event: Any = None, on_update: Any = None) -> Any:
        ctx = runner.create_context()
        return await definition.execute(tool_call_id, params, cancel_event, on_update, ctx)

    return {
        "name": definition.name,
        "label": definition.label,
        "description": definition.description,
        "parameters": definition.parameters,
        "execute": _execute,
    }


def wrap_registered_tools(registered_tools: list[Any], runner: Any) -> list[dict[str, Any]]:
    """Wrap all registered tools into AgentTool dicts."""
    return [wrap_registered_tool(rt, runner) for rt in registered_tools]


def wrap_tool_with_extensions(tool: dict[str, Any], runner: Any) -> dict[str, Any]:
    """Wrap an agent tool dict with extension interception callbacks.

    - Emits tool_call before execution (can block)
    - Emits tool_result after execution (can modify result)
    """
    from pi_coding_agent.core.extensions.types import ToolCallEvent, ToolResultEvent

    original_execute = tool["execute"]

    async def _wrapped_execute(
        tool_call_id: str,
        params: dict,
        cancel_event: Any = None,
        on_update: Any = None,
    ) -> Any:
        if runner.has_handlers("tool_call"):
            call_event = ToolCallEvent(
                tool_call_id=tool_call_id,
                tool_name=tool["name"],
                input=params,
            )
            call_result = await runner.emit_tool_call(call_event)
            if call_result and getattr(call_result, "block", False):
                reason = getattr(call_result, "reason", None) or "Tool execution was blocked by an extension"
                raise RuntimeError(reason)

        try:
            result = await original_execute(tool_call_id, params, cancel_event, on_update)

            if runner.has_handlers("tool_result"):
                content = result.get("content", []) if isinstance(result, dict) else getattr(result, "content", [])
                details = result.get("details") if isinstance(result, dict) else getattr(result, "details", None)
                result_event = ToolResultEvent(
                    tool_call_id=tool_call_id,
                    tool_name=tool["name"],
                    input=params,
                    content=content,
                    is_error=False,
                    details=details,
                )
                result_result = await runner.emit_tool_result(result_event)
                if result_result:
                    new_content = getattr(result_result, "content", None)
                    new_details = getattr(result_result, "details", None)
                    if isinstance(result, dict):
                        out = dict(result)
                        if new_content is not None:
                            out["content"] = new_content
                        if new_details is not None:
                            out["details"] = new_details
                        return out
                    else:
                        if new_content is not None:
                            result.content = new_content
                        if new_details is not None:
                            result.details = new_details

            return result

        except Exception as err:
            if runner.has_handlers("tool_result"):
                err_event = ToolResultEvent(
                    tool_call_id=tool_call_id,
                    tool_name=tool["name"],
                    input=params,
                    content=[{"type": "text", "text": str(err)}],
                    is_error=True,
                )
                await runner.emit_tool_result(err_event)
            raise

    return {**tool, "execute": _wrapped_execute}


def wrap_tools_with_extensions(tools: list[dict[str, Any]], runner: Any) -> list[dict[str, Any]]:
    """Wrap all agent tools with extension interception."""
    return [wrap_tool_with_extensions(tool, runner) for tool in tools]
