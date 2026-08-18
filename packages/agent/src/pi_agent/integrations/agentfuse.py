"""Optional AgentFuse policy guard for pi-agent tools."""

from __future__ import annotations

import asyncio
from typing import Any

from pi_ai.types import TextContent

from ..types import AgentTool, AgentToolResult, AgentToolUpdateCallback


class AgentFuseToolGuard:
    """Wrap ``AgentTool`` execution with an AgentFuse pre-dispatch decision."""

    def __init__(self, guard: Any) -> None:
        try:
            from dhms_agentfuse import RuntimeGuard, ToolCallRequest
        except ImportError as exc:
            raise ImportError(
                "AgentFuse support requires the 'agentfuse' optional dependency: pip install 'pi-agent[agentfuse]'"
            ) from exc

        if not isinstance(guard, RuntimeGuard):
            raise TypeError("guard must be a dhms_agentfuse.RuntimeGuard")

        self._guard = guard
        self._request_type = ToolCallRequest
        self._decisions: dict[str, Any] = {}

    def decision_for(self, tool_call_id: str) -> Any | None:
        """Return the completed policy decision for a tool call, if available."""
        return self._decisions.get(tool_call_id)

    def wrap(self, tool: AgentTool) -> AgentTool:
        """Return a copy of ``tool`` guarded immediately before dispatch."""
        original_execute = tool.execute

        async def guarded_execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            decision = await self._guard.aevaluate(
                self._request_type(
                    tool_call_id=tool_call_id,
                    tool_name=tool.name,
                    arguments=params,
                    safe_metadata={"integration": "pi-agent"},
                )
            )
            self._decisions[tool_call_id] = decision

            if decision.action == "block":
                return AgentToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=(f"Tool call blocked before execution: {decision.reason_code}"),
                        )
                    ],
                    details={
                        "agentfuse_decision": decision.to_safe_dict(),
                        "host_execution": {
                            "outcome": "not_executed",
                            "handler_started": False,
                        },
                    },
                )

            return await original_execute(tool_call_id, params, cancel_event, on_update)

        return tool.model_copy(update={"execute": guarded_execute})
