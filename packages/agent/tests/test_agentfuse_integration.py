"""Real pi-agent dispatch tests for the optional AgentFuse tool guard."""

from __future__ import annotations

import time

import pytest
from dhms_agentfuse import RuntimeGuard
from pi_agent.agent_loop import _execute_tool_calls
from pi_agent.integrations.agentfuse import AgentFuseToolGuard
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import AssistantMessage, TextContent, ToolCall, Usage
from pi_ai.utils.event_stream import EventStream


def _assistant_tool_call(call_id: str) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[
            ToolCall(
                type="toolCall",
                id=call_id,
                name="protected_write",
                arguments={"value": "synthetic-value"},
            )
        ],
        api="anthropic-messages",
        provider="anthropic",
        model="agentfuse-test-model",
        usage=Usage(),
        stop_reason="toolUse",
        timestamp=int(time.time() * 1000),
    )


async def _run_guarded_call(guard: RuntimeGuard, call_id: str):
    handler_calls: list[tuple[str, dict[str, str]]] = []

    async def protected_write(tool_call_id, params, cancel_event=None, on_update=None):
        del cancel_event, on_update
        handler_calls.append((tool_call_id, params))
        return AgentToolResult(
            content=[TextContent(type="text", text="write completed")],
            details={"host_outcome": "executed"},
        )

    tool = AgentTool(
        name="protected_write",
        label="Protected write",
        description="Write one synthetic value",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        execute=protected_write,
    )
    integration = AgentFuseToolGuard(guard)
    execution = await _execute_tool_calls(
        [integration.wrap(tool)],
        _assistant_tool_call(call_id),
        None,
        EventStream(),
    )
    return handler_calls, execution["tool_results"][0], integration


@pytest.mark.asyncio
async def test_allow_dispatches_once_and_preserves_identity():
    calls, result, integration = await _run_guarded_call(RuntimeGuard(allow_tools={"protected_write"}), "pi-allow-001")

    assert calls == [("pi-allow-001", {"value": "synthetic-value"})]
    assert result.tool_call_id == "pi-allow-001"
    assert result.is_error is False
    assert result.details == {"host_outcome": "executed"}
    assert integration.decision_for("pi-allow-001").action == "allow"


@pytest.mark.asyncio
async def test_block_returns_terminal_result_without_dispatch():
    calls, result, integration = await _run_guarded_call(RuntimeGuard(deny_tools={"protected_write"}), "pi-block-001")

    assert calls == []
    assert result.tool_call_id == "pi-block-001"
    assert result.is_error is False
    assert result.details["agentfuse_decision"]["action"] == "block"
    assert result.details["host_execution"] == {
        "outcome": "not_executed",
        "handler_started": False,
    }
    assert integration.decision_for("pi-block-001").tool_call_id == "pi-block-001"


@pytest.mark.asyncio
async def test_guard_failure_is_fail_closed_without_dispatch():
    async def failing_policy(tool_call):
        del tool_call
        raise RuntimeError("synthetic policy failure")

    calls, result, integration = await _run_guarded_call(RuntimeGuard(policy=failing_policy), "pi-failure-001")

    assert calls == []
    assert result.tool_call_id == "pi-failure-001"
    assert result.is_error is False
    assert result.details["agentfuse_decision"]["action"] == "block"
    assert result.details["agentfuse_decision"]["reason_code"] == "policy_exception"
    assert result.details["host_execution"] == {
        "outcome": "not_executed",
        "handler_started": False,
    }
    assert integration.decision_for("pi-failure-001").reason_code == "policy_exception"
