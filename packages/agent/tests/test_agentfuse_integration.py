"""Real pi-agent dispatch tests for the optional AgentFuse tool guard."""

from __future__ import annotations

import asyncio
import time

import pytest
from dhms_agentfuse import RuntimeGuard, RuntimeGuardDecision, ToolCallRequest
from pi_agent.agent_loop import _execute_tool_calls
from pi_agent.integrations.agentfuse import AgentFuseToolGuard
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import AssistantMessage, TextContent, ToolCall, Usage
from pi_ai.utils.event_stream import EventStream


class RecordingRuntimeGuard(RuntimeGuard):
    """Keep completed decisions inside one test instead of production state."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.recorded_decisions: list[RuntimeGuardDecision] = []

    async def aevaluate(self, tool_call: ToolCallRequest) -> RuntimeGuardDecision:
        decision = await super().aevaluate(tool_call)
        self.recorded_decisions.append(decision)
        return decision


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


async def _run_guarded_call(
    guard: RuntimeGuard,
    call_id: str,
    *,
    handler_error: Exception | None = None,
):
    handler_calls: list[tuple[str, dict[str, str]]] = []

    async def protected_write(tool_call_id, params, cancel_event=None, on_update=None):
        del cancel_event, on_update
        handler_calls.append((tool_call_id, params))
        if handler_error is not None:
            raise handler_error
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
    execution = await _execute_tool_calls(
        [AgentFuseToolGuard(guard).wrap(tool)],
        _assistant_tool_call(call_id),
        None,
        EventStream(),
    )
    return handler_calls, execution["tool_results"][0]


@pytest.mark.asyncio
async def test_allow_dispatches_once_and_preserves_identity():
    guard = RecordingRuntimeGuard(allow_tools={"protected_write"})
    calls, result = await _run_guarded_call(guard, "pi-allow-001")

    assert calls == [("pi-allow-001", {"value": "synthetic-value"})]
    assert result.tool_call_id == "pi-allow-001"
    assert result.is_error is False
    assert result.details == {"host_outcome": "executed"}
    assert guard.recorded_decisions[0].action == "allow"


@pytest.mark.asyncio
async def test_block_returns_terminal_result_without_dispatch():
    guard = RecordingRuntimeGuard(deny_tools={"protected_write"})
    calls, result = await _run_guarded_call(guard, "pi-block-001")

    assert calls == []
    assert result.tool_call_id == "pi-block-001"
    assert result.is_error is False
    assert result.details["agentfuse_decision"]["action"] == "block"
    assert result.details["host_execution"] == {
        "outcome": "not_executed",
        "handler_started": False,
    }
    assert guard.recorded_decisions[0].tool_call_id == "pi-block-001"


@pytest.mark.asyncio
async def test_guard_failure_is_fail_closed_without_dispatch():
    async def failing_policy(tool_call):
        del tool_call
        raise RuntimeError("synthetic policy failure")

    guard = RecordingRuntimeGuard(policy=failing_policy)
    calls, result = await _run_guarded_call(guard, "pi-failure-001")

    assert calls == []
    assert result.tool_call_id == "pi-failure-001"
    assert result.is_error is False
    assert result.details["agentfuse_decision"]["action"] == "block"
    assert result.details["agentfuse_decision"]["reason_code"] == "policy_exception"
    assert result.details["host_execution"] == {
        "outcome": "not_executed",
        "handler_started": False,
    }
    assert guard.recorded_decisions[0].reason_code == "policy_exception"


@pytest.mark.asyncio
async def test_adapter_guard_evaluation_exception_returns_non_execution():
    class RaisingRuntimeGuard(RuntimeGuard):
        async def aevaluate(self, tool_call: ToolCallRequest) -> RuntimeGuardDecision:
            del tool_call
            raise RuntimeError("synthetic adapter-boundary failure")

    calls, result = await _run_guarded_call(RaisingRuntimeGuard(), "pi-raise-001")

    assert calls == []
    assert result.tool_call_id == "pi-raise-001"
    assert result.is_error is False
    assert result.details == {
        "guard_failed": True,
        "policy_denied": False,
        "tool_failure": False,
        "reason_code": "guard_evaluation_exception",
        "tool_call_id": "pi-raise-001",
        "host_execution": {
            "outcome": "not_executed",
            "handler_started": False,
        },
    }


@pytest.mark.asyncio
async def test_handler_failure_after_allow_remains_host_owned():
    guard = RecordingRuntimeGuard(allow_tools={"protected_write"})
    calls, result = await _run_guarded_call(
        guard,
        "pi-handler-failure-001",
        handler_error=RuntimeError("synthetic handler failure"),
    )

    assert calls == [("pi-handler-failure-001", {"value": "synthetic-value"})]
    assert result.tool_call_id == "pi-handler-failure-001"
    assert result.is_error is True
    assert result.details == {}
    assert guard.recorded_decisions[0].action == "allow"


@pytest.mark.asyncio
async def test_guard_cancellation_remains_host_owned():
    class CancellingRuntimeGuard(RuntimeGuard):
        async def aevaluate(self, tool_call: ToolCallRequest) -> RuntimeGuardDecision:
            del tool_call
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await _run_guarded_call(CancellingRuntimeGuard(), "pi-cancel-001")
