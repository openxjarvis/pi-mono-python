"""
Tests for modes/rpc/ subpackage.

Covers: types.py (pydantic models), client.py API surface
"""
from __future__ import annotations

import pytest


# ============================================================================
# RPC Types
# ============================================================================

class TestRpcTypes:
    def test_rpc_command_prompt_serializes(self):
        from pi_coding_agent.modes.rpc.types import RpcCommandPrompt
        cmd = RpcCommandPrompt(type="prompt", message="Hello world")
        data = cmd.model_dump()
        assert data["type"] == "prompt"
        assert data["message"] == "Hello world"

    def test_rpc_command_steer_serializes(self):
        from pi_coding_agent.modes.rpc.types import RpcCommandSteer
        cmd = RpcCommandSteer(type="steer", message="Override")
        data = cmd.model_dump()
        assert data["type"] == "steer"

    def test_rpc_command_abort_serializes(self):
        from pi_coding_agent.modes.rpc.types import RpcCommandAbort
        cmd = RpcCommandAbort(type="abort")
        data = cmd.model_dump()
        assert data["type"] == "abort"

    def test_rpc_response_success(self):
        from pi_coding_agent.modes.rpc.types import RpcResponseSuccess
        r = RpcResponseSuccess(command="get_state", data={"thinkingLevel": "off"})
        assert r.success is True
        assert r.command == "get_state"

    def test_rpc_response_error(self):
        from pi_coding_agent.modes.rpc.types import RpcResponseError
        r = RpcResponseError(command="prompt", error="Something went wrong")
        assert r.success is False
        assert "Something went wrong" in r.error

    def test_rpc_session_state_fields(self):
        from pi_coding_agent.modes.rpc.types import RpcSessionState
        state = RpcSessionState(
            thinkingLevel="off",
            isStreaming=False,
            isCompacting=False,
            steeringMode="all",
            followUpMode="all",
            sessionId="abc-123",
            autoCompactionEnabled=True,
            messageCount=5,
            pendingMessageCount=0,
        )
        assert state.sessionId == "abc-123"
        assert state.messageCount == 5

    def test_rpc_slash_command(self):
        from pi_coding_agent.modes.rpc.types import RpcSlashCommand
        cmd = RpcSlashCommand(name="test", source="skill", description="A skill cmd")
        assert cmd.name == "test"
        assert cmd.source == "skill"

    def test_rpc_extension_ui_request_notify(self):
        from pi_coding_agent.modes.rpc.types import RpcExtensionUIRequestNotify
        req = RpcExtensionUIRequestNotify(id="abc", method="notify", message="Hello!")
        data = req.model_dump()
        assert data["type"] == "extension_ui_request"
        assert data["method"] == "notify"

    def test_rpc_extension_ui_request_select(self):
        from pi_coding_agent.modes.rpc.types import RpcExtensionUIRequestSelect
        req = RpcExtensionUIRequestSelect(id="x", method="select", title="Choose", options=["A", "B"])
        assert req.options == ["A", "B"]

    def test_rpc_extension_ui_response_value(self):
        from pi_coding_agent.modes.rpc.types import RpcExtensionUIResponseValue
        resp = RpcExtensionUIResponseValue(id="x", value="chosen")
        assert resp.value == "chosen"

    def test_rpc_extension_ui_response_cancelled(self):
        from pi_coding_agent.modes.rpc.types import RpcExtensionUIResponseCancelled
        resp = RpcExtensionUIResponseCancelled(id="x", cancelled=True)
        assert resp.cancelled is True

    def test_rpc_command_set_model(self):
        from pi_coding_agent.modes.rpc.types import RpcCommandSetModel
        cmd = RpcCommandSetModel(type="set_model", provider="anthropic", modelId="claude-3-5-sonnet")
        assert cmd.provider == "anthropic"
        assert cmd.modelId == "claude-3-5-sonnet"

    def test_rpc_command_compact(self):
        from pi_coding_agent.modes.rpc.types import RpcCommandCompact
        cmd = RpcCommandCompact(type="compact", customInstructions="Focus on recent changes")
        assert cmd.customInstructions == "Focus on recent changes"


# ============================================================================
# RpcClient instantiation and API surface
# ============================================================================

class TestRpcClientAPI:
    def test_rpc_client_instantiation(self):
        from pi_coding_agent.modes.rpc.client import RpcClient, RpcClientOptions
        opts = RpcClientOptions(cwd="/tmp", provider="anthropic")
        client = RpcClient(opts)
        assert client is not None

    def test_rpc_client_default_options(self):
        from pi_coding_agent.modes.rpc.client import RpcClient
        client = RpcClient()
        assert client is not None

    def test_rpc_client_on_event_returns_unsubscribe(self):
        from pi_coding_agent.modes.rpc.client import RpcClient
        client = RpcClient()
        events = []
        unsub = client.on_event(lambda e: events.append(e))
        assert callable(unsub)
        # Manually trigger the listener
        client._handle_line({"type": "agent_start"})
        assert len(events) == 1
        unsub()
        client._handle_line({"type": "agent_end"})
        assert len(events) == 1  # No new events after unsubscribe

    def test_rpc_client_not_started_raises(self):
        from pi_coding_agent.modes.rpc.client import RpcClient
        import asyncio
        client = RpcClient()
        with pytest.raises((RuntimeError, Exception)):
            asyncio.get_event_loop().run_until_complete(client.prompt("hello"))

    def test_rpc_client_get_stderr_empty(self):
        from pi_coding_agent.modes.rpc.client import RpcClient
        client = RpcClient()
        assert client.get_stderr() == ""
