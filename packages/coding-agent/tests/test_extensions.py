"""
Tests for core/extensions/ subpackage.

Covers: types.py, loader.py, runner.py, wrapper.py
"""
from __future__ import annotations

import asyncio
import os
import tempfile

import pytest


# ============================================================================
# Types
# ============================================================================

class TestExtensionTypes:
    def test_tool_definition_creation(self):
        from pi_coding_agent.core.extensions.types import ToolDefinition
        # ToolDefinition has: name, label, description, parameters, execute, render_call, render_result
        tool = ToolDefinition(
            name="my_tool",
            label="My Tool",
            description="A tool",
            parameters={"type": "object", "properties": {}},
        )
        assert tool.name == "my_tool"
        assert tool.description == "A tool"

    def test_registered_command_creation(self):
        from pi_coding_agent.core.extensions.types import RegisteredCommand
        cmd = RegisteredCommand(
            name="my-cmd",
            description="My command",
            handler=lambda args, ctx: None,
        )
        assert cmd.name == "my-cmd"

    def test_extension_flag_creation(self):
        from pi_coding_agent.core.extensions.types import ExtensionFlag
        # ExtensionFlag has: name, description, type, default, extension_path
        flag = ExtensionFlag(
            name="verbose",
            description="Enable verbose mode",
            type="boolean",
            default=False,
            extension_path="/path/to/ext.py",
        )
        assert flag.name == "verbose"
        assert flag.type == "boolean"

    def test_session_start_event(self):
        from pi_coding_agent.core.extensions.types import SessionStartEvent
        event = SessionStartEvent()
        assert event.type == "session_start"

    def test_tool_call_event(self):
        from pi_coding_agent.core.extensions.types import ToolCallEvent
        # ToolCallEvent has: tool_call_id, tool_name, input, type
        event = ToolCallEvent(
            tool_call_id="tc-1",
            tool_name="bash",
            input={"command": "ls"},
        )
        assert event.type == "tool_call"
        assert event.tool_name == "bash"

    def test_input_event(self):
        from pi_coding_agent.core.extensions.types import InputEvent
        # InputEvent has: text, source, images, type
        event = InputEvent(text="Hello", source="interactive")
        assert event.type == "input"
        assert event.text == "Hello"

    def test_tool_call_event_result(self):
        from pi_coding_agent.core.extensions.types import ToolCallEventResult
        result = ToolCallEventResult(block=True, reason="Not allowed")
        assert result.block is True

    def test_context_usage(self):
        from pi_coding_agent.core.extensions.types import ContextUsage
        usage = ContextUsage(tokens=1000, context_window=200000, percent=0.5)
        assert usage.tokens == 1000


# ============================================================================
# Loader
# ============================================================================

class TestExtensionLoader:
    def _write_extension(self, path: str, content: str) -> None:
        with open(path, "w") as f:
            f.write(content)

    @pytest.mark.asyncio
    async def test_load_valid_extension(self):
        from pi_coding_agent.core.event_bus import EventBus
        from pi_coding_agent.core.extensions.loader import (
            create_extension_runtime,
            load_extension_from_path,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_path = os.path.join(tmpdir, "my_extension.py")
            self._write_extension(ext_path, "def extension_factory(api):\n    return {}\n")
            runtime = create_extension_runtime()
            event_bus = EventBus()
            ext = await load_extension_from_path(ext_path, tmpdir, event_bus, runtime)
            assert ext is not None

    @pytest.mark.asyncio
    async def test_load_missing_factory_raises(self):
        from pi_coding_agent.core.event_bus import EventBus
        from pi_coding_agent.core.extensions.loader import (
            create_extension_runtime,
            load_extension_from_path,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_path = os.path.join(tmpdir, "bad.py")
            self._write_extension(ext_path, "# No factory function\nx = 1\n")
            runtime = create_extension_runtime()
            event_bus = EventBus()
            with pytest.raises(ImportError):
                await load_extension_from_path(ext_path, tmpdir, event_bus, runtime)

    @pytest.mark.asyncio
    async def test_load_nonexistent_file_raises(self):
        from pi_coding_agent.core.event_bus import EventBus
        from pi_coding_agent.core.extensions.loader import (
            create_extension_runtime,
            load_extension_from_path,
        )
        runtime = create_extension_runtime()
        event_bus = EventBus()
        with pytest.raises(FileNotFoundError):
            await load_extension_from_path("/nonexistent/extension.py", "/tmp", event_bus, runtime)

    @pytest.mark.asyncio
    async def test_load_extensions_from_list(self):
        from pi_coding_agent.core.event_bus import EventBus
        from pi_coding_agent.core.extensions.loader import load_extensions
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "ext1.py")
            p2 = os.path.join(tmpdir, "ext2.py")
            for p in [p1, p2]:
                with open(p, "w") as f:
                    f.write("def extension_factory(api): return {}\n")
            event_bus = EventBus()
            result = await load_extensions([p1, p2], tmpdir, event_bus)
            assert len(result.extensions) == 2

    def test_create_extension_runtime(self):
        from pi_coding_agent.core.extensions.loader import create_extension_runtime
        runtime = create_extension_runtime()
        assert runtime is not None


# ============================================================================
# Wrapper
# ============================================================================

class TestToolWrapper:
    @pytest.mark.asyncio
    async def test_wrap_tool_with_extensions_passthrough(self):
        from unittest.mock import MagicMock

        async def _execute(tool_call_id, params, cancel_event=None, on_update=None):
            return {"content": [{"type": "text", "text": "result"}]}

        tool = {"name": "my_tool", "label": "My Tool", "description": "d",
                "parameters": {}, "execute": _execute}
        mock_runner = MagicMock()
        mock_runner.has_handlers.return_value = False

        from pi_coding_agent.core.extensions.wrapper import wrap_tool_with_extensions
        wrapped = wrap_tool_with_extensions(tool, mock_runner)
        result = await wrapped["execute"]("tcid", {"key": "value"})
        assert result["content"][0]["text"] == "result"

    def test_wrap_tools_with_extensions_returns_list(self):
        from unittest.mock import MagicMock

        async def _execute(tool_call_id, params, cancel_event=None, on_update=None):
            return {}

        tools = [
            {"name": "t1", "label": "T1", "description": "d", "parameters": {}, "execute": _execute},
            {"name": "t2", "label": "T2", "description": "d", "parameters": {}, "execute": _execute},
        ]
        mock_runner = MagicMock()
        mock_runner.has_handlers.return_value = False

        from pi_coding_agent.core.extensions.wrapper import wrap_tools_with_extensions
        wrapped = wrap_tools_with_extensions(tools, mock_runner)
        assert len(wrapped) == 2
        assert wrapped[0]["name"] == "t1"
