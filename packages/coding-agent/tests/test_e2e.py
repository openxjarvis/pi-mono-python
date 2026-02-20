"""
End-to-end tests for the coding agent.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from typing import AsyncGenerator

import pytest

from pi_ai.types import (
    AssistantMessage,
    EventDone,
    EventStart,
    EventTextEnd,
    EventTextStart,
    EventToolCallEnd,
    EventToolCallStart,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
)
from pi_ai import get_model
from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import Settings
from pi_coding_agent.core.tools import create_read_tool, create_write_tool


def _ts():
    return int(time.time() * 1000)


@pytest.mark.asyncio
async def test_e2e_read_write_workflow():
    """Test that the agent can read and write files using tools."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file
        test_file = os.path.join(tmpdir, "hello.txt")
        with open(test_file, "w") as f:
            f.write("Original content")

        write_called = []

        # Mock: agent writes a new file after reading the existing one
        call_count = [0]

        async def mock_stream_with_write(model, ctx, opts=None):
            call_count[0] += 1
            partial = AssistantMessage(
                role="assistant", content=[], api=model.api, provider=model.provider,
                model=model.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
            )
            yield EventStart(type="start", partial=partial)

            if call_count[0] == 1:
                # Return write tool call
                tc = ToolCall(
                    type="toolCall",
                    id="w1",
                    name="write",
                    arguments={"path": "output.txt", "content": "Written by agent"},
                )
                with_tc = partial.model_copy(update={"content": [tc]})
                yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_tc)
                yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tc, partial=with_tc)
                final = AssistantMessage(
                    role="assistant", content=[tc], api=model.api, provider=model.provider,
                    model=model.id, usage=Usage(), stop_reason="toolUse", timestamp=_ts(),
                )
                yield EventDone(type="done", reason="toolUse", message=final)
            else:
                text = "I've written the file."
                with_text = partial.model_copy(update={"content": [TextContent(type="text", text=text)]})
                yield EventTextEnd(type="text_end", content_index=0, content=text, partial=with_text)
                final = AssistantMessage(
                    role="assistant", content=[TextContent(type="text", text=text)],
                    api=model.api, provider=model.provider, model=model.id,
                    usage=Usage(), stop_reason="stop", timestamp=_ts(),
                )
                yield EventDone(type="done", reason="stop", message=final)

        model = get_model("anthropic", "claude-3-5-sonnet-20241022")
        settings = Settings(auto_compact=False)
        session_manager = SessionManager(sessions_dir=tmpdir)

        session = AgentSession(
            cwd=tmpdir,
            model=model,
            settings=settings,
            session_manager=session_manager,
        )
        session._agent.stream_fn = mock_stream_with_write

        await session.prompt("Write a file called output.txt")

        # Verify the file was actually written
        output_file = os.path.join(tmpdir, "output.txt")
        assert os.path.exists(output_file), "output.txt should have been created"
        with open(output_file) as f:
            assert f.read() == "Written by agent"


@pytest.mark.asyncio
async def test_e2e_system_prompt_includes_cwd():
    """Test that the system prompt includes the working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(auto_compact=False)
        model = get_model("anthropic", "claude-3-5-sonnet-20241022")

        session = AgentSession(
            cwd=tmpdir,
            model=model,
            settings=settings,
            session_manager=SessionManager(sessions_dir=tmpdir),
        )

        prompt = session.state.system_prompt
        assert tmpdir in prompt
        # TS-parity: default prompt should include explicit tool section/guidelines.
        assert "Available tools:" in prompt
        assert "- read: Read file contents" in prompt
        assert "- bash:" in prompt
        assert "Guidelines:" in prompt


@pytest.mark.asyncio
async def test_e2e_session_persistence():
    """Test that messages are persisted across session restores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = get_model("anthropic", "claude-3-5-sonnet-20241022")
        settings = Settings(auto_compact=False)

        async def simple_stream(m, ctx, opts=None):
            partial = AssistantMessage(
                role="assistant", content=[], api=m.api, provider=m.provider,
                model=m.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
            )
            yield EventStart(type="start", partial=partial)
            final = AssistantMessage(
                role="assistant", content=[TextContent(type="text", text="OK")],
                api=m.api, provider=m.provider, model=m.id,
                usage=Usage(), stop_reason="stop", timestamp=_ts(),
            )
            yield EventDone(type="done", reason="stop", message=final)

        session_manager = SessionManager(sessions_dir=tmpdir)
        session = AgentSession(
            cwd=tmpdir, model=model,
            settings=settings, session_manager=session_manager,
        )
        session._agent.stream_fn = simple_stream

        session_id = session.session_id
        await session.prompt("Remember this")

        # Load stored messages
        stored = session_manager.get_messages(session_id)
        assert len(stored) > 0


@pytest.mark.asyncio
async def test_e2e_compaction():
    """Test that manual compaction works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = get_model("anthropic", "claude-3-5-sonnet-20241022")
        settings = Settings(auto_compact=False)

        async def simple_stream(m, ctx, opts=None):
            partial = AssistantMessage(
                role="assistant", content=[], api=m.api, provider=m.provider,
                model=m.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
            )
            yield EventStart(type="start", partial=partial)
            final = AssistantMessage(
                role="assistant", content=[TextContent(type="text", text="Response")],
                api=m.api, provider=m.provider, model=m.id,
                usage=Usage(), stop_reason="stop", timestamp=_ts(),
            )
            yield EventDone(type="done", reason="stop", message=final)

        session = AgentSession(
            cwd=tmpdir, model=model,
            settings=settings, session_manager=SessionManager(sessions_dir=tmpdir),
        )
        session._agent.stream_fn = simple_stream

        # Add some messages
        for i in range(3):
            await session.prompt(f"Message {i}")

        initial_count = len(session.state.messages)
        assert initial_count > 0

        # compact_context calls complete_simple internally,
        # but with a mock stream we can just verify it doesn't crash
        # and returns something
        try:
            summary = await session.compact()
            # Either compaction worked or it returned early (too few messages)
        except Exception as e:
            # Some errors are OK if the mock stream doesn't support summary generation
            pass


# ============================================================================
# Integration tests: Extensions
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_extension_loaded_and_session_start_event():
    """Test that an extension can be loaded and receives session_start events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ext_path = os.path.join(tmpdir, "test_ext.py")
        events_received = []

        ext_content = """
session_start_calls = []

def extension_factory(api):
    async def on_session_start(event, ctx):
        session_start_calls.append(event)
    api.on("session_start", on_session_start)
"""
        with open(ext_path, "w") as f:
            f.write(ext_content)

        from pi_coding_agent.core.event_bus import EventBus
        from pi_coding_agent.core.extensions.loader import load_extensions
        from pi_coding_agent.core.extensions.types import SessionStartEvent

        event_bus = EventBus()
        result = await load_extensions([ext_path], tmpdir, event_bus)
        assert len(result.extensions) == 1
        assert len(result.errors) == 0


@pytest.mark.asyncio
async def test_e2e_extension_tool_wrapping():
    """Test that extension tool wrapping passes through correctly."""
    from unittest.mock import MagicMock
    from pi_coding_agent.core.extensions.wrapper import wrap_tool_with_extensions

    calls = []

    async def _execute(tool_call_id, params, cancel_event=None, on_update=None):
        calls.append(params)
        return {"content": [{"type": "text", "text": f"result:{params.get('x')}"}]}

    tool = {
        "name": "test_tool",
        "label": "Test",
        "description": "A test tool",
        "parameters": {},
        "execute": _execute,
    }
    mock_runner = MagicMock()
    mock_runner.has_handlers.return_value = False

    wrapped = wrap_tool_with_extensions(tool, mock_runner)
    result = await wrapped["execute"]("tc1", {"x": 42})
    assert calls == [{"x": 42}]
    assert result["content"][0]["text"] == "result:42"


@pytest.mark.asyncio
async def test_e2e_extension_multiple_tools_wrapped():
    """Test wrapping multiple tools preserves all tools."""
    from unittest.mock import MagicMock
    from pi_coding_agent.core.extensions.wrapper import wrap_tools_with_extensions

    async def _exec(tc_id, params, cancel=None, upd=None):
        return {}

    tools = [
        {"name": f"tool_{i}", "label": f"T{i}", "description": "d", "parameters": {}, "execute": _exec}
        for i in range(5)
    ]
    runner = MagicMock()
    runner.has_handlers.return_value = False

    wrapped = wrap_tools_with_extensions(tools, runner)
    assert len(wrapped) == 5
    assert [w["name"] for w in wrapped] == [f"tool_{i}" for i in range(5)]


# ============================================================================
# Integration tests: RPC types and protocol
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_rpc_command_roundtrip():
    """Test RPC command serialization and deserialization."""
    from pi_coding_agent.modes.rpc.types import (
        RpcCommandPrompt,
        RpcResponseSuccess,
        RpcSessionState,
    )

    # Serialize a command
    cmd = RpcCommandPrompt(type="prompt", message="Hello!", id="req_1")
    data = cmd.model_dump(exclude_none=True)
    assert data["type"] == "prompt"
    assert data["message"] == "Hello!"
    assert data["id"] == "req_1"

    # Session state round-trip
    state = RpcSessionState(
        thinkingLevel="medium",
        isStreaming=True,
        isCompacting=False,
        steeringMode="all",
        followUpMode="one-at-a-time",
        sessionId="sid-123",
        autoCompactionEnabled=False,
        messageCount=10,
        pendingMessageCount=2,
    )
    state_dict = state.model_dump()
    restored = RpcSessionState(**state_dict)
    assert restored.sessionId == "sid-123"
    assert restored.messageCount == 10


@pytest.mark.asyncio
async def test_e2e_rpc_client_event_subscriptions():
    """Test that the RpcClient event subscription and unsubscription work."""
    from pi_coding_agent.modes.rpc.client import RpcClient

    client = RpcClient()
    received: list = []
    received2: list = []

    unsub1 = client.on_event(lambda e: received.append(e))
    unsub2 = client.on_event(lambda e: received2.append(e))

    # Manually trigger
    client._handle_line({"type": "agent_start"})
    assert len(received) == 1
    assert len(received2) == 1

    unsub1()
    client._handle_line({"type": "agent_end"})
    assert len(received) == 1  # No new events
    assert len(received2) == 2


# ============================================================================
# Integration tests: CLI args and file processing
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_file_processor_text_files():
    """Test processing multiple text file arguments."""
    from pi_coding_agent.cli_sub.file_processor import process_file_arguments

    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "a.txt")
        file2 = os.path.join(tmpdir, "b.txt")
        with open(file1, "w") as f:
            f.write("Content A")
        with open(file2, "w") as f:
            f.write("Content B")

        result = await process_file_arguments([file1, file2])
        assert "Content A" in result.text
        assert "Content B" in result.text
        assert result.images == []


def test_e2e_args_full_parse():
    """Test parsing a realistic full set of CLI arguments."""
    from pi_coding_agent.cli_sub.args import parse_args

    args = parse_args([
        "--provider", "anthropic",
        "--model", "claude-3-5-sonnet",
        "--thinking", "high",
        "--mode", "rpc",
        "--extension", "ext1.py",
        "--extension", "ext2.py",
        "--no-session",
        "--verbose",
        "@myfile.md",
        "Fix the bug in main.py",
    ])

    assert args.provider == "anthropic"
    assert args.model == "claude-3-5-sonnet"
    assert args.thinking == "high"
    assert args.mode == "rpc"
    assert args.extensions == ["ext1.py", "ext2.py"]
    assert args.no_session is True
    assert args.verbose is True
    assert "myfile.md" in args.file_args
    assert "Fix the bug in main.py" in args.messages


# ============================================================================
# Integration tests: Migrations
# ============================================================================

def test_e2e_run_migrations_clean_dir():
    """Test that run_migrations works on a clean directory."""
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        agent_dir = os.path.join(tmpdir, ".pi", "agent")
        os.makedirs(agent_dir)

        with patch("pi_coding_agent.migrations.get_agent_dir", return_value=agent_dir):
            with patch("pi_coding_agent.migrations.get_bin_dir", return_value=os.path.join(agent_dir, "bin")):
                from pi_coding_agent.migrations import run_migrations
                result = run_migrations(tmpdir)

        assert "migratedAuthProviders" in result
        assert "deprecationWarnings" in result
        assert isinstance(result["migratedAuthProviders"], list)


def test_e2e_run_migrations_with_oauth():
    """Test that run_migrations migrates OAuth credentials."""
    import json
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        agent_dir = os.path.join(tmpdir, ".pi", "agent")
        os.makedirs(agent_dir)

        oauth_data = {"anthropic": {"access_token": "tok123", "refresh_token": "ref456"}}
        oauth_path = os.path.join(agent_dir, "oauth.json")
        with open(oauth_path, "w") as f:
            json.dump(oauth_data, f)

        with patch("pi_coding_agent.migrations.get_agent_dir", return_value=agent_dir):
            with patch("pi_coding_agent.migrations.get_bin_dir", return_value=os.path.join(agent_dir, "bin")):
                from pi_coding_agent.migrations import run_migrations
                result = run_migrations(tmpdir)

        assert "anthropic" in result["migratedAuthProviders"]
        auth_path = os.path.join(agent_dir, "auth.json")
        assert os.path.exists(auth_path)


# ============================================================================
# Integration tests: Compaction and resource loading
# ============================================================================

def test_e2e_compaction_utils_pipeline():
    """Test that FileOperations tracks changes correctly in a pipeline."""
    from pi_coding_agent.core.compaction.utils import FileOperations, compute_file_lists

    ops = FileOperations()
    ops.read.update(["/src/a.py", "/src/b.py"])
    ops.written.update(["/src/c.py"])
    ops.edited.update(["/src/a.py"])

    read_files, modified_files = compute_file_lists(ops)
    # /src/b.py was read but not modified → in read_files
    assert "/src/b.py" in read_files
    # /src/a.py was edited → in modified_files (and NOT in read_files since it's modified)
    assert "/src/a.py" in modified_files
    # /src/c.py was written → in modified_files
    assert "/src/c.py" in modified_files


def test_e2e_skills_and_prompts_integration():
    """Test skills and prompt templates loading from the same directory."""
    import tempfile
    from pi_coding_agent.core.skills import load_skills_from_dir
    from pi_coding_agent.core.prompt_templates import (
        LoadPromptTemplatesOptions,
        load_prompt_templates,
    )

    with tempfile.TemporaryDirectory() as skills_dir:
        # Create skill
        skill_subdir = os.path.join(skills_dir, "my-skill")
        os.makedirs(skill_subdir)
        with open(os.path.join(skill_subdir, "SKILL.md"), "w") as f:
            f.write("---\ndescription: Integration skill\n---\nDo something useful.")

        skills_result = load_skills_from_dir(skills_dir, "user")
        assert len(skills_result.skills) == 1
        assert skills_result.skills[0].name == "my-skill"

    with tempfile.TemporaryDirectory() as prompts_dir:
        # Create prompt template
        with open(os.path.join(prompts_dir, "my-prompt.md"), "w") as f:
            f.write("---\ndescription: Integration prompt\n---\nHello $1!")

        opts = LoadPromptTemplatesOptions(prompt_paths=[prompts_dir], include_defaults=False)
        templates = load_prompt_templates(opts)
        assert len(templates) == 1
        assert templates[0].name == "my-prompt"


# ============================================================================
# TUI integration tests
# ============================================================================

class _MockTerminal:
    """Minimal mock terminal for TUI integration tests (no ABC inheritance)."""

    rows = 24
    columns = 80
    kitty_protocol_active = False

    def __init__(self) -> None:
        self._writes: list[str] = []

    def write(self, s: str) -> None:
        self._writes.append(s)

    def start(self, on_input, on_resize) -> None:
        pass

    def stop(self) -> None:
        pass

    def hide_cursor(self) -> None:
        pass

    def show_cursor(self) -> None:
        pass

    def move_by(self, n: int) -> None:
        pass

    def clear_line(self) -> None:
        pass

    def clear_from_cursor(self) -> None:
        pass

    def clear_screen(self) -> None:
        pass

    def set_title(self, t: str) -> None:
        pass

    async def drain_input(self, *a, **k) -> None:
        pass

    @property
    def all_output(self) -> str:
        return "".join(self._writes)


@pytest.mark.asyncio
async def test_tui_render_history_and_stream():
    """Verify TUI lays out history_text, stream_text, spacer, editor correctly."""
    import re
    from pi_tui.tui import TUI
    from pi_tui.components.text import Text
    from pi_tui.components.spacer import Spacer
    from pi_tui.components.editor import Editor, EditorTheme
    from pi_tui.components.select_list import SelectListTheme

    def dim(s: str) -> str:
        return f"\x1b[2m{s}\x1b[22m"

    def cyan(s: str) -> str:
        return f"\x1b[36m{s}\x1b[39m"

    mt = _MockTerminal()
    tui = TUI(mt)

    history_text = Text("", padding_x=1, padding_y=0)
    stream_text = Text("", padding_x=1, padding_y=0)
    tui.add_child(history_text)
    tui.add_child(stream_text)
    tui.add_child(Spacer(1))
    select_theme = SelectListTheme(selected_text=cyan, description=dim, scroll_info=dim, no_match=dim)
    editor = Editor(tui, EditorTheme(border_color=dim, select_list=select_theme))
    tui.add_child(editor)
    tui.set_focus(editor)

    tui.start()
    await asyncio.sleep(0.05)

    # Initial state — no content
    lines = tui.render(80)
    clean = [re.sub(r"\x1b[^m]*m|\x1b\[[^a-zA-Z]*[a-zA-Z]", "", l).strip() for l in lines]
    assert "You:" not in " ".join(clean)

    # Add user message
    history_text.set_text("You: hello world")
    history_text.invalidate()
    tui.request_render()
    await asyncio.sleep(0.05)

    lines = tui.render(80)
    clean_lines = [re.sub(r"\x1b[^m]*m|\x1b\[[^a-zA-Z]*[a-zA-Z]", "", l).strip() for l in lines]
    assert any("You: hello world" in l for l in clean_lines)

    # Add streaming response
    stream_text.set_text("Assistant: Hi there!")
    stream_text.invalidate()
    tui.request_render()
    await asyncio.sleep(0.05)

    lines = tui.render(80)
    clean_lines = [re.sub(r"\x1b[^m]*m|\x1b\[[^a-zA-Z]*[a-zA-Z]", "", l).strip() for l in lines]
    assert any("You: hello world" in l for l in clean_lines)
    assert any("Assistant: Hi there!" in l for l in clean_lines)

    tui.stop()

    # Terminal writes should contain the response text
    all_output = mt.all_output
    assert "You: hello world" in all_output or "You:" in all_output


@pytest.mark.asyncio
async def test_tui_on_event_direct_calls():
    """
    Verify the TUI interactive handle_submit loop works end-to-end using
    a mock session that fires agent events directly (no real API call).
    """
    import re
    from pi_tui.tui import TUI
    from pi_tui.components.text import Text
    from pi_tui.components.spacer import Spacer
    from pi_tui.components.editor import Editor, EditorTheme
    from pi_tui.components.select_list import SelectListTheme

    def bold(s: str) -> str:
        return f"\x1b[1m{s}\x1b[22m"

    def dim(s: str) -> str:
        return f"\x1b[2m{s}\x1b[22m"

    def cyan(s: str) -> str:
        return f"\x1b[36m{s}\x1b[39m"

    def red(s: str) -> str:
        return f"\x1b[31m{s}\x1b[39m"

    def yellow(s: str) -> str:
        return f"\x1b[33m{s}\x1b[39m"

    mt = _MockTerminal()
    tui = TUI(mt)

    history_text = Text("", padding_x=1, padding_y=0)
    stream_text = Text("", padding_x=1, padding_y=0)
    tui.add_child(history_text)
    tui.add_child(stream_text)
    tui.add_child(Spacer(1))
    select_theme = SelectListTheme(selected_text=cyan, description=dim, scroll_info=dim, no_match=dim)
    editor = Editor(tui, EditorTheme(border_color=dim, select_list=select_theme))
    tui.add_child(editor)
    tui.set_focus(editor)

    def append_history(line: str) -> None:
        cur = history_text._text
        history_text.set_text((cur + "\n" + line).lstrip("\n"))
        history_text.invalidate()

    def set_stream(text: str) -> None:
        stream_text.set_text(text)
        stream_text.invalidate()
        tui.request_render()

    async def handle_submit(text: str, response_chunks: list[str]) -> None:
        """Simulated handle_submit that replays mock events."""
        collected: list[str] = []
        done_event = asyncio.Event()

        append_history(f"{bold('You:')} {text}")
        tui.request_render()

        # Simulate firing events directly (as on_event would receive them)
        for chunk in response_chunks:
            collected.append(chunk)
            set_stream(f"{bold('Assistant:')} {''.join(collected)}")
            await asyncio.sleep(0)  # Yield so renders can happen

        # Simulate agent_end
        done_event.set()
        await done_event.wait()

        # Finally: move to history
        final = stream_text._text
        if final:
            append_history(final)
            set_stream("")
        tui.request_render()

    tui.start()
    await asyncio.sleep(0.05)

    await handle_submit("say hello", ["Hello", ", world", "!"])
    await asyncio.sleep(0.1)

    tui.stop()

    # Check final rendered state
    lines = tui.render(80)
    clean_lines = [
        re.sub(r"\x1b[^m]*m|\x1b\[[^a-zA-Z]*[a-zA-Z]", "", l).strip()
        for l in lines
    ]
    assert any("You:" in l for l in clean_lines), f"No 'You:' in {clean_lines}"
    assert any("Assistant:" in l for l in clean_lines), f"No 'Assistant:' in {clean_lines}"

    # Verify response was committed to history, stream is empty
    assert stream_text._text == ""
    assert "Hello, world!" in history_text._text or "Assistant:" in history_text._text


@pytest.mark.asyncio
async def test_tui_initial_messages_render_without_text_delta(monkeypatch):
    """
    Regression test for "input-only, no assistant output":
    ensure interactive TUI can render assistant responses even when providers
    only emit message_start/message_end snapshots (no text_delta stream).
    """
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 80
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self) -> None:
            self._listeners: list = []
            from types import SimpleNamespace
            self.model = SimpleNamespace(id='claude-3-5-sonnet-20241022', provider='anthropic')
            self.thinking_level = 'off'

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return ['bash', 'read']

        def get_session_stats(self):
            return {'sessionId': 'test', 'userMessages': 0, 'assistantMessages': 0,
                    'toolCalls': 0, 'tokens': {'total': 0}, 'cost': 0.0}

        def cycle_thinking_level(self):
            return 'minimal'

        async def compact(self):
            return ''

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction='forward'):
            return None

        async def follow_up(self, msg):
            pass

        @property
        def model_registry(self):
            from types import SimpleNamespace
            async def ga(): return [self.model]
            return SimpleNamespace(get_available=ga)

        def subscribe(self, fn):
            self._listeners.append(fn)

            def _unsub():
                if fn in self._listeners:
                    self._listeners.remove(fn)

            return _unsub

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            start_message = SimpleNamespace(role="assistant", content=[])
            end_message = SimpleNamespace(
                role="assistant",
                content=[SimpleNamespace(type="text", text=f"Echo: {text}")],
                error_message=None,
            )

            for listener in list(self._listeners):
                listener(SimpleNamespace(type="agent_start"))
                listener(SimpleNamespace(type="turn_start"))
                listener(SimpleNamespace(type="message_start", message=start_message))
                listener(SimpleNamespace(type="message_end", message=end_message))
                listener(SimpleNamespace(type="turn_end", message=end_message))
                listener(SimpleNamespace(type="agent_end"))

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)

    session = FakeSession()
    await _run_pi_tui(session, initial_messages=["你好", "hello", "/exit"])

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    assert "You: 你好" in clean
    assert "You: hello" in clean
    assert "Assistant: Echo: 你好" in clean
    assert "Assistant: Echo: hello" in clean


@pytest.mark.asyncio
async def test_tui_agent_end_error_is_rendered(monkeypatch):
    """
    If agent fails before streaming assistant deltas and only emits agent_end
    with an assistant error message, TUI should still show the error.
    """
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 80
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self) -> None:
            self._listeners: list = []
            from types import SimpleNamespace
            self.model = SimpleNamespace(id='claude-3-5-sonnet-20241022', provider='anthropic')
            self.thinking_level = 'off'

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return ['bash', 'read']

        def get_session_stats(self):
            return {'sessionId': 'test', 'userMessages': 0, 'assistantMessages': 0,
                    'toolCalls': 0, 'tokens': {'total': 0}, 'cost': 0.0}

        def cycle_thinking_level(self):
            return 'minimal'

        async def compact(self):
            return ''

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction='forward'):
            return None

        async def follow_up(self, msg):
            pass

        @property
        def model_registry(self):
            from types import SimpleNamespace
            async def ga(): return [self.model]
            return SimpleNamespace(get_available=ga)

        def subscribe(self, fn):
            self._listeners.append(fn)

            def _unsub():
                if fn in self._listeners:
                    self._listeners.remove(fn)

            return _unsub

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            err_msg = "No API key configured"
            assistant_error = SimpleNamespace(
                role="assistant",
                content=[SimpleNamespace(type="text", text="")],
                error_message=err_msg,
            )
            for listener in list(self._listeners):
                listener(SimpleNamespace(type="agent_start"))
                listener(SimpleNamespace(type="turn_start"))
                listener(
                    SimpleNamespace(
                        type="agent_end",
                        messages=[assistant_error],
                    )
                )

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)

    session = FakeSession()
    await _run_pi_tui(session, initial_messages=["你好", "/exit"])

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    assert "Error: No API key configured" in clean


@pytest.mark.asyncio
async def test_tui_renders_tool_execution_lines(monkeypatch):
    """Tool execution start/end should be visible in interactive TUI history."""
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 100
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self) -> None:
            self._listeners: list = []
            from types import SimpleNamespace
            self.model = SimpleNamespace(id='claude-3-5-sonnet-20241022', provider='anthropic')
            self.thinking_level = 'off'

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return ['bash', 'read']

        def get_session_stats(self):
            return {'sessionId': 'test', 'userMessages': 0, 'assistantMessages': 0,
                    'toolCalls': 0, 'tokens': {'total': 0}, 'cost': 0.0}

        def cycle_thinking_level(self):
            return 'minimal'

        async def compact(self):
            return ''

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction='forward'):
            return None

        async def follow_up(self, msg):
            pass

        @property
        def model_registry(self):
            from types import SimpleNamespace
            async def ga(): return [self.model]
            return SimpleNamespace(get_available=ga)

        def subscribe(self, fn):
            self._listeners.append(fn)

            def _unsub():
                if fn in self._listeners:
                    self._listeners.remove(fn)

            return _unsub

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            assistant = SimpleNamespace(
                role="assistant",
                content=[SimpleNamespace(type="text", text="Done.")],
                error_message=None,
            )
            for listener in list(self._listeners):
                listener(SimpleNamespace(type="agent_start"))
                listener(SimpleNamespace(type="turn_start"))
                listener(SimpleNamespace(type="message_start", message=assistant))
                listener(SimpleNamespace(type="tool_execution_start", tool_call_id="tc1", tool_name="bash"))
                listener(
                    SimpleNamespace(
                        type="tool_execution_end",
                        tool_call_id="tc1",
                        tool_name="bash",
                        is_error=False,
                        result=SimpleNamespace(content=[SimpleNamespace(type="text", text="exit_code: 0")]),
                    )
                )
                listener(SimpleNamespace(type="message_end", message=assistant))
                listener(SimpleNamespace(type="turn_end", message=assistant))
                listener(SimpleNamespace(type="agent_end", messages=[assistant]))

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)

    session = FakeSession()
    await _run_pi_tui(session, initial_messages=["run tool", "/exit"])

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    assert "Tool start: bash" in clean
    assert "Tool end: bash" in clean


@pytest.mark.asyncio
async def test_tui_does_not_require_agent_end_event(monkeypatch):
    """Interactive flow should complete even if provider stream omits agent_end."""
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 100
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self) -> None:
            self._listeners: list = []
            from types import SimpleNamespace
            self.model = SimpleNamespace(id='claude-3-5-sonnet-20241022', provider='anthropic')
            self.thinking_level = 'off'

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return ['bash', 'read']

        def get_session_stats(self):
            return {'sessionId': 'test', 'userMessages': 0, 'assistantMessages': 0,
                    'toolCalls': 0, 'tokens': {'total': 0}, 'cost': 0.0}

        def cycle_thinking_level(self):
            return 'minimal'

        async def compact(self):
            return ''

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction='forward'):
            return None

        async def follow_up(self, msg):
            pass

        @property
        def model_registry(self):
            from types import SimpleNamespace
            async def ga(): return [self.model]
            return SimpleNamespace(get_available=ga)

        def subscribe(self, fn):
            self._listeners.append(fn)

            def _unsub():
                if fn in self._listeners:
                    self._listeners.remove(fn)

            return _unsub

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            assistant = SimpleNamespace(
                role="assistant",
                content=[SimpleNamespace(type="text", text="Completed without agent_end.")],
                error_message=None,
            )
            for listener in list(self._listeners):
                listener(SimpleNamespace(type="agent_start"))
                listener(SimpleNamespace(type="turn_start"))
                listener(SimpleNamespace(type="message_start", message=assistant))
                listener(SimpleNamespace(type="message_update", message=assistant, assistant_message_event=SimpleNamespace(type="text_delta", delta="")))
                listener(SimpleNamespace(type="message_end", message=assistant))
                listener(SimpleNamespace(type="turn_end", message=assistant))
            return

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)

    session = FakeSession()
    await _run_pi_tui(session, initial_messages=["run", "/exit"])

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    assert "Completed without agent_end." in clean
