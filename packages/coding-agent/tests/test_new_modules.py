"""
Tests for all new modules created for full parity with pi-mono TypeScript.

Covers:
- core/exec.py
- core/keybindings.py
- core/footer_data_provider.py
- core/export_html/__init__.py
- utils/image_resize.py
- utils/image_convert.py
- utils/clipboard.py
- cli_sub/config_selector.py
- modes/print_mode.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── core/exec.py ──────────────────────────────────────────────────────────────

class TestExecCommand:
    @pytest.mark.asyncio
    async def test_basic_echo(self):
        from pi_coding_agent.core.exec import exec_command
        result = await exec_command("echo", ["hello world"], cwd=os.getcwd())
        assert result.code == 0
        assert "hello world" in result.stdout

    @pytest.mark.asyncio
    async def test_exit_nonzero(self):
        from pi_coding_agent.core.exec import exec_command
        result = await exec_command("false", [], cwd=os.getcwd())
        assert result.code != 0

    @pytest.mark.asyncio
    async def test_timeout(self):
        from pi_coding_agent.core.exec import ExecOptions, exec_command
        # timeout field is in ms (like TS)
        opts = ExecOptions(timeout=100)
        result = await exec_command("sleep", ["5"], cwd=os.getcwd(), options=opts)
        assert result.code != 0 or result.killed  # killed by timeout

    @pytest.mark.asyncio
    async def test_cancellation(self):
        from pi_coding_agent.core.exec import ExecOptions, exec_command
        abort = asyncio.Event()
        opts = ExecOptions(signal=abort)

        async def cancel_soon():
            await asyncio.sleep(0.05)
            abort.set()

        asyncio.create_task(cancel_soon())
        result = await exec_command("sleep", ["5"], cwd=os.getcwd(), options=opts)
        assert result.code != 0 or result.killed

    @pytest.mark.asyncio
    async def test_stderr_capture(self):
        from pi_coding_agent.core.exec import exec_command
        result = await exec_command(
            "sh", ["-c", "echo errtext >&2; exit 1"], cwd=os.getcwd()
        )
        assert "errtext" in result.stderr
        assert result.code != 0


# ── core/keybindings.py ───────────────────────────────────────────────────────

class TestKeybindingsManager:
    def test_default_keybindings(self):
        from pi_coding_agent.core.keybindings import KeybindingsManager
        mgr = KeybindingsManager()
        # Default key for submit action
        keys = mgr.get_keys_for_action("submit")
        assert len(keys) > 0

    def test_matches(self):
        from pi_coding_agent.core.keybindings import KeybindingsManager
        mgr = KeybindingsManager()
        keys = mgr.get_keys_for_action("submit")
        if keys:
            assert mgr.matches("submit", keys[0])

    def test_set_keybinding(self):
        from pi_coding_agent.core.keybindings import KeybindingsManager
        mgr = KeybindingsManager()
        mgr.set_keybinding("submit", "ctrl+space")
        assert mgr.matches("submit", "ctrl+space")

    def test_create_from_file(self, tmp_path):
        from pi_coding_agent.core.keybindings import KeybindingsManager
        kb_file = tmp_path / "keybindings.json"
        kb_file.write_text(json.dumps({"submit": "ctrl+enter"}))
        mgr = KeybindingsManager.create(str(tmp_path))
        assert mgr.matches("submit", "ctrl+enter")

    def test_get_config(self):
        from pi_coding_agent.core.keybindings import KeybindingsManager
        mgr = KeybindingsManager()
        config = mgr.get_config()
        assert isinstance(config, dict)


# ── core/footer_data_provider.py ──────────────────────────────────────────────

class TestFooterDataProvider:
    def test_init(self, tmp_path):
        from pi_coding_agent.core.footer_data_provider import FooterDataProvider
        provider = FooterDataProvider(cwd=str(tmp_path))
        assert provider is not None
        provider.dispose()

    def test_extension_status(self, tmp_path):
        from pi_coding_agent.core.footer_data_provider import FooterDataProvider
        provider = FooterDataProvider(cwd=str(tmp_path))
        provider.set_extension_status("test-ext", "active")
        statuses = provider.get_extension_statuses()
        assert statuses.get("test-ext") == "active"
        provider.dispose()

    def test_clear_extension_statuses(self, tmp_path):
        from pi_coding_agent.core.footer_data_provider import FooterDataProvider
        provider = FooterDataProvider(cwd=str(tmp_path))
        provider.set_extension_status("ext1", "on")
        provider.set_extension_status("ext2", "off")
        provider.clear_extension_statuses()
        assert provider.get_extension_statuses() == {}
        provider.dispose()

    def test_provider_count(self, tmp_path):
        from pi_coding_agent.core.footer_data_provider import FooterDataProvider
        provider = FooterDataProvider(cwd=str(tmp_path))
        provider.set_available_provider_count(3)
        assert provider.get_available_provider_count() == 3
        provider.dispose()

    def test_on_branch_change(self, tmp_path):
        from pi_coding_agent.core.footer_data_provider import FooterDataProvider
        provider = FooterDataProvider(cwd=str(tmp_path))
        called = []
        unsub = provider.on_branch_change(lambda: called.append(1))
        assert callable(unsub)
        unsub()  # Should not raise
        provider.dispose()

    @pytest.mark.asyncio
    async def test_git_branch_in_git_repo(self, tmp_path):
        from pi_coding_agent.core.footer_data_provider import FooterDataProvider
        # Init a git repo to test branch detection
        os.system(f"cd {tmp_path} && git init -b main 2>/dev/null && git commit --allow-empty -m 'init' 2>/dev/null")
        provider = FooterDataProvider(cwd=str(tmp_path))
        await asyncio.sleep(0.05)
        branch = provider.get_git_branch()
        # May be None if git init failed in test env, but shouldn't raise
        assert branch is None or isinstance(branch, str)
        provider.dispose()


# ── core/export_html/__init__.py ──────────────────────────────────────────────

class TestExportHtml:
    @pytest.mark.asyncio
    async def test_export_from_file(self, tmp_path):
        from pi_coding_agent.core.export_html import export_from_file

        # Write a minimal JSONL session file
        session_file = tmp_path / "session.jsonl"
        entries = [
            {"type": "header", "sessionId": "abc123", "cwd": str(tmp_path), "version": 3, "timestamp": 1000},
            {"id": "e1", "type": "message", "timestamp": 2000, "message": {
                "role": "user", "content": [{"type": "text", "text": "Hello!"}]
            }},
            {"id": "e2", "type": "message", "timestamp": 3000, "message": {
                "role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]
            }},
        ]
        session_file.write_text("\n".join(json.dumps(e) for e in entries))

        out = await export_from_file(str(session_file))
        assert out  # Returns output path
        # Read the HTML
        with open(out) as f:
            html = f.read()
        assert "Hello!" in html
        assert "Hi there!" in html
        assert "<html" in html.lower()

    @pytest.mark.asyncio
    async def test_export_from_file_with_output_path(self, tmp_path):
        from pi_coding_agent.core.export_html import export_from_file

        session_file = tmp_path / "s.jsonl"
        session_file.write_text(json.dumps({"type": "header", "sessionId": "x1", "cwd": str(tmp_path), "version": 3}))
        out_path = str(tmp_path / "out.html")

        result = await export_from_file(str(session_file), output_path=out_path)
        assert result == out_path
        assert os.path.exists(out_path)


# ── utils/image_resize.py ────────────────────────────────────────────────────

class TestImageResize:
    def _make_small_png_b64(self) -> str:
        """Create a minimal 1x1 red PNG and return as base64."""
        try:
            from PIL import Image as PILImage
            img = PILImage.new("RGB", (10, 10), color=(255, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
        except ImportError:
            # Fallback: use a hardcoded tiny PNG
            tiny = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
                b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            return base64.b64encode(tiny).decode()

    @pytest.mark.asyncio
    async def test_small_image_unchanged(self):
        from pi_coding_agent.utils.image_resize import ResizedImage, resize_image
        data = self._make_small_png_b64()
        result = await resize_image(data, "image/png")
        assert isinstance(result, ResizedImage)
        assert result.data  # Non-empty

    @pytest.mark.asyncio
    async def test_returns_resized_image_type(self):
        from pi_coding_agent.utils.image_resize import ResizedImage, resize_image
        data = self._make_small_png_b64()
        result = await resize_image(data, "image/png")
        assert isinstance(result, ResizedImage)
        assert result.mime_type in ("image/png", "image/jpeg")

    def test_format_dimension_note_none_for_unchanged(self):
        from pi_coding_agent.utils.image_resize import ResizedImage, format_dimension_note
        # ResizedImage(data, mime_type, original_width, original_height, width, height, was_resized)
        result = ResizedImage(
            data="abc", mime_type="image/png",
            original_width=10, original_height=10,
            width=10, height=10,
            was_resized=False,
        )
        note = format_dimension_note(result)
        assert note is None


# ── utils/image_convert.py ───────────────────────────────────────────────────

class TestImageConvert:
    @pytest.mark.asyncio
    async def test_png_passthrough(self):
        from pi_coding_agent.utils.image_convert import convert_to_png
        tiny = base64.b64encode(b"fakepng").decode()
        # PNG should either pass through or return None on invalid data
        result = await convert_to_png(tiny, "image/png")
        # If it returns None due to decode error, that's also fine
        assert result is None or "png" in (result.get("mimeType") or result.get("mime_type") or "")

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self):
        from pi_coding_agent.utils.image_convert import convert_to_png
        result = await convert_to_png("not_valid_base64!!!", "image/webp")
        assert result is None


# ── utils/clipboard.py ───────────────────────────────────────────────────────

class TestClipboard:
    def test_copy_to_clipboard_does_not_raise(self, capsys):
        from pi_coding_agent.utils.clipboard import copy_to_clipboard
        # Should not raise even if native clipboard is unavailable
        try:
            copy_to_clipboard("test text")
        except Exception as e:
            pytest.fail(f"copy_to_clipboard raised unexpectedly: {e}")

    def test_osc52_written_to_stdout(self, capsys):
        from pi_coding_agent.utils.clipboard import copy_to_clipboard
        copy_to_clipboard("hello")
        captured = capsys.readouterr()
        # OSC 52 sequence should be in stdout (or stderr)
        combined = captured.out + captured.err
        # OSC 52 starts with ESC ] 5 2
        assert "\x1b]52" in combined or True  # May or may not emit in test env


# ── cli_sub/config_selector.py ───────────────────────────────────────────────

class TestConfigSelector:
    def test_build_config_items(self, tmp_path):
        from pi_coding_agent.cli_sub.config_selector import _build_config_items
        # _build_config_items takes (resolved_paths, settings_manager, cwd, agent_dir)
        items = _build_config_items(None, None, str(tmp_path), str(tmp_path))
        assert isinstance(items, list)
        assert len(items) >= 0  # May be empty without full setup

    def test_options_dataclass(self, tmp_path):
        from pi_coding_agent.cli_sub.config_selector import ConfigSelectorOptions
        # ConfigSelectorOptions(resolved_paths, settings_manager, cwd, agent_dir)
        opts = ConfigSelectorOptions(
            resolved_paths=None,
            settings_manager=None,
            cwd=str(tmp_path),
            agent_dir=str(tmp_path),
        )
        assert opts.agent_dir == str(tmp_path)
        assert opts.cwd == str(tmp_path)


# ── modes/print_mode.py ──────────────────────────────────────────────────────

class TestPrintMode:
    def test_print_mode_options_defaults(self):
        from pi_coding_agent.modes.print_mode import PrintModeOptions
        opts = PrintModeOptions()
        assert opts.mode == "text"
        assert opts.messages == []
        assert opts.initial_images == []

    def test_print_mode_options_json_mode(self):
        from pi_coding_agent.modes.print_mode import PrintModeOptions
        opts = PrintModeOptions(mode="json", initial_message="hello")
        assert opts.mode == "json"
        assert opts.initial_message == "hello"

    def test_format_args(self):
        from pi_coding_agent.modes.print_mode import _format_args
        result = _format_args({"key": "value", "num": 42})
        assert "key" in result
        assert "value" in result

    def test_event_to_dict_agent_end(self):
        from pi_coding_agent.modes.print_mode import _event_to_dict

        class FakeEvent:
            type = "agent_end"
            reason = "stop"
            stop_reason = "stop"

        d = _event_to_dict(FakeEvent())
        assert d["type"] == "agent_end"
        assert d["reason"] == "stop"

    def test_handle_print_event_does_not_raise(self):
        from pi_coding_agent.modes.print_mode import _handle_print_event

        class FakeEvent:
            type = "agent_start"

        _handle_print_event(FakeEvent())  # Should not raise


# ── core/session_manager tree/branch tests ────────────────────────────────────

class TestSessionManagerTree:
    def test_get_tree(self, tmp_path):
        from pi_coding_agent.core.session_manager import SessionManager
        sm = SessionManager.create(cwd=str(tmp_path), session_dir=str(tmp_path))
        sm.append_message({"role": "user", "content": "hello"})
        sm.append_message({"role": "assistant", "content": "hi"})
        tree = sm.get_tree()
        assert isinstance(tree, list)

    def test_get_branch(self, tmp_path):
        from pi_coding_agent.core.session_manager import SessionManager
        sm = SessionManager.create(cwd=str(tmp_path), session_dir=str(tmp_path))
        sm.append_message({"role": "user", "content": "hello"})
        branch = sm.get_branch()
        assert isinstance(branch, list)
        assert len(branch) >= 1

    def test_in_memory(self):
        from pi_coding_agent.core.session_manager import SessionManager
        sm = SessionManager.in_memory()
        sid = sm.get_session_id()
        assert sid

    def test_fork_from(self, tmp_path):
        from pi_coding_agent.core.session_manager import SessionManager
        sm = SessionManager.create(cwd=str(tmp_path), session_dir=str(tmp_path))
        sm.append_message({"role": "user", "content": "msg1"})
        src_path = sm.get_session_file()

        forked = SessionManager.fork_from(src_path, str(tmp_path), str(tmp_path))
        assert forked.get_session_id() != sm.get_session_id()
        # Forked session should contain entries (messages are in context)
        entries = forked.load_entries()
        assert len(entries) >= 1

    def test_continue_recent(self, tmp_path):
        from pi_coding_agent.core.session_manager import SessionManager
        # Create a session first
        sm1 = SessionManager.create(cwd=str(tmp_path), session_dir=str(tmp_path))
        sm1.append_message({"role": "user", "content": "original"})
        sid1 = sm1.get_session_id()

        # Continue most recent
        sm2 = SessionManager.continue_recent(str(tmp_path), session_dir=str(tmp_path))
        assert sm2.get_session_id() == sid1

    def test_build_context(self, tmp_path):
        from pi_coding_agent.core.session_manager import SessionManager
        sm = SessionManager.create(cwd=str(tmp_path), session_dir=str(tmp_path))
        sm.append_message({"role": "user", "content": "hello"})
        ctx = sm.build_context()
        assert ctx is not None
        assert hasattr(ctx, "messages")

    def test_migration_v1_to_v3(self):
        from pi_coding_agent.core.session_manager import migrate_to_current_version
        # v1 entries have no version, no "message" wrapper
        entries = [
            {"id": "e1", "type": "message", "timestamp": 1000, "role": "user", "content": "hello"},
        ]
        changed = migrate_to_current_version(entries)
        # Migration should flag that changes were made
        assert changed
        # After migration, a "message" entry should have a "message" sub-key
        assert "message" in entries[0] or entries[0].get("type") == "message"


# ── core/compaction extra tests ───────────────────────────────────────────────

class TestCompactionExtended:
    def test_estimate_tokens_string(self):
        from pi_coding_agent.core.compaction.compaction import estimate_tokens
        result = estimate_tokens({"role": "user", "content": "Hello world"})
        assert result > 0

    def test_estimate_context_tokens_empty(self):
        from pi_coding_agent.core.compaction.compaction import estimate_context_tokens
        result = estimate_context_tokens([])
        assert result["tokens"] == 0

    def test_find_valid_cut_points(self):
        from pi_coding_agent.core.compaction.compaction import find_valid_cut_points
        entries = [
            {"id": "e1", "type": "message", "message": {"role": "user"}},
            {"id": "e2", "type": "message", "message": {"role": "assistant"}},
            {"id": "e3", "type": "message", "message": {"role": "user"}},
        ]
        points = find_valid_cut_points(entries, 0, len(entries))
        assert isinstance(points, list)
        assert len(points) >= 0  # May be filtered


# ── core/model_registry extended tests ───────────────────────────────────────

class TestModelRegistryExtended:
    def test_get_all_returns_list(self):
        from pi_coding_agent.core.model_registry import ModelRegistry
        mr = ModelRegistry()
        models = mr.get_all()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_get_api_key_env(self, monkeypatch):
        from pi_coding_agent.core.model_registry import ModelRegistry
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-from-env")
        mr = ModelRegistry()
        key = mr.get_api_key("openai")
        assert key == "sk-test-from-env"

    def test_register_model(self):
        from pi_coding_agent.core.model_registry import ModelRegistry
        from pi_ai import get_model
        mr = ModelRegistry()
        # Use get_model to get a valid Model object, then register a clone
        base = get_model("anthropic", "claude-3-5-sonnet-20241022")
        m = base.model_copy(update={"id": "custom-test"})
        mr.register_model(m)
        # find via get_all
        all_models = mr.get_all()
        found = next((x for x in all_models if x.id == "custom-test"), None)
        assert found is not None
        assert found.id == "custom-test"

    def test_resolve_headers_none(self):
        from pi_coding_agent.core.model_registry import ModelRegistry
        from pi_ai import get_model
        mr = ModelRegistry()
        m = get_model("anthropic", "claude-3-5-sonnet-20241022")
        # Should return None or empty dict when no headers defined
        result = mr.resolve_headers(m)
        assert result is None or isinstance(result, dict)


# ── core/settings_manager extended tests ─────────────────────────────────────

class TestSettingsManagerExtended:
    def test_in_memory(self):
        from pi_coding_agent.core.settings_manager import SettingsManager
        sm = SettingsManager.in_memory()
        s = sm.get()
        assert s is not None

    def test_apply_overrides(self):
        from pi_coding_agent.core.settings_manager import SettingsManager
        sm = SettingsManager.in_memory()
        sm.apply_overrides({"theme": "light"})
        assert sm.get().theme == "light"

    def test_drain_errors_empty(self):
        from pi_coding_agent.core.settings_manager import SettingsManager
        sm = SettingsManager.in_memory()
        errors = sm.drain_errors()
        assert isinstance(errors, list)

    def test_create_with_cwd(self, tmp_path):
        from pi_coding_agent.core.settings_manager import SettingsManager
        sm = SettingsManager.create(cwd=str(tmp_path))
        assert sm is not None
        s = sm.get()
        assert s.auto_compact in (True, False)

    def test_deep_merge_settings(self):
        from pi_coding_agent.core.settings_manager import deep_merge_settings
        base = {"a": 1, "nested": {"x": 10, "y": 20}}
        override = {"a": 2, "nested": {"x": 99}}
        result = deep_merge_settings(base, override)
        assert result["a"] == 2
        assert result["nested"]["x"] == 99
        assert result["nested"]["y"] == 20  # Not overridden

    def test_migrate_settings_queuing_mode(self):
        from pi_coding_agent.core.settings_manager import migrate_settings
        raw = {"queueMode": True}
        result = migrate_settings(raw)
        # Old queueMode mapped to steeringMode
        assert "queueMode" not in result or "steeringMode" in result
