"""
Tests for SessionManager — mirrors packages/coding-agent/test/ session tests.
Updated to use the new per-session SessionManager API.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from pi_coding_agent.core.session_manager import SessionEntry, SessionManager
from pi_coding_agent.core.settings_manager import Settings, SettingsManager
from pi_coding_agent.core.auth_storage import AuthStorage


@pytest.fixture
def session_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def session_manager(session_dir):
    """Create a fresh per-session SessionManager using the new factory."""
    return SessionManager.create(cwd=session_dir, session_dir=session_dir)


# ── SessionManager tests ──────────────────────────────────────────────────────

def test_create_session(session_manager):
    """Session ID is an 8-char hex string (matching TypeScript generate_id)."""
    sid = session_manager.get_session_id()
    assert sid
    assert len(sid) == 8  # 8-char hex (TypeScript parity)


def test_create_session_with_label(session_dir):
    sm = SessionManager.create(cwd=session_dir, session_dir=session_dir)
    sm.append_session_info(name="My Session")
    entries = sm.load_entries()
    assert any(e.type == "session_info" for e in entries)


def test_append_and_load_message(session_manager):
    msg = {"role": "user", "content": "Hello", "timestamp": 1234567890}
    entry_id = session_manager.append_message(msg)
    assert entry_id

    messages = session_manager.get_messages()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"


def test_multiple_messages(session_manager):
    messages_to_add = [
        {"role": "user", "content": f"Message {i}", "timestamp": i}
        for i in range(5)
    ]
    for msg in messages_to_add:
        session_manager.append_message(msg)

    loaded = session_manager.get_messages()
    assert len(loaded) == 5
    for i, msg in enumerate(loaded):
        assert msg["content"] == f"Message {i}"


def test_model_change(session_manager):
    session_manager.append_model_change("claude-3-5-sonnet-20241022", "anthropic")

    entries = session_manager.load_entries()
    model_entries = [e for e in entries if e.type == "model_change"]
    assert len(model_entries) == 1
    # Data uses camelCase (TypeScript parity)
    assert model_entries[0].data.get("modelId") == "claude-3-5-sonnet-20241022"


def test_thinking_level_change(session_manager):
    session_manager.append_thinking_level_change("high")

    entries = session_manager.load_entries()
    level_entries = [e for e in entries if e.type == "thinking_level_change"]
    assert len(level_entries) == 1
    # Data uses camelCase (TypeScript parity)
    assert level_entries[0].data.get("thinkingLevel") == "high"


def test_compaction(session_manager):
    # append_compaction(summary, first_kept_entry_id, tokens_before)
    session_manager.append_compaction("Summary text", "id1")

    entries = session_manager.load_entries()
    compact_entries = [e for e in entries if e.type == "compaction"]
    assert len(compact_entries) == 1
    assert compact_entries[0].data["summary"] == "Summary text"


def test_list_sessions(session_dir):
    """list_sessions() should return all sessions in the sessions directory."""
    managers = [
        SessionManager.create(cwd=session_dir, session_dir=session_dir)
        for _ in range(3)
    ]
    # Any of the managers shares the same sessions_dir
    sessions = managers[0].list_sessions()
    assert len(sessions) == 3


def test_delete_session(session_dir):
    sm = SessionManager.create(cwd=session_dir, session_dir=session_dir)
    sid = sm.get_session_id()
    sm.delete_session()
    # Create a new manager to list sessions
    sm2 = SessionManager.create(cwd=session_dir, session_dir=session_dir)
    sessions = sm2.list_sessions()
    assert all(s.session_id != sid for s in sessions)


def test_set_label(session_manager):
    session_manager.set_label(session_manager.get_session_id(), "My Label")

    entries = session_manager.load_entries()
    # The entry type is "label" (TypeScript parity)
    label_entries = [e for e in entries if e.type == "label"]
    assert len(label_entries) == 1
    assert label_entries[0].data["label"] == "My Label"


# ── SettingsManager tests ─────────────────────────────────────────────────────

def test_settings_defaults():
    settings = Settings()
    assert settings.thinking_level == "off"
    assert settings.auto_compact is True


def test_settings_from_dict():
    data = {"thinking_level": "high", "auto_compact": False, "theme": "light"}
    settings = Settings.from_dict(data)
    assert settings.thinking_level == "high"
    assert settings.auto_compact is False
    assert settings.theme == "light"


def test_settings_merge():
    base = Settings(thinking_level="off", theme="dark")
    override = Settings(thinking_level="high")
    merged = base.merge(override)
    assert merged.thinking_level == "high"
    assert merged.theme == "dark"  # Not overridden


def test_settings_manager_load_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = tmpdir
        manager = SettingsManager(project_root=project_root)

        # Use the new API: save individual keys
        manager.save_project("thinking_level", "medium")
        manager.save_project("theme", "light")

        manager2 = SettingsManager(project_root=project_root)
        loaded = manager2.get()
        assert loaded.thinking_level == "medium"
        assert loaded.theme == "light"


# ── AuthStorage tests ──────────────────────────────────────────────────────────

def test_auth_storage_api_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        auth = AuthStorage()
        auth.AUTH_FILE = os.path.join(tmpdir, "auth.json")
        auth.AUTH_DIR = tmpdir
        auth._loaded = False
        auth._data = {}

        auth.set_api_key("anthropic", "sk-test-key-123")
        key = auth.get_api_key("anthropic")
        assert key == "sk-test-key-123"


def test_auth_storage_oauth_token():
    with tempfile.TemporaryDirectory() as tmpdir:
        auth = AuthStorage()
        auth.AUTH_FILE = os.path.join(tmpdir, "auth.json")
        auth.AUTH_DIR = tmpdir
        auth._loaded = False
        auth._data = {}

        token = {"access_token": "tok_123", "expires_at": 9999999}
        auth.set_oauth_token("github-copilot", token)

        loaded = auth.get_oauth_token("github-copilot")
        assert loaded == token


def test_auth_storage_delete_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        auth = AuthStorage()
        auth.AUTH_FILE = os.path.join(tmpdir, "auth.json")
        auth.AUTH_DIR = tmpdir
        auth._loaded = False
        auth._data = {}

        auth.set_api_key("openai", "sk-test")
        auth.delete_api_key("openai")
        assert auth.get_api_key("openai") is None


def test_auth_storage_env_fallback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-xyz")
    with tempfile.TemporaryDirectory() as tmpdir:
        auth = AuthStorage()
        auth.AUTH_FILE = os.path.join(tmpdir, "auth.json")
        auth.AUTH_DIR = tmpdir
        auth._loaded = True
        auth._data = {}

        resolved = auth.resolve_api_key("anthropic")
        assert resolved == "env-key-xyz"
