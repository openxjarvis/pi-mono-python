"""
Tests for migrations.py.
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest


class TestMigrateAuthToAuthJson:
    def _setup_agent_dir(self, tmpdir: str) -> str:
        agent_dir = os.path.join(tmpdir, ".pi", "agent")
        os.makedirs(agent_dir, exist_ok=True)
        return agent_dir

    def test_no_migration_when_auth_exists(self, tmp_path):
        agent_dir = self._setup_agent_dir(str(tmp_path))
        auth_path = os.path.join(agent_dir, "auth.json")
        with open(auth_path, "w") as f:
            json.dump({"existing": {}}, f)

        with patch("pi_coding_agent.migrations.get_agent_dir", return_value=agent_dir):
            from pi_coding_agent.migrations import migrate_auth_to_auth_json
            result = migrate_auth_to_auth_json()
        assert result == []

    def test_migrates_oauth_json(self, tmp_path):
        agent_dir = self._setup_agent_dir(str(tmp_path))
        oauth_path = os.path.join(agent_dir, "oauth.json")
        with open(oauth_path, "w") as f:
            json.dump({"anthropic": {"access_token": "abc"}}, f)

        with patch("pi_coding_agent.migrations.get_agent_dir", return_value=agent_dir):
            from pi_coding_agent.migrations import migrate_auth_to_auth_json
            result = migrate_auth_to_auth_json()

        assert "anthropic" in result
        auth_path = os.path.join(agent_dir, "auth.json")
        assert os.path.exists(auth_path)
        with open(auth_path) as f:
            data = json.load(f)
        assert data["anthropic"]["type"] == "oauth"

    def test_migrates_settings_json_api_keys(self, tmp_path):
        agent_dir = self._setup_agent_dir(str(tmp_path))
        settings_path = os.path.join(agent_dir, "settings.json")
        with open(settings_path, "w") as f:
            json.dump({"apiKeys": {"openai": "sk-test123"}, "theme": "dark"}, f)

        with patch("pi_coding_agent.migrations.get_agent_dir", return_value=agent_dir):
            from pi_coding_agent.migrations import migrate_auth_to_auth_json
            result = migrate_auth_to_auth_json()

        assert "openai" in result
        with open(settings_path) as f:
            settings = json.load(f)
        assert "apiKeys" not in settings
        assert settings.get("theme") == "dark"

    def test_skips_on_corrupt_json(self, tmp_path):
        agent_dir = self._setup_agent_dir(str(tmp_path))
        oauth_path = os.path.join(agent_dir, "oauth.json")
        with open(oauth_path, "w") as f:
            f.write("not valid json {")

        with patch("pi_coding_agent.migrations.get_agent_dir", return_value=agent_dir):
            from pi_coding_agent.migrations import migrate_auth_to_auth_json
            result = migrate_auth_to_auth_json()

        assert result == []


class TestMigrateSessionsFromAgentRoot:
    def test_moves_jsonl_files(self, tmp_path):
        agent_dir = str(tmp_path)
        session_file = os.path.join(agent_dir, "my-session.jsonl")
        header = {"type": "session", "cwd": "/home/user/project"}
        with open(session_file, "w") as f:
            f.write(json.dumps(header) + "\n")
            f.write(json.dumps({"role": "user", "content": "hello"}) + "\n")

        with patch("pi_coding_agent.migrations.get_agent_dir", return_value=agent_dir):
            from pi_coding_agent.migrations import migrate_sessions_from_agent_root
            migrate_sessions_from_agent_root()

        # File should be moved to sessions/<safe-path>/
        sessions_dir = os.path.join(agent_dir, "sessions")
        assert os.path.exists(sessions_dir)
        moved = False
        for root, _, files in os.walk(sessions_dir):
            if "my-session.jsonl" in files:
                moved = True
                break
        assert moved, "Session file was not moved to sessions/ directory"

    def test_handles_empty_dir(self, tmp_path):
        agent_dir = str(tmp_path)
        with patch("pi_coding_agent.migrations.get_agent_dir", return_value=agent_dir):
            from pi_coding_agent.migrations import migrate_sessions_from_agent_root
            migrate_sessions_from_agent_root()

    def test_handles_corrupt_jsonl(self, tmp_path):
        agent_dir = str(tmp_path)
        session_file = os.path.join(agent_dir, "bad.jsonl")
        with open(session_file, "w") as f:
            f.write("not json\n")

        with patch("pi_coding_agent.migrations.get_agent_dir", return_value=agent_dir):
            from pi_coding_agent.migrations import migrate_sessions_from_agent_root
            migrate_sessions_from_agent_root()


class TestMigrateToolsToBin:
    def test_moves_fd_rg_to_bin(self, tmp_path):
        agent_dir = str(tmp_path)
        tools_dir = os.path.join(agent_dir, "tools")
        bin_dir = os.path.join(agent_dir, "bin")
        os.makedirs(tools_dir)

        fd_path = os.path.join(tools_dir, "fd")
        with open(fd_path, "w") as f:
            f.write("binary")

        with patch("pi_coding_agent.migrations.get_agent_dir", return_value=agent_dir):
            with patch("pi_coding_agent.migrations.get_bin_dir", return_value=bin_dir):
                from pi_coding_agent.migrations import _migrate_tools_to_bin
                _migrate_tools_to_bin()

        assert os.path.exists(os.path.join(bin_dir, "fd"))
        assert not os.path.exists(fd_path)


class TestMigrateCommandsToPrompts:
    def test_renames_commands_to_prompts(self, tmp_path):
        base_dir = str(tmp_path)
        commands_dir = os.path.join(base_dir, "commands")
        os.makedirs(commands_dir)
        with open(os.path.join(commands_dir, "test.md"), "w") as f:
            f.write("# test")

        from pi_coding_agent.migrations import _migrate_commands_to_prompts
        result = _migrate_commands_to_prompts(base_dir, "Global")

        assert result is True
        assert os.path.exists(os.path.join(base_dir, "prompts"))
        assert not os.path.exists(commands_dir)

    def test_skips_when_prompts_exists(self, tmp_path):
        base_dir = str(tmp_path)
        commands_dir = os.path.join(base_dir, "commands")
        prompts_dir = os.path.join(base_dir, "prompts")
        os.makedirs(commands_dir)
        os.makedirs(prompts_dir)

        from pi_coding_agent.migrations import _migrate_commands_to_prompts
        result = _migrate_commands_to_prompts(base_dir, "Global")

        assert result is False
        assert os.path.exists(commands_dir)


class TestRunMigrations:
    def test_run_migrations_returns_dict(self, tmp_path):
        agent_dir = os.path.join(str(tmp_path), ".pi", "agent")
        os.makedirs(agent_dir)

        with patch("pi_coding_agent.migrations.get_agent_dir", return_value=agent_dir):
            with patch("pi_coding_agent.migrations.get_bin_dir", return_value=os.path.join(agent_dir, "bin")):
                from pi_coding_agent.migrations import run_migrations
                result = run_migrations(str(tmp_path))

        assert "migratedAuthProviders" in result
        assert "deprecationWarnings" in result
        assert isinstance(result["migratedAuthProviders"], list)
        assert isinstance(result["deprecationWarnings"], list)
