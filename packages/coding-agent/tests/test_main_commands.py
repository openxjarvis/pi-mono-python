from __future__ import annotations

import pytest


def test_parse_package_command_install_local() -> None:
    from pi_coding_agent.main import _parse_package_command

    parsed = _parse_package_command(["install", "npm:@foo/bar", "--local"])
    assert parsed is not None
    assert parsed["command"] == "install"
    assert parsed["source"] == "npm:@foo/bar"
    assert parsed["local"] is True


def test_parse_package_command_invalid() -> None:
    from pi_coding_agent.main import _parse_package_command

    assert _parse_package_command(["chat", "hello"]) is None


@pytest.mark.asyncio
async def test_handle_package_command_list(monkeypatch, capsys) -> None:
    from pi_coding_agent import main as main_mod

    class _Settings:
        def drain_errors(self):
            return []

        def get_global_settings(self):
            return {"packages": ["npm:@foo/bar"]}

        def get_project_settings(self):
            return {"packages": ["./local-ext"]}

    class _PkgMgr:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def set_progress_callback(self, _cb):
            return None

        def get_installed_path(self, source, scope):
            return f"/tmp/{scope}/{source.replace('/', '_')}"

    monkeypatch.setattr(main_mod, "SettingsManager", type("S", (), {"create": staticmethod(lambda *_: _Settings())}))
    monkeypatch.setattr(main_mod, "DefaultPackageManager", _PkgMgr)
    monkeypatch.setattr(main_mod, "get_agent_dir", lambda: "/tmp/agent")

    handled, code = await main_mod._handle_package_command(["list"])
    out = capsys.readouterr().out

    assert handled is True
    assert code == 0
    assert "User packages:" in out
    assert "Project packages:" in out
    assert "npm:@foo/bar" in out
    assert "./local-ext" in out
