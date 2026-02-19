"""
Configuration paths and package detection.

Mirrors packages/coding-agent/src/config.ts
"""
from __future__ import annotations

import os
from pathlib import Path


# App metadata (mirrors piConfig in package.json)
APP_NAME: str = "pi"
CONFIG_DIR_NAME: str = ".pi"
VERSION: str = "0.0.1"

ENV_AGENT_DIR: str = f"{APP_NAME.upper()}_CODING_AGENT_DIR"


# ============================================================================
# User Config Paths (~/.pi/agent/*)
# ============================================================================


def get_agent_dir() -> str:
    """Get the agent config directory (e.g., ~/.pi/agent/)."""
    env_dir = os.environ.get(ENV_AGENT_DIR)
    if env_dir:
        home = os.path.expanduser("~")
        if env_dir == "~":
            return home
        if env_dir.startswith("~/"):
            return home + env_dir[1:]
        return env_dir
    return os.path.join(os.path.expanduser("~"), CONFIG_DIR_NAME, "agent")


def get_prompts_dir() -> str:
    """Get path to prompt templates directory."""
    return os.path.join(get_agent_dir(), "prompts")


def get_bin_dir() -> str:
    """Get path to managed binaries directory (fd, rg)."""
    return os.path.join(get_agent_dir(), "bin")


def get_sessions_dir() -> str:
    """Get path to sessions directory."""
    return os.path.join(get_agent_dir(), "sessions")


def get_models_path() -> str:
    """Get path to models.json."""
    return os.path.join(get_agent_dir(), "models.json")


def get_auth_path() -> str:
    """Get path to auth.json."""
    return os.path.join(get_agent_dir(), "auth.json")


def get_settings_path() -> str:
    """Get path to settings.json."""
    return os.path.join(get_agent_dir(), "settings.json")


def get_debug_log_path() -> str:
    """Get path to debug log file."""
    return os.path.join(get_agent_dir(), f"{APP_NAME}-debug.log")


def get_share_viewer_url(gist_id: str) -> str:
    base_url = os.environ.get("PI_SHARE_VIEWER_URL", "https://pi.dev/session/")
    return f"{base_url}#{gist_id}"


# ============================================================================
# Legacy aliases kept for backward compatibility
# ============================================================================


def get_global_config_dir() -> str:
    """Get the global Pi config directory (~/.pi)."""
    return os.path.join(os.path.expanduser("~"), CONFIG_DIR_NAME)


def get_global_agent_dir() -> str:
    """Get the global Pi agent directory (~/.pi/agent)."""
    return get_agent_dir()


def get_global_sessions_dir() -> str:
    """Get the sessions directory (~/.pi/agent/sessions)."""
    return get_sessions_dir()


def get_project_config_dir(cwd: str | None = None) -> str:
    """Get the project-local Pi config directory (.pi)."""
    base = cwd or os.getcwd()
    return os.path.join(base, CONFIG_DIR_NAME)


def find_project_root(cwd: str | None = None) -> str:
    """Find the project root by looking for known markers."""
    current = Path(cwd or os.getcwd())
    markers = {".git", "package.json", "pyproject.toml", "Cargo.toml", "go.mod"}

    while True:
        for marker in markers:
            if (current / marker).exists():
                return str(current)
        parent = current.parent
        if parent == current:
            break
        current = parent

    return cwd or os.getcwd()


def is_git_repo(cwd: str | None = None) -> bool:
    """Check if the directory is inside a git repo."""
    current = Path(cwd or os.getcwd())
    while True:
        if (current / ".git").exists():
            return True
        parent = current.parent
        if parent == current:
            break
        current = parent
    return False
