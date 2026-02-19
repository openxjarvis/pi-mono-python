"""
pi_coding_agent — Coding agent CLI
Python mirror of @mariozechner/pi-coding-agent
"""

from .core.agent_session import AgentSession
from .core.sdk import AgentSessionOptions, create_agent_session
from .core.session_manager import SessionManager
from .core.settings_manager import Settings, SettingsManager
from .core.auth_storage import AuthStorage
from .core.model_registry import ModelRegistry
from .core.system_prompt import build_system_prompt
from .core.tools import (
    create_bash_tool, bash_tool,
    create_edit_tool, edit_tool,
    create_find_tool, find_tool,
    create_grep_tool, grep_tool,
    create_ls_tool, ls_tool,
    create_read_tool, read_tool,
    create_write_tool, write_tool,
)

__all__ = [
    "AgentSession",
    "AgentSessionOptions",
    "create_agent_session",
    "SessionManager",
    "Settings",
    "SettingsManager",
    "AuthStorage",
    "ModelRegistry",
    "build_system_prompt",
    "create_bash_tool", "bash_tool",
    "create_edit_tool", "edit_tool",
    "create_find_tool", "find_tool",
    "create_grep_tool", "grep_tool",
    "create_ls_tool", "ls_tool",
    "create_read_tool", "read_tool",
    "create_write_tool", "write_tool",
]
