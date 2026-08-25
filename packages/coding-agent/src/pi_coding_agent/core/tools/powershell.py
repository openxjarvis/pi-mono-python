"""
PowerShell execution tool — mirrors packages/coding-agent/src/core/tools/powershell.ts

Windows-only wrapper around the shell tool that runs commands through
PowerShell (pwsh.exe preferred, then powershell.exe).
"""
from __future__ import annotations

from pi_coding_agent.utils.shell import get_powershell_config

from .bash import create_bash_tool

UTF8_OUTPUT_PREFIX = "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}\n"

POWERSHELL_TOOL_SYSTEM_PROMPT_CONTRIBUTION = {
    "snippet": "Execute PowerShell commands",
    "guidelines": ["You can inspect PI_* environment variables for current model and session details."],
}


def create_powershell_tool(cwd: str, command_prefix: str | None = None):
    """Create a PowerShell execution tool. Only available on Windows."""
    shell, args = get_powershell_config()
    prefix = UTF8_OUTPUT_PREFIX + (command_prefix or "")
    return create_bash_tool(
        cwd,
        command_prefix=prefix,
        shell_config=(shell, args),
        name="powershell",
        label="powershell",
        description=(
            "Execute a PowerShell command in the current working directory. "
            "Returns stdout and stderr. Output is truncated. "
            "Optionally provide a timeout in seconds."
        ),
        command_param_description="PowerShell command to execute",
    )
