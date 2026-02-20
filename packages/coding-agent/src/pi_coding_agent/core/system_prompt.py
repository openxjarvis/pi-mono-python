"""
System prompt builder — mirrors packages/coding-agent/src/core/system-prompt.ts
"""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

SYSTEM_PROMPT_FILENAME = "SYSTEM.md"
APPEND_SYSTEM_FILENAME = "APPEND_SYSTEM.md"
AGENTS_FILENAME = "AGENTS.md"
CLAUDE_FILENAME = "CLAUDE.md"


def build_system_prompt(
    cwd: str,
    base_prompt: str | None = None,
    include_cwd: bool = True,
    selected_tools: list[str] | None = None,
) -> str:
    """
    Build the system prompt for the coding agent.
    Mirrors buildSystemPrompt() in TypeScript.

    Priority:
    1. SYSTEM.md file in cwd or parent dirs
    2. Base prompt (default)
    3. Appended by APPEND_SYSTEM.md if found
    4. AGENTS.md / CLAUDE.md context files
    """
    prompt_parts: list[str] = []

    # Check for SYSTEM.md override
    system_file = _find_file(cwd, SYSTEM_PROMPT_FILENAME)
    if system_file:
        with open(system_file, encoding="utf-8") as f:
            prompt_parts.append(f.read().strip())
    elif base_prompt:
        prompt_parts.append(base_prompt)
    else:
        prompt_parts.append(_default_system_prompt(cwd, selected_tools))

    # Append APPEND_SYSTEM.md if found
    append_file = _find_file(cwd, APPEND_SYSTEM_FILENAME)
    if append_file:
        with open(append_file, encoding="utf-8") as f:
            append_content = f.read().strip()
        if append_content:
            prompt_parts.append(append_content)

    # Include context files (AGENTS.md, CLAUDE.md)
    for context_file in [AGENTS_FILENAME, CLAUDE_FILENAME]:
        found = _find_file(cwd, context_file)
        if found:
            with open(found, encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                prompt_parts.append(f"## {context_file}\n\n{content}")
            break  # Only include first found

    return "\n\n".join(p for p in prompt_parts if p)


def _default_system_prompt(cwd: str, selected_tools: list[str] | None = None) -> str:
    """Default system prompt aligned with TypeScript coding-agent guidance."""
    tool_descriptions = {
        "read": "Read file contents",
        "bash": "Execute any shell command: run Python/Node/shell scripts, compile, test, install packages, etc.",
        "edit": "Make surgical edits to files (find exact text and replace)",
        "write": "Create or overwrite files",
        "grep": "Search file contents for patterns (respects .gitignore)",
        "find": "Find files by glob pattern (respects .gitignore)",
        "ls": "List directory contents",
    }
    default_tools = ["read", "bash", "edit", "write"]
    tools = [t for t in (selected_tools or default_tools) if t in tool_descriptions]
    tools_list = "\n".join(f"- {t}: {tool_descriptions[t]}" for t in tools) if tools else "- (none)"

    has_bash = "bash" in tools
    has_edit = "edit" in tools
    has_write = "write" in tools
    has_read = "read" in tools
    has_grep = "grep" in tools
    has_find = "find" in tools
    has_ls = "ls" in tools

    guidelines: list[str] = []
    if has_bash and not (has_grep or has_find or has_ls):
        guidelines.append("Use bash for file exploration (ls, rg, find).")
    elif has_bash:
        guidelines.append("Prefer grep/find/ls tools over bash for exploration where available.")
    if has_read and has_edit:
        guidelines.append("Read files before editing; do not edit blind.")
    if has_edit:
        guidelines.append("Use edit for precise changes; old text must match exactly.")
    if has_write:
        guidelines.append("Use write for new files or complete rewrites.")
    if has_edit or has_write:
        guidelines.append("After writing or editing code, run it with bash to verify it works.")
    if has_bash:
        guidelines.append(
            "When the user asks to run, execute, or test something, ALWAYS use the bash tool to do it immediately. "
            "Never tell the user to run it themselves."
        )
    guidelines.append("Be concise and action-oriented.")
    guidelines.append("Show clear file paths when reporting changes.")
    guidelines.append(
        "Do not claim you cannot execute commands. If bash is available, use it."
    )
    guidelines_block = "\n".join(f"- {g}" for g in guidelines)

    now = datetime.now().astimezone()
    date_time = now.strftime("%A, %Y-%m-%d %H:%M:%S %Z")

    return f"""You are an expert coding assistant operating inside pi, a coding agent harness.
You help users by reading files, executing commands, editing code, and writing files.

Available tools:
{tools_list}

Guidelines:
{guidelines_block}

Current date and time: {date_time}
Current working directory: {cwd}"""


def _find_file(cwd: str, filename: str) -> str | None:
    """Search for a file in cwd and parent directories."""
    current = Path(cwd)
    while True:
        candidate = current / filename
        if candidate.exists():
            return str(candidate)
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None
