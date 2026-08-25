"""
System prompt construction — direct port of packages/coding-agent/src/core/system-prompt.ts
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pi_coding_agent.utils.text import strip_bom

# ── Tool descriptions (mirrors toolDescriptions in TS) ────────────────────────
TOOL_DESCRIPTIONS: dict[str, str] = {
    "read":  "Read file contents",
    "bash":  "Execute bash commands (ls, grep, find, etc.)",
    "edit":  "Make surgical edits to files (find exact text and replace)",
    "write": "Create or overwrite files",
    "grep":  "Search file contents for patterns (respects .gitignore)",
    "find":  "Find files by glob pattern (respects .gitignore)",
    "ls":    "List directory contents",
    "powershell": "Execute PowerShell commands",
}

# Context-file names checked in cwd / parent dirs (mirrors TS config)
SYSTEM_PROMPT_FILENAME  = "SYSTEM.md"
APPEND_SYSTEM_FILENAME  = "APPEND_SYSTEM.md"
AGENTS_FILENAME         = "AGENTS.md"
CLAUDE_FILENAME         = "CLAUDE.md"


# ── Public interface ──────────────────────────────────────────────────────────

def build_system_prompt(
    cwd: str,
    *,
    custom_prompt: str | None = None,
    selected_tools: list[str] | None = None,
    tool_snippets: dict[str, str] | None = None,
    prompt_guidelines: list[str] | None = None,
    append_system_prompt: str | None = None,
    context_files: list[dict[str, str]] | None = None,   # [{"path": ..., "content": ...}]
    skills: list[Any] | None = None,
    # Legacy positional-style aliases kept for back-compat
    base_prompt: str | None = None,
) -> str:
    """
    Build the system prompt with tools, guidelines, and context.
    Direct port of buildSystemPrompt() in TypeScript.

    Priority (when no custom_prompt / SYSTEM.md):
    1. Default prompt with tool list and guidelines
    2. Appended by append_system_prompt / APPEND_SYSTEM.md
    3. Context files (AGENTS.md / CLAUDE.md / explicit list)
    4. Skills section (if read tool is active)
    5. date/time + cwd appended last
    """
    resolved_cwd = cwd.replace("\\", "/")

    now = datetime.now().astimezone()
    date_time = now.strftime("%A, %B %d, %Y, %I:%M:%S %p %Z")

    # Resolve append section
    _append = append_system_prompt or _load_file(cwd, APPEND_SYSTEM_FILENAME)
    append_section = f"\n\n{_append}" if _append else ""

    # Resolve context files list
    _ctx_files: list[dict[str, str]] = context_files or []
    if not _ctx_files:
        for name in (AGENTS_FILENAME, CLAUDE_FILENAME):
            found = _find_file(cwd, name)
            if found:
                _ctx_files = [{"path": name, "content": strip_bom(Path(found).read_text("utf-8")).strip()}]
                break

    _skills: list[dict[str, str]] = skills or []

    # ── Custom / SYSTEM.md path ───────────────────────────────────────────────
    _custom = custom_prompt or base_prompt or _load_file(cwd, SYSTEM_PROMPT_FILENAME)
    if _custom:
        prompt = _custom

        if append_section:
            prompt += append_section

        # Append project context files
        if _ctx_files:
            prompt += _format_project_context(_ctx_files)

        # Append skills (only if read tool is available)
        has_read = not selected_tools or "read" in selected_tools
        if has_read and _skills:
            prompt += _format_skills(selected_tools, _skills)

        prompt += f"\nCurrent date and time: {date_time}"
        prompt += f"\nCurrent working directory: {resolved_cwd}"
        return prompt

    # ── Default prompt ────────────────────────────────────────────────────────
    snippets = tool_snippets or TOOL_DESCRIPTIONS
    tools = selected_tools or ["read", "bash", "edit", "write"]
    visible = [t for t in tools if snippets.get(t)]
    tools_list = (
        "\n".join(f"- {t}: {snippets[t]}" for t in visible)
        if visible else "(none)"
    )

    has_bash = "bash" in tools
    has_powershell = "powershell" in tools
    has_grep = "grep" in tools
    has_find = "find" in tools
    has_ls = "ls" in tools
    has_read = "read" in tools

    guidelines: list[str] = []
    seen: set[str] = set()

    def add_guideline(text: str) -> None:
        if text and text not in seen:
            seen.add(text)
            guidelines.append(text)

    if (has_bash or has_powershell) and not has_grep and not has_find and not has_ls:
        if has_bash and has_powershell:
            add_guideline("Use bash or PowerShell for file operations like listing, searching, and finding files")
        elif has_powershell:
            add_guideline("Use PowerShell for file operations like listing, searching, and finding files")
        else:
            add_guideline("Use bash for file operations like ls, rg, find")

    for guideline in prompt_guidelines or []:
        normalized = guideline.strip()
        if normalized:
            add_guideline(normalized)

    add_guideline("Be concise in your responses")
    add_guideline("Show file paths clearly when working with files")

    guidelines_block = "\n".join(f"- {g}" for g in guidelines)

    # Pi docs paths — point to the Python project's own README/docs
    import os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    readme_path   = _os.path.join(_pkg_root, "README.md")
    docs_path     = _os.path.join(_pkg_root, "docs")
    examples_path = _os.path.join(_pkg_root, "examples")

    prompt = (
        f"You are an expert coding assistant operating inside pi, a coding agent harness. "
        f"You help users by reading files, executing commands, editing code, and writing new files.\n\n"
        f"Available tools:\n{tools_list}\n\n"
        f"In addition to the tools above, you may have access to other custom tools depending on the project.\n\n"
        f"Guidelines:\n{guidelines_block}\n\n"
        f"Pi documentation (read only when the user asks about pi itself, its SDK, extensions, or TUI):\n"
        f"- Main documentation: {readme_path}\n"
        f"- Additional docs: {docs_path}\n"
        f"- Examples: {examples_path} (extensions, custom tools, SDK)"
    )

    if append_section:
        prompt += append_section

    # Context files
    if _ctx_files:
        prompt += _format_project_context(_ctx_files)

    # Skills
    if has_read and _skills:
        prompt += _format_skills(selected_tools, _skills)

    prompt += f"\nCurrent date and time: {date_time}"
    prompt += f"\nCurrent working directory: {resolved_cwd}"

    return prompt


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_file(cwd: str, filename: str) -> str | None:
    """Search for filename in cwd and parent directories."""
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


def _load_file(cwd: str, filename: str) -> str | None:
    """Load file content if found, else None."""
    path = _find_file(cwd, filename)
    if path:
        content = strip_bom(Path(path).read_text("utf-8")).strip()
        return content or None
    return None


def _format_project_context(context_files: list[dict[str, str]]) -> str:
    """Format AGENTS.md-style context using TS <project_context> tags."""
    parts = [
        "\n\n<project_context>\n\n",
        "Project-specific instructions and guidelines:\n\n",
    ]
    for cf in context_files:
        parts.append(f"<project_instructions path=\"{cf['path']}\">\n{cf['content']}\n</project_instructions>\n\n")
    parts.append("</project_context>\n")
    return "".join(parts)


def _format_skills(selected_tools: list[str] | None, skills: list[Any]) -> str:
    """Format skills section for system prompt (mirrors formatSkillsForPrompt in TS)."""
    if not skills:
        return ""
    from pi_coding_agent.core.skills import Skill, format_skills_for_prompt

    skill_objs: list[Skill] = []
    leftover: list[dict[str, str]] = []
    for skill in skills:
        if isinstance(skill, Skill):
            skill_objs.append(skill)
        elif isinstance(skill, dict):
            leftover.append(skill)
    formatted = format_skills_for_prompt(skill_objs) if skill_objs else ""
    if leftover:
        parts = ["\n\n## Skills\n"]
        for skill in leftover:
            name = skill.get("name", "unknown")
            content = skill.get("content", "").strip()
            parts.append(f"### {name}\n{content}")
        formatted += "\n\n".join(parts)
    return formatted
