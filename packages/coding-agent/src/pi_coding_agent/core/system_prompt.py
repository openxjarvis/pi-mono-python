"""
System prompt construction — direct port of packages/coding-agent/src/core/system-prompt.ts
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

# ── Tool descriptions (mirrors toolDescriptions in TS) ────────────────────────
TOOL_DESCRIPTIONS: dict[str, str] = {
    "read":  "Read file contents",
    "bash":  "Execute bash commands (ls, grep, find, etc.)",
    "edit":  "Make surgical edits to files (find exact text and replace)",
    "write": "Create or overwrite files",
    "grep":  "Search file contents for patterns (respects .gitignore)",
    "find":  "Find files by glob pattern (respects .gitignore)",
    "ls":    "List directory contents",
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
    append_system_prompt: str | None = None,
    context_files: list[dict[str, str]] | None = None,   # [{"path": ..., "content": ...}]
    skills: list[dict[str, str]] | None = None,          # [{"name": ..., "content": ...}]
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
    resolved_cwd = cwd

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
                _ctx_files = [{"path": name, "content": Path(found).read_text("utf-8").strip()}]
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
            prompt += "\n\n# Project Context\n\n"
            prompt += "Project-specific instructions and guidelines:\n\n"
            for cf in _ctx_files:
                prompt += f"## {cf['path']}\n\n{cf['content']}\n\n"

        # Append skills (only if read tool is available)
        has_read = not selected_tools or "read" in selected_tools
        if has_read and _skills:
            prompt += _format_skills(selected_tools, _skills)

        prompt += f"\nCurrent date and time: {date_time}"
        prompt += f"\nCurrent working directory: {resolved_cwd}"
        return prompt

    # ── Default prompt ────────────────────────────────────────────────────────
    tools = [t for t in (selected_tools or ["read", "bash", "edit", "write"])
             if t in TOOL_DESCRIPTIONS]
    tools_list = (
        "\n".join(f"- {t}: {TOOL_DESCRIPTIONS[t]}" for t in tools)
        if tools else "(none)"
    )

    has_bash  = "bash"  in tools
    has_edit  = "edit"  in tools
    has_write = "write" in tools
    has_grep  = "grep"  in tools
    has_find  = "find"  in tools
    has_ls    = "ls"    in tools
    has_read  = "read"  in tools

    guidelines: list[str] = []

    # File exploration
    if has_bash and not (has_grep or has_find or has_ls):
        guidelines.append("Use bash for file operations like ls, rg, find")
    elif has_bash and (has_grep or has_find or has_ls):
        guidelines.append(
            "Prefer grep/find/ls tools over bash for file exploration (faster, respects .gitignore)"
        )

    # Read before edit
    if has_read and has_edit:
        guidelines.append(
            "Use read to examine files before editing. "
            "You must use this tool instead of cat or sed."
        )

    # Edit
    if has_edit:
        guidelines.append("Use edit for precise changes (old text must match exactly)")

    # Write
    if has_write:
        guidelines.append("Use write only for new files or complete rewrites")

    # Output summary
    if has_edit or has_write:
        guidelines.append(
            "When summarizing your actions, output plain text directly"
            " - do NOT use cat or bash to display what you did"
        )

    # Always include
    guidelines.append("Be concise in your responses")
    guidelines.append("Show file paths clearly when working with files")

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
        prompt += "\n\n# Project Context\n\n"
        prompt += "Project-specific instructions and guidelines:\n\n"
        for cf in _ctx_files:
            prompt += f"## {cf['path']}\n\n{cf['content']}\n\n"

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
        content = Path(path).read_text("utf-8").strip()
        return content or None
    return None


def _format_skills(selected_tools: list[str] | None, skills: list[dict[str, str]]) -> str:
    """Format skills section for system prompt (mirrors formatSkillsForPrompt in TS)."""
    if not skills:
        return ""
    parts = ["\n\n## Skills\n"]
    for skill in skills:
        name = skill.get("name", "unknown")
        content = skill.get("content", "").strip()
        parts.append(f"### {name}\n{content}")
    return "\n\n".join(parts)
