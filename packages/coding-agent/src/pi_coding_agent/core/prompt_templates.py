"""
Prompt template management.

Loads and processes markdown prompt template files from configured directories.
Templates support argument substitution via $1, $@, $ARGUMENTS, etc.

Mirrors core/prompt-templates.ts
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from pi_coding_agent.utils.frontmatter import parse_frontmatter


@dataclass
class PromptTemplate:
    """A prompt template loaded from a markdown file."""

    name: str
    description: str
    content: str
    source: str
    file_path: str


def parse_command_args(args_string: str) -> list[str]:
    """Parse command arguments respecting quoted strings (bash-style).

    Returns list of arguments.
    """
    args: list[str] = []
    current = ""
    in_quote: str | None = None

    for ch in args_string:
        if in_quote:
            if ch == in_quote:
                in_quote = None
            else:
                current += ch
        elif ch in ('"', "'"):
            in_quote = ch
        elif ch in (" ", "\t"):
            if current:
                args.append(current)
                current = ""
        else:
            current += ch

    if current:
        args.append(current)

    return args


def substitute_args(content: str, args: list[str]) -> str:
    """Substitute argument placeholders in template content.

    Supports:
    - $1, $2, ... for positional args
    - $@ and $ARGUMENTS for all args joined
    - ${@:N} for args from Nth onwards (1-indexed)
    - ${@:N:L} for L args starting from Nth
    """
    def _pos_replace(m: re.Match) -> str:
        idx = int(m.group(1)) - 1
        return args[idx] if 0 <= idx < len(args) else ""

    result = re.sub(r"\$(\d+)", _pos_replace, content)

    def _slice_replace(m: re.Match) -> str:
        start = max(int(m.group(1)) - 1, 0)
        length_str = m.group(2)
        if length_str is not None:
            length = int(length_str)
            return " ".join(args[start : start + length])
        return " ".join(args[start:])

    result = re.sub(r"\$\{@:(\d+)(?::(\d+))?\}", _slice_replace, result)

    all_args = " ".join(args)
    result = result.replace("$ARGUMENTS", all_args)
    result = result.replace("$@", all_args)

    return result


def _load_template_from_file(
    file_path: str, source: str, source_label: str
) -> PromptTemplate | None:
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            raw_content = f.read()

        frontmatter, body = parse_frontmatter(raw_content)
        name = os.path.splitext(os.path.basename(file_path))[0]

        description = frontmatter.get("description", "")
        if not description:
            for line in body.splitlines():
                stripped = line.strip()
                if stripped:
                    description = stripped[:60]
                    if len(stripped) > 60:
                        description += "..."
                    break

        description = f"{description} {source_label}" if description else source_label

        return PromptTemplate(
            name=name,
            description=description,
            content=body,
            source=source,
            file_path=file_path,
        )
    except Exception:
        return None


def _load_templates_from_dir(
    dir_path: str, source: str, source_label: str
) -> list[PromptTemplate]:
    templates: list[PromptTemplate] = []
    if not os.path.isdir(dir_path):
        return templates

    try:
        entries = sorted(os.scandir(dir_path), key=lambda e: e.name)
    except OSError:
        return templates

    for entry in entries:
        full_path = entry.path
        is_file = False
        try:
            if entry.is_symlink():
                is_file = os.path.isfile(full_path)
            else:
                is_file = entry.is_file()
        except OSError:
            continue

        if is_file and entry.name.endswith(".md"):
            tmpl = _load_template_from_file(full_path, source, source_label)
            if tmpl:
                templates.append(tmpl)

    return templates


@dataclass
class LoadPromptTemplatesOptions:
    cwd: str | None = None
    agent_dir: str | None = None
    prompt_paths: list[str] = field(default_factory=list)
    include_defaults: bool = True


def load_prompt_templates(
    options: LoadPromptTemplatesOptions | None = None,
) -> list[PromptTemplate]:
    """Load prompt templates from all configured locations.

    Order:
    1. Global (agentDir/prompts/)
    2. Project (cwd/{CONFIG_DIR_NAME}/prompts/)
    3. Explicit paths
    """
    from pi_coding_agent.config import CONFIG_DIR_NAME, get_prompts_dir

    opts = options or LoadPromptTemplatesOptions()
    cwd = opts.cwd or os.getcwd()
    prompts_dir = opts.agent_dir if opts.agent_dir else get_prompts_dir()
    prompt_paths = opts.prompt_paths or []
    include_defaults = opts.include_defaults

    templates: list[PromptTemplate] = []

    if include_defaults:
        global_dir = (
            os.path.join(opts.agent_dir, "prompts") if opts.agent_dir else prompts_dir
        )
        templates.extend(_load_templates_from_dir(global_dir, "user", "(user)"))

        project_dir = os.path.join(cwd, CONFIG_DIR_NAME, "prompts")
        templates.extend(_load_templates_from_dir(project_dir, "project", "(project)"))

    user_prompts_dir = (
        os.path.join(opts.agent_dir, "prompts") if opts.agent_dir else prompts_dir
    )
    project_prompts_dir = os.path.join(cwd, CONFIG_DIR_NAME, "prompts")

    def _is_under(target: str, root: str) -> bool:
        norm_root = os.path.abspath(root)
        norm_target = os.path.abspath(target)
        return norm_target == norm_root or norm_target.startswith(norm_root + os.sep)

    def get_source_info(resolved_path: str) -> tuple[str, str]:
        if not include_defaults:
            if _is_under(resolved_path, user_prompts_dir):
                return "user", "(user)"
            if _is_under(resolved_path, project_prompts_dir):
                return "project", "(project)"
        base = os.path.splitext(os.path.basename(resolved_path))[0] or "path"
        return "path", f"(path:{base})"

    home = os.path.expanduser("~")

    def _normalize(p: str) -> str:
        t = p.strip()
        if t == "~":
            return home
        if t.startswith("~/"):
            return os.path.join(home, t[2:])
        if t.startswith("~"):
            return os.path.join(home, t[1:])
        return t

    for raw_path in prompt_paths:
        norm = _normalize(raw_path)
        resolved = norm if os.path.isabs(norm) else os.path.abspath(os.path.join(cwd, norm))

        if not os.path.exists(resolved):
            continue

        try:
            source, label = get_source_info(resolved)
            if os.path.isdir(resolved):
                templates.extend(_load_templates_from_dir(resolved, source, label))
            elif os.path.isfile(resolved) and resolved.endswith(".md"):
                tmpl = _load_template_from_file(resolved, source, label)
                if tmpl:
                    templates.append(tmpl)
        except Exception:
            pass

    return templates


def expand_prompt_template(text: str, templates: list[PromptTemplate]) -> str:
    """Expand a template reference if text starts with '/'.

    Returns expanded content or original text if not a template.
    """
    if not text.startswith("/"):
        return text

    space_idx = text.find(" ")
    template_name = text[1:] if space_idx == -1 else text[1:space_idx]
    args_string = "" if space_idx == -1 else text[space_idx + 1:]

    tmpl = next((t for t in templates if t.name == template_name), None)
    if tmpl:
        args = parse_command_args(args_string)
        return substitute_args(tmpl.content, args)

    return text
