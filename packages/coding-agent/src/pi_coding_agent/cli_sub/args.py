"""
CLI argument parsing and help display.

Mirrors packages/coding-agent/src/cli/args.ts
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Literal

from pi_coding_agent.config import APP_NAME, CONFIG_DIR_NAME, ENV_AGENT_DIR

Mode = Literal["text", "json", "rpc"]

VALID_THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh")


def is_valid_thinking_level(level: str) -> bool:
    return level in VALID_THINKING_LEVELS


@dataclass
class Args:
    messages: list[str] = field(default_factory=list)
    file_args: list[str] = field(default_factory=list)
    unknown_flags: dict[str, bool | str] = field(default_factory=dict)

    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    thinking: str | None = None
    continue_: bool = False
    resume: bool = False
    help: bool = False
    version: bool = False
    mode: Mode | None = None
    no_session: bool = False
    session: str | None = None
    session_dir: str | None = None
    models: list[str] | None = None
    tools: list[str] | None = None
    no_tools: bool = False
    extensions: list[str] | None = None
    no_extensions: bool = False
    print_mode: bool = False
    export: str | None = None
    no_skills: bool = False
    skills: list[str] | None = None
    prompt_templates: list[str] | None = None
    no_prompt_templates: bool = False
    themes: list[str] | None = None
    no_themes: bool = False
    list_models: str | bool | None = None
    verbose: bool = False


VALID_TOOL_NAMES = {"read", "bash", "edit", "write", "grep", "find", "ls"}


def parse_args(
    args: list[str],
    extension_flags: dict[str, Literal["boolean", "string"]] | None = None,
) -> Args:
    """Parse CLI arguments into an Args dataclass."""
    result = Args()
    i = 0
    while i < len(args):
        arg = args[i]

        if arg in ("--help", "-h"):
            result.help = True
        elif arg in ("--version", "-v"):
            result.version = True
        elif arg == "--mode" and i + 1 < len(args):
            i += 1
            m = args[i]
            if m in ("text", "json", "rpc"):
                result.mode = m  # type: ignore[assignment]
        elif arg in ("--continue", "-c"):
            result.continue_ = True
        elif arg in ("--resume", "-r"):
            result.resume = True
        elif arg == "--provider" and i + 1 < len(args):
            i += 1
            result.provider = args[i]
        elif arg == "--model" and i + 1 < len(args):
            i += 1
            result.model = args[i]
        elif arg == "--api-key" and i + 1 < len(args):
            i += 1
            result.api_key = args[i]
        elif arg == "--system-prompt" and i + 1 < len(args):
            i += 1
            result.system_prompt = args[i]
        elif arg == "--append-system-prompt" and i + 1 < len(args):
            i += 1
            result.append_system_prompt = args[i]
        elif arg == "--no-session":
            result.no_session = True
        elif arg == "--session" and i + 1 < len(args):
            i += 1
            result.session = args[i]
        elif arg == "--session-dir" and i + 1 < len(args):
            i += 1
            result.session_dir = args[i]
        elif arg == "--models" and i + 1 < len(args):
            i += 1
            result.models = [s.strip() for s in args[i].split(",")]
        elif arg == "--no-tools":
            result.no_tools = True
        elif arg == "--tools" and i + 1 < len(args):
            i += 1
            tool_names = [s.strip() for s in args[i].split(",")]
            valid: list[str] = []
            for name in tool_names:
                if name in VALID_TOOL_NAMES:
                    valid.append(name)
                else:
                    print(
                        f"Warning: Unknown tool \"{name}\". Valid tools: {', '.join(sorted(VALID_TOOL_NAMES))}",
                        file=sys.stderr,
                    )
            result.tools = valid
        elif arg == "--thinking" and i + 1 < len(args):
            i += 1
            level = args[i]
            if is_valid_thinking_level(level):
                result.thinking = level
            else:
                print(
                    f'Warning: Invalid thinking level "{level}". Valid values: {", ".join(VALID_THINKING_LEVELS)}',
                    file=sys.stderr,
                )
        elif arg in ("--print", "-p"):
            result.print_mode = True
        elif arg == "--export" and i + 1 < len(args):
            i += 1
            result.export = args[i]
        elif arg in ("--extension", "-e") and i + 1 < len(args):
            i += 1
            if result.extensions is None:
                result.extensions = []
            result.extensions.append(args[i])
        elif arg in ("--no-extensions", "-ne"):
            result.no_extensions = True
        elif arg == "--skill" and i + 1 < len(args):
            i += 1
            if result.skills is None:
                result.skills = []
            result.skills.append(args[i])
        elif arg == "--prompt-template" and i + 1 < len(args):
            i += 1
            if result.prompt_templates is None:
                result.prompt_templates = []
            result.prompt_templates.append(args[i])
        elif arg == "--theme" and i + 1 < len(args):
            i += 1
            if result.themes is None:
                result.themes = []
            result.themes.append(args[i])
        elif arg in ("--no-skills", "-ns"):
            result.no_skills = True
        elif arg in ("--no-prompt-templates", "-np"):
            result.no_prompt_templates = True
        elif arg == "--no-themes":
            result.no_themes = True
        elif arg == "--list-models":
            if i + 1 < len(args) and not args[i + 1].startswith("-") and not args[i + 1].startswith("@"):
                i += 1
                result.list_models = args[i]
            else:
                result.list_models = True
        elif arg == "--verbose":
            result.verbose = True
        elif arg.startswith("@"):
            result.file_args.append(arg[1:])
        elif arg.startswith("--") and extension_flags:
            flag_name = arg[2:]
            ext_flag = extension_flags.get(flag_name)
            if ext_flag:
                if ext_flag == "boolean":
                    result.unknown_flags[flag_name] = True
                elif ext_flag == "string" and i + 1 < len(args):
                    i += 1
                    result.unknown_flags[flag_name] = args[i]
        elif not arg.startswith("-"):
            result.messages.append(arg)

        i += 1

    return result


def print_help() -> None:
    """Print CLI help text."""
    print(f"""{APP_NAME} - AI coding assistant with read, bash, edit, write tools

Usage:
  {APP_NAME} [options] [@files...] [messages...]

Commands:
  {APP_NAME} install <source> [-l]    Install extension source and add to settings
  {APP_NAME} remove <source> [-l]     Remove extension source from settings
  {APP_NAME} update [source]          Update installed extensions (skips pinned sources)
  {APP_NAME} list                     List installed extensions from settings
  {APP_NAME} config                   Open TUI to enable/disable package resources
  {APP_NAME} <command> --help         Show help for install/remove/update/list

Options:
  --provider <name>              Provider name (default: google)
  --model <pattern>              Model pattern or ID (supports "provider/id" and optional ":<thinking>")
  --api-key <key>                API key (defaults to env vars)
  --system-prompt <text>         System prompt (default: coding assistant prompt)
  --append-system-prompt <text>  Append text or file contents to the system prompt
  --mode <mode>                  Output mode: text (default), json, or rpc
  --print, -p                    Non-interactive mode: process prompt and exit
  --continue, -c                 Continue previous session
  --resume, -r                   Select a session to resume
  --session <path>               Use specific session file
  --session-dir <dir>            Directory for session storage and lookup
  --no-session                   Don't save session (ephemeral)
  --models <patterns>            Comma-separated model patterns for cycling
  --no-tools                     Disable all built-in tools
  --tools <tools>                Comma-separated list of tools to enable
  --thinking <level>             Set thinking level: {', '.join(VALID_THINKING_LEVELS)}
  --extension, -e <path>         Load an extension file (can be used multiple times)
  --no-extensions, -ne           Disable extension discovery
  --skill <path>                 Load a skill file or directory
  --no-skills, -ns               Disable skills discovery and loading
  --prompt-template <path>       Load a prompt template file or directory
  --no-prompt-templates, -np     Disable prompt template discovery and loading
  --theme <path>                 Load a theme file or directory
  --no-themes                    Disable theme discovery and loading
  --export <file>                Export session file to HTML and exit
  --list-models [search]         List available models
  --verbose                      Force verbose startup
  --help, -h                     Show this help
  --version, -v                  Show version number

Environment Variables:
  ANTHROPIC_API_KEY                - Anthropic Claude API key
  OPENAI_API_KEY                   - OpenAI GPT API key
  GEMINI_API_KEY                   - Google Gemini API key
  {ENV_AGENT_DIR:<32} - Session storage directory (default: ~/.{CONFIG_DIR_NAME}/agent)
""")
