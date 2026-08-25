"""
Slash command definitions.

Defines built-in slash commands and the SlashCommandInfo type.

Mirrors core/slash-commands.ts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SlashCommandSource = Literal["extension", "prompt", "skill"]
SlashCommandLocation = Literal["user", "project", "path"]


@dataclass
class SlashCommandInfo:
    name: str
    description: str | None = None
    source: SlashCommandSource = "skill"
    location: SlashCommandLocation | None = None
    path: str | None = None


@dataclass
class BuiltinSlashCommand:
    name: str
    description: str
    argument_hint: str | None = None


BUILTIN_SLASH_COMMANDS: list[BuiltinSlashCommand] = [
    BuiltinSlashCommand("settings", "Open settings menu"),
    BuiltinSlashCommand("model", "Select model (opens selector UI)", argument_hint="<provider/model>"),
    BuiltinSlashCommand("tree", "Navigate session tree (switch branches)"),
    BuiltinSlashCommand("thinking", "Set thinking level", argument_hint="<level>"),
    BuiltinSlashCommand("scoped-models", "Enable/disable models for Ctrl+P cycling"),
    BuiltinSlashCommand("export", "Export session (HTML default, or specify path: .html/.jsonl)"),
    BuiltinSlashCommand("import", "Import and resume a session from a JSONL file"),
    BuiltinSlashCommand("share", "Share session as a secret GitHub gist"),
    BuiltinSlashCommand("copy", "Copy last agent message to clipboard"),
    BuiltinSlashCommand("name", "Set session display name"),
    BuiltinSlashCommand("session", "Show session info and stats"),
    BuiltinSlashCommand("changelog", "Show changelog entries"),
    BuiltinSlashCommand("hotkeys", "Show all keyboard shortcuts"),
    BuiltinSlashCommand("fork", "Create a new fork from a previous user message"),
    BuiltinSlashCommand("clone", "Duplicate the current session at the current position"),
    BuiltinSlashCommand("trust", "Save project trust decision for future sessions"),
    BuiltinSlashCommand("login", "Configure provider authentication", argument_hint="<provider>"),
    BuiltinSlashCommand("logout", "Remove provider authentication"),
    BuiltinSlashCommand("new", "Start a new session"),
    BuiltinSlashCommand("compact", "Manually compact the session context"),
    BuiltinSlashCommand("resume", "Resume a different session"),
    BuiltinSlashCommand("reload", "Reload keybindings, extensions, skills, prompts, themes, and context files"),
    BuiltinSlashCommand("quit", "Quit pi"),
]
