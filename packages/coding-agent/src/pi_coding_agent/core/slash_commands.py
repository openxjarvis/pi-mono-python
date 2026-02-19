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


BUILTIN_SLASH_COMMANDS: list[BuiltinSlashCommand] = [
    BuiltinSlashCommand("settings", "Open settings menu"),
    BuiltinSlashCommand("model", "Select model (opens selector UI)"),
    BuiltinSlashCommand("scoped-models", "Enable/disable models for Ctrl+P cycling"),
    BuiltinSlashCommand("export", "Export session to HTML file"),
    BuiltinSlashCommand("share", "Share session as a secret GitHub gist"),
    BuiltinSlashCommand("copy", "Copy last agent message to clipboard"),
    BuiltinSlashCommand("name", "Set session display name"),
    BuiltinSlashCommand("session", "Show session info and stats"),
    BuiltinSlashCommand("changelog", "Show changelog entries"),
    BuiltinSlashCommand("hotkeys", "Show all keyboard shortcuts"),
    BuiltinSlashCommand("fork", "Create a new fork from a previous message"),
    BuiltinSlashCommand("tree", "Navigate session tree (switch branches)"),
    BuiltinSlashCommand("login", "Login with OAuth provider"),
    BuiltinSlashCommand("logout", "Logout from OAuth provider"),
    BuiltinSlashCommand("new", "Start a new session"),
    BuiltinSlashCommand("compact", "Manually compact the session context"),
    BuiltinSlashCommand("resume", "Resume a different session"),
    BuiltinSlashCommand("reload", "Reload extensions, skills, prompts, and themes"),
    BuiltinSlashCommand("quit", "Quit pi"),
]
