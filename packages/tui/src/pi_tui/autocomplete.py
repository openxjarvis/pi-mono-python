"""
Autocomplete providers — mirrors packages/tui/src/autocomplete.ts

Provides:
- AutocompleteItem: value/label/description tuple
- SlashCommand: slash command definition
- AutocompleteProvider: abstract base
- CombinedAutocompleteProvider: slash commands + file paths + @ references
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from .fuzzy import fuzzy_filter

_PATH_DELIMITERS = frozenset([" ", "\t", '"', "'", "="])


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AutocompleteItem:
    value: str
    label: str
    description: str | None = None


@dataclass
class SlashCommand:
    name: str
    description: str | None = None
    get_argument_completions: Callable[[str], list[AutocompleteItem] | None] | None = None


@dataclass
class SuggestionResult:
    items: list[AutocompleteItem]
    prefix: str


@dataclass
class ApplyResult:
    lines: list[str]
    cursor_line: int
    cursor_col: int


# ─────────────────────────────────────────────────────────────────────────────
# AutocompleteProvider protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class AutocompleteProvider(Protocol):
    def get_suggestions(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
    ) -> SuggestionResult | None: ...

    def apply_completion(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
        item: AutocompleteItem,
        prefix: str,
    ) -> ApplyResult: ...


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — mirrors private functions in autocomplete.ts
# ─────────────────────────────────────────────────────────────────────────────

def _find_last_delimiter(text: str) -> int:
    for i in range(len(text) - 1, -1, -1):
        if text[i] in _PATH_DELIMITERS:
            return i
    return -1


def _find_unclosed_quote_start(text: str) -> int | None:
    in_quotes = False
    quote_start = -1
    for i, ch in enumerate(text):
        if ch == '"':
            in_quotes = not in_quotes
            if in_quotes:
                quote_start = i
    return quote_start if in_quotes else None


def _is_token_start(text: str, index: int) -> bool:
    return index == 0 or text[index - 1] in _PATH_DELIMITERS


def _extract_quoted_prefix(text: str) -> str | None:
    quote_start = _find_unclosed_quote_start(text)
    if quote_start is None:
        return None
    if quote_start > 0 and text[quote_start - 1] == "@":
        if not _is_token_start(text, quote_start - 1):
            return None
        return text[quote_start - 1:]
    if not _is_token_start(text, quote_start):
        return None
    return text[quote_start:]


def _parse_path_prefix(prefix: str) -> tuple[str, bool, bool]:
    """Returns (raw_prefix, is_at_prefix, is_quoted_prefix)."""
    if prefix.startswith('@"'):
        return prefix[2:], True, True
    if prefix.startswith('"'):
        return prefix[1:], False, True
    if prefix.startswith("@"):
        return prefix[1:], True, False
    return prefix, False, False


def _build_completion_value(
    path: str,
    is_directory: bool,
    is_at_prefix: bool,
    is_quoted_prefix: bool,
) -> str:
    needs_quotes = is_quoted_prefix or " " in path
    prefix = "@" if is_at_prefix else ""
    if not needs_quotes:
        return f"{prefix}{path}"
    return f'{prefix}"{path}"'


def _walk_directory_with_fd(
    base_dir: str,
    fd_path: str,
    query: str,
    max_results: int = 100,
) -> list[tuple[str, bool]]:
    """Walk directory using fd. Returns list of (path, is_directory)."""
    args = [
        fd_path,
        "--base-directory", base_dir,
        "--max-results", str(max_results),
        "--type", "f",
        "--type", "d",
        "--full-path",
        "--hidden",
        "--exclude", ".git",
        "--exclude", ".git/*",
        "--exclude", ".git/**",
    ]
    if query:
        args.append(query)

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout:
            return []
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []

    entries: list[tuple[str, bool]] = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        normalized = line.rstrip("/")
        if normalized == ".git" or normalized.startswith(".git/") or "/.git/" in normalized:
            continue
        is_directory = line.endswith("/")
        entries.append((line, is_directory))
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# CombinedAutocompleteProvider
# ─────────────────────────────────────────────────────────────────────────────

class CombinedAutocompleteProvider:
    """
    Handles slash commands, file path completion, and @ file references.
    Mirrors CombinedAutocompleteProvider in autocomplete.ts.
    """

    def __init__(
        self,
        commands: list[SlashCommand | AutocompleteItem] | None = None,
        base_path: str | None = None,
        fd_path: str | None = None,
    ) -> None:
        self.commands = commands or []
        self.base_path = base_path or os.getcwd()
        self.fd_path = fd_path

    def get_suggestions(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
    ) -> SuggestionResult | None:
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""
        text_before = current_line[:cursor_col]

        # @ file reference
        at_prefix = self._extract_at_prefix(text_before)
        if at_prefix is not None:
            raw_prefix, _, is_quoted = _parse_path_prefix(at_prefix)
            suggestions = self._get_fuzzy_file_suggestions(raw_prefix, is_quoted_prefix=is_quoted)
            if not suggestions:
                return None
            return SuggestionResult(items=suggestions, prefix=at_prefix)

        # Slash commands
        if text_before.startswith("/"):
            space_idx = text_before.find(" ")
            if space_idx == -1:
                prefix = text_before[1:]
                cmd_items = []
                for cmd in self.commands:
                    name = cmd.name if isinstance(cmd, SlashCommand) else cmd.value
                    label = cmd.name if isinstance(cmd, SlashCommand) else cmd.label
                    desc = cmd.description
                    cmd_items.append(AutocompleteItem(value=name, label=label, description=desc))
                filtered = fuzzy_filter(cmd_items, prefix, lambda x: x.value)
                if not filtered:
                    return None
                return SuggestionResult(items=filtered, prefix=text_before)
            else:
                cmd_name = text_before[1:space_idx]
                arg_text = text_before[space_idx + 1:]
                command = None
                for cmd in self.commands:
                    n = cmd.name if isinstance(cmd, SlashCommand) else cmd.value
                    if n == cmd_name:
                        command = cmd
                        break
                if command is None or not isinstance(command, SlashCommand) or not command.get_argument_completions:
                    return None
                arg_suggestions = command.get_argument_completions(arg_text)
                if not arg_suggestions:
                    return None
                return SuggestionResult(items=arg_suggestions, prefix=arg_text)

        # File paths
        path_match = self._extract_path_prefix(text_before, force=False)
        if path_match is not None:
            suggestions = self._get_file_suggestions(path_match)
            if not suggestions:
                return None
            return SuggestionResult(items=suggestions, prefix=path_match)

        return None

    def apply_completion(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
        item: AutocompleteItem,
        prefix: str,
    ) -> ApplyResult:
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""
        before_prefix = current_line[:cursor_col - len(prefix)]
        after_cursor = current_line[cursor_col:]

        is_quoted_prefix = prefix.startswith('"') or prefix.startswith('@"')
        has_leading_quote_after = after_cursor.startswith('"')
        has_trailing_quote_in_item = item.value.endswith('"')
        adjusted_after = after_cursor[1:] if (
            is_quoted_prefix and has_trailing_quote_in_item and has_leading_quote_after
        ) else after_cursor

        text_before = current_line[:cursor_col]

        # Slash command name completion
        is_slash_cmd = (
            prefix.startswith("/") and
            before_prefix.strip() == "" and
            "/" not in prefix[1:]
        )
        if is_slash_cmd:
            new_line = f"{before_prefix}/{item.value} {adjusted_after}"
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return ApplyResult(
                lines=new_lines,
                cursor_line=cursor_line,
                cursor_col=len(before_prefix) + len(item.value) + 2,
            )

        # @ file attachment
        if prefix.startswith("@"):
            is_directory = item.label.endswith("/")
            suffix = "" if is_directory else " "
            new_line = before_prefix + item.value + suffix + adjusted_after
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            has_trailing = item.value.endswith('"')
            cursor_offset = len(item.value) - 1 if (is_directory and has_trailing) else len(item.value)
            return ApplyResult(
                lines=new_lines,
                cursor_line=cursor_line,
                cursor_col=len(before_prefix) + cursor_offset + len(suffix),
            )

        # Command argument or file path
        new_line = before_prefix + item.value + adjusted_after
        new_lines = list(lines)
        new_lines[cursor_line] = new_line
        is_directory = item.label.endswith("/")
        has_trailing = item.value.endswith('"')
        cursor_offset = len(item.value) - 1 if (is_directory and has_trailing) else len(item.value)
        return ApplyResult(
            lines=new_lines,
            cursor_line=cursor_line,
            cursor_col=len(before_prefix) + cursor_offset,
        )

    def _extract_at_prefix(self, text: str) -> str | None:
        quoted = _extract_quoted_prefix(text)
        if quoted and quoted.startswith('@"'):
            return quoted

        last_delim = _find_last_delimiter(text)
        token_start = 0 if last_delim == -1 else last_delim + 1
        if token_start < len(text) and text[token_start] == "@":
            return text[token_start:]
        return None

    def _extract_path_prefix(self, text: str, force: bool = False) -> str | None:
        quoted = _extract_quoted_prefix(text)
        if quoted:
            return quoted

        last_delim = _find_last_delimiter(text)
        path_prefix = text if last_delim == -1 else text[last_delim + 1:]

        if force:
            return path_prefix
        if "/" in path_prefix or path_prefix.startswith(".") or path_prefix.startswith("~/"):
            return path_prefix
        if path_prefix == "" and text.endswith(" "):
            return path_prefix
        return None

    def _expand_home(self, path: str) -> str:
        if path.startswith("~/"):
            return str(Path.home() / path[2:])
        if path == "~":
            return str(Path.home())
        return path

    def _get_file_suggestions(self, prefix: str) -> list[AutocompleteItem]:
        try:
            raw_prefix, is_at_prefix, is_quoted_prefix = _parse_path_prefix(prefix)
            expanded = self._expand_home(raw_prefix) if raw_prefix.startswith("~") else raw_prefix

            is_root = raw_prefix in ("", "./", "../", "~", "~/", "/") or (is_at_prefix and raw_prefix == "")

            if is_root:
                if raw_prefix.startswith("~") or expanded.startswith("/"):
                    search_dir = expanded if expanded else self.base_path
                else:
                    search_dir = os.path.join(self.base_path, expanded) if expanded else self.base_path
                search_prefix = ""
            elif raw_prefix.endswith("/"):
                if raw_prefix.startswith("~") or expanded.startswith("/"):
                    search_dir = expanded
                else:
                    search_dir = os.path.join(self.base_path, expanded)
                search_prefix = ""
            else:
                d = os.path.dirname(expanded)
                f = os.path.basename(expanded)
                if raw_prefix.startswith("~") or expanded.startswith("/"):
                    search_dir = d
                else:
                    search_dir = os.path.join(self.base_path, d) if d else self.base_path
                search_prefix = f

            entries = os.scandir(search_dir)
            suggestions: list[AutocompleteItem] = []

            for entry in entries:
                if not entry.name.lower().startswith(search_prefix.lower()):
                    continue
                try:
                    is_directory = entry.is_dir(follow_symlinks=True)
                except OSError:
                    is_directory = False

                display_prefix = raw_prefix
                name = entry.name

                if display_prefix.endswith("/"):
                    rel_path = display_prefix + name
                elif "/" in display_prefix:
                    if display_prefix.startswith("~/"):
                        home_rel = display_prefix[2:]
                        d2 = os.path.dirname(home_rel)
                        rel_path = f"~/{name}" if d2 == "." else f"~/{os.path.join(d2, name)}"
                    elif display_prefix.startswith("/"):
                        d2 = os.path.dirname(display_prefix)
                        rel_path = f"/{name}" if d2 == "/" else f"{d2}/{name}"
                    else:
                        rel_path = os.path.join(os.path.dirname(display_prefix), name)
                else:
                    rel_path = f"~/{name}" if display_prefix.startswith("~") else name

                path_val = f"{rel_path}/" if is_directory else rel_path
                value = _build_completion_value(path_val, is_directory, is_at_prefix, is_quoted_prefix)
                suggestions.append(AutocompleteItem(value=value, label=name + ("/" if is_directory else "")))

            # Sort dirs first, then alphabetically
            suggestions.sort(key=lambda x: (not x.label.endswith("/"), x.label.lower()))
            return suggestions
        except (OSError, PermissionError):
            return []

    def _score_entry(self, file_path: str, query: str, is_directory: bool) -> int:
        fname = os.path.basename(file_path)
        lower_f = fname.lower()
        lower_q = query.lower()
        score = 0
        if lower_f == lower_q:
            score = 100
        elif lower_f.startswith(lower_q):
            score = 80
        elif lower_q in lower_f:
            score = 50
        elif lower_q in file_path.lower():
            score = 30
        if is_directory and score > 0:
            score += 10
        return score

    def _get_fuzzy_file_suggestions(
        self,
        query: str,
        is_quoted_prefix: bool = False,
    ) -> list[AutocompleteItem]:
        if not self.fd_path:
            return []

        try:
            # Check for scoped query (path/query)
            slash_idx = query.rfind("/")
            if slash_idx != -1:
                display_base = query[:slash_idx + 1]
                sub_query = query[slash_idx + 1:]
                if display_base.startswith("~/"):
                    base_dir = self._expand_home(display_base)
                elif display_base.startswith("/"):
                    base_dir = display_base
                else:
                    base_dir = os.path.join(self.base_path, display_base)
                if not os.path.isdir(base_dir):
                    display_base = None
                    base_dir = self.base_path
                    sub_query = query
            else:
                display_base = None
                base_dir = self.base_path
                sub_query = query

            entries = _walk_directory_with_fd(base_dir, self.fd_path, sub_query)
            scored = [
                (e, self._score_entry(e[0], sub_query, e[1]))
                for e in entries
                if sub_query == "" or self._score_entry(e[0], sub_query, e[1]) > 0
            ]
            scored.sort(key=lambda x: -x[1])
            top = scored[:20]

            suggestions: list[AutocompleteItem] = []
            for (entry_path, is_dir), _ in top:
                path_no_slash = entry_path.rstrip("/")
                if display_base:
                    display_path = f"{display_base}{path_no_slash}"
                else:
                    display_path = path_no_slash
                entry_name = os.path.basename(path_no_slash)
                completion_path = f"{display_path}/" if is_dir else display_path
                value = _build_completion_value(completion_path, is_dir, True, is_quoted_prefix)
                suggestions.append(AutocompleteItem(
                    value=value,
                    label=entry_name + ("/" if is_dir else ""),
                    description=display_path,
                ))
            return suggestions
        except Exception:
            return []

    def get_force_file_suggestions(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
    ) -> SuggestionResult | None:
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""
        text_before = current_line[:cursor_col]

        # Don't trigger in slash command context
        stripped = text_before.strip()
        if stripped.startswith("/") and " " not in stripped:
            return None

        path_match = self._extract_path_prefix(text_before, force=True)
        if path_match is None:
            return None
        suggestions = self._get_file_suggestions(path_match)
        if not suggestions:
            return None
        return SuggestionResult(items=suggestions, prefix=path_match)

    def should_trigger_file_completion(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
    ) -> bool:
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""
        text_before = current_line[:cursor_col]
        stripped = text_before.strip()
        if stripped.startswith("/") and " " not in stripped:
            return False
        return True
