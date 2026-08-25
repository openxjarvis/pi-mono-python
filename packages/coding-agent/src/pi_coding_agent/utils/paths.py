"""
Path helpers — mirrors packages/coding-agent/src/utils/paths.ts
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

UNICODE_SPACES = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")


@dataclass
class PathInputOptions:
    trim: bool = False
    expand_tilde: bool = True
    home_dir: str | None = None
    strip_at_prefix: bool = False
    normalize_unicode_spaces: bool = False


def canonicalize_path(path: str) -> str:
    try:
        return os.path.realpath(path)
    except OSError:
        return path


def get_file_revision(path: str) -> str | None:
    try:
        stats = os.stat(path)
        return f"{stats.st_dev}:{stats.st_ino}:{stats.st_size}:{stats.st_mtime_ns}:{stats.st_ctime_ns}"
    except OSError:
        return None


def is_local_path(value: str) -> bool:
    trimmed = value.strip()
    prefixes = ("npm:", "git:", "github:", "http:", "https:", "ssh:")
    return not trimmed.startswith(prefixes)


def normalize_windows_shell_path(file_path: str) -> str:
    if not file_path.startswith("/") or file_path.startswith("//") or "\\" in file_path:
        return file_path
    match = re.match(r"^/(?:mnt/|cygdrive/)?([a-z])(?:/(.*))?$", file_path, re.IGNORECASE)
    if not match:
        return file_path
    suffix = (match.group(2) or "").replace("/", "\\")
    return f"{match.group(1).upper()}:\\{suffix}"


def normalize_path(input_path: str, options: PathInputOptions | None = None) -> str:
    opts = options or PathInputOptions()
    normalized = input_path.strip() if opts.trim else input_path
    if opts.normalize_unicode_spaces:
        normalized = UNICODE_SPACES.sub(" ", normalized)
    if opts.strip_at_prefix and normalized.startswith("@"):
        normalized = normalized[1:]
    if os.name == "nt":
        normalized = normalize_windows_shell_path(normalized)
    if opts.expand_tilde:
        home = opts.home_dir or str(Path.home())
        if normalized == "~":
            return home
        if normalized.startswith("~/") or (os.name == "nt" and normalized.startswith("~\\")):
            return os.path.join(home, normalized[2:])
    if normalized.startswith("file://"):
        parsed = urlparse(normalized)
        return unquote(parsed.path)
    return normalized


def resolve_path(input_path: str, base_dir: str | None = None, options: PathInputOptions | None = None) -> str:
    normalized = normalize_path(input_path, options)
    base = normalize_path(base_dir or os.getcwd())
    return os.path.abspath(normalized if os.path.isabs(normalized) else os.path.join(base, normalized))


def get_cwd_relative_path(file_path: str, cwd: str) -> str | None:
    resolved_cwd = resolve_path(cwd)
    resolved_path = resolve_path(file_path, resolved_cwd)
    try:
        relative = os.path.relpath(resolved_path, resolved_cwd)
    except ValueError:
        return None
    if relative == "..":
        return None
    if relative.startswith(f"..{os.sep}") or os.path.isabs(relative):
        return None
    return relative or "."


def format_path_relative_to_cwd_or_absolute(file_path: str, cwd: str) -> str:
    absolute = resolve_path(file_path, cwd)
    return (get_cwd_relative_path(absolute, cwd) or absolute).replace(os.sep, "/")


def mark_path_ignored_by_cloud_sync(path: str) -> None:
    return None
