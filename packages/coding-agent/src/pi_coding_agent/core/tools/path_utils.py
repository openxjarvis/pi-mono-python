"""
Path resolution utilities — mirrors packages/coding-agent/src/core/tools/path-utils.ts
"""
from __future__ import annotations

import os


def resolve_to_cwd(path: str, cwd: str) -> str:
    """Resolve a path relative to cwd, or return absolute as-is."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(cwd, path))


def resolve_read_path(path: str, cwd: str) -> str:
    """Resolve a read path — same as resolve_to_cwd."""
    return resolve_to_cwd(path, cwd)
