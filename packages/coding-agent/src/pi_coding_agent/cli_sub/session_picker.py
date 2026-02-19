"""
Session selector for --resume flag.

Mirrors packages/coding-agent/src/cli/session-picker.ts
"""
from __future__ import annotations

import sys
from typing import Any, Callable


SessionsLoader = Callable[..., Any]


async def select_session(
    current_sessions_loader: SessionsLoader,
    all_sessions_loader: SessionsLoader,
) -> str | None:
    """
    Display a session selector and return selected session path, or None if cancelled.

    Falls back to a simple terminal list when Textual is not available.
    """
    try:
        return await _select_session_textual(current_sessions_loader, all_sessions_loader)
    except ImportError:
        return await _select_session_readline(current_sessions_loader, all_sessions_loader)


async def _select_session_textual(
    current_sessions_loader: SessionsLoader,
    all_sessions_loader: SessionsLoader,
) -> str | None:
    """Use Textual TUI for session selection (raises ImportError if Textual unavailable)."""
    import importlib
    importlib.import_module("textual")

    # Placeholder: full Textual integration mirrors interactive-mode components
    # Falls back to readline for now
    raise ImportError("Textual session picker not yet implemented; falling back")


async def _select_session_readline(
    current_sessions_loader: SessionsLoader,
    all_sessions_loader: SessionsLoader,
) -> str | None:
    """Simple readline-based session selector."""
    import asyncio

    sessions = await current_sessions_loader()
    if not sessions:
        sessions = await all_sessions_loader()

    if not sessions:
        print("No sessions found.", file=sys.stderr)
        return None

    print("\nAvailable sessions:")
    for i, session in enumerate(sessions):
        path = session.get("path", "") if isinstance(session, dict) else getattr(session, "path", "")
        name = session.get("name", path) if isinstance(session, dict) else getattr(session, "name", path)
        print(f"  {i + 1}. {name} ({path})")

    print("\nEnter number to select, or press Enter to cancel: ", end="", flush=True)
    loop = asyncio.get_event_loop()
    try:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        line = line.strip()
        if not line:
            return None
        idx = int(line) - 1
        if 0 <= idx < len(sessions):
            session = sessions[idx]
            return session.get("path", "") if isinstance(session, dict) else getattr(session, "path", "")
    except (ValueError, IndexError):
        pass

    return None
