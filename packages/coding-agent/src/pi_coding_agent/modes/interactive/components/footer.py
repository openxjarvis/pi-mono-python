"""Footer — mirrors footer.ts"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .component import Component


def format_tokens(count: int) -> str:
    if count < 1000:
        return str(count)
    if count < 10_000:
        return f"{count / 1000:.1f}k"
    if count < 1_000_000:
        return f"{round(count / 1000)}k"
    if count < 10_000_000:
        return f"{count / 1_000_000:.1f}M"
    return f"{round(count / 1_000_000)}M"


def format_cwd_for_footer(cwd: str, home: str | None = None) -> str:
    home_dir = home or str(Path.home())
    resolved_cwd = os.path.abspath(cwd)
    resolved_home = os.path.abspath(home_dir)
    try:
        relative = os.path.relpath(resolved_cwd, resolved_home)
    except ValueError:
        return cwd
    if relative == "..":
        return cwd
    if relative.startswith(f"..{os.sep}") or os.path.isabs(relative):
        return cwd
    return "~" if relative in ("", ".") else f"~{os.sep}{relative}"


class FooterComponent(Component):
    name = "footer"

    def __init__(self, session: Any | None = None, footer_data: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.session = session
        self.footer_data = footer_data
        self.auto_compact_enabled = True

    def set_session(self, session: Any) -> None:
        self.session = session
        self.invalidate()

    def set_auto_compact_enabled(self, enabled: bool) -> None:
        self.auto_compact_enabled = enabled

    def dispose(self) -> None:
        return None

    def _render_body(self, width: int) -> str:
        session = self.session
        cwd = getattr(session, "cwd", os.getcwd()) if session is not None else os.getcwd()
        model = getattr(session, "model", None) if session is not None else None
        model_id = getattr(model, "id", None) or "no model"
        thinking = getattr(session, "thinking_level", "off") if session is not None else "off"
        parts = [format_cwd_for_footer(cwd), str(model_id), f"thinking: {thinking}"]
        if session is not None and hasattr(session, "get_context_usage"):
            ctx = session.get_context_usage()
            if ctx and ctx.get("percent") is not None:
                parts.append(f"ctx: {ctx['percent']:.0f}%")
        name = getattr(session, "session_name", None) if session is not None else None
        if name:
            parts.append(str(name))
        return " | ".join(parts)
