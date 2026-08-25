"""
Interactive mode entry — mirrors modes/interactive/interactive-mode.ts exports
used by the CLI (`run_interactive_mode`).
"""
from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

from .interactive_mode import (
    InteractiveMode,
    InteractiveModeOptions,
    create_interactive_tui,
    create_interactive_tui_reference,
    format_resume_command,
    run_interactive_mode,
)

if TYPE_CHECKING:
    from pi_coding_agent.core.agent_session import AgentSession

__all__ = [
    "InteractiveMode",
    "InteractiveModeOptions",
    "create_interactive_tui",
    "create_interactive_tui_reference",
    "format_resume_command",
    "run_interactive_mode",
]


async def _run_readline_fallback(session: "AgentSession", initial_messages: list[str] | None = None) -> None:
    """Fallback readline-based interactive loop when no TTY is available."""
    from rich.console import Console

    console = Console()
    console.print("[bold green]Pi Coding Agent[/bold green] (interactive mode)")
    console.print("Type your message and press Enter. Ctrl+C to exit.\n")

    if initial_messages:
        for msg in initial_messages:
            console.print(f"[dim]> {msg}[/dim]")
            await _send_and_wait(session, msg, console)

    loop = asyncio.get_event_loop()
    while True:
        try:
            console.print("\n[bold cyan]>[/bold cyan] ", end="")
            user_input = await loop.run_in_executor(None, sys.stdin.readline)
            if not user_input:
                break
            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                break
            await _send_and_wait(session, user_input, console)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting...[/dim]")
            break


async def _send_and_wait(session: "AgentSession", message: str, console: Any) -> None:
    done = asyncio.Event()

    def on_event(event: dict[str, Any]) -> None:
        event_type = event.get("type", "") if isinstance(event, dict) else getattr(event, "type", "")
        if event_type == "text_delta":
            text = event.get("text", "") if isinstance(event, dict) else getattr(event, "text", "")
            console.print(text, end="", highlight=False)
        elif event_type in ("agent_end", "turn_end"):
            done.set()

    unsubscribe = session.subscribe(on_event) if hasattr(session, "subscribe") else None
    try:
        await session.prompt(message)
        if not done.is_set():
            done.set()
        await done.wait()
        console.print()
    finally:
        if unsubscribe:
            unsubscribe()
