"""
CLI entry point — mirrors packages/coding-agent/src/cli.ts
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

import typer
from rich.console import Console

from .core.sdk import AgentSessionOptions, create_agent_session
from .modes.print_mode import run_print_mode

app = typer.Typer(
    name="pi",
    help="Pi — Python coding agent",
    no_args_is_help=True,
)

console = Console()


@app.command()
def prompt_cmd(
    prompt: str = typer.Argument(..., help="Prompt to send to the agent"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model ID to use"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Provider name"),
    cwd: Optional[str] = typer.Option(None, "--cwd", help="Working directory"),
    thinking: str = typer.Option("off", "--thinking", help="Thinking level: off/minimal/low/medium/high"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
    show_thinking: bool = typer.Option(False, "--show-thinking", help="Show thinking blocks"),
) -> None:
    """Send a prompt to the coding agent."""
    opts = AgentSessionOptions(
        cwd=cwd or os.getcwd(),
        model_id=model,
        provider=provider,
        thinking_level=thinking,
    )

    session = create_agent_session(opts)

    exit_code = asyncio.run(run_print_mode(
        session,
        prompt,
        show_thinking=show_thinking,
        json_output=json_out,
    ))
    sys.exit(exit_code)


@app.command()
def list_models() -> None:
    """List available models."""
    from pi_ai import get_models, get_providers
    from rich.table import Table

    table = Table(title="Available Models")
    table.add_column("Provider")
    table.add_column("Model ID")
    table.add_column("API")
    table.add_column("Context")
    table.add_column("Reasoning")

    for model in get_models():
        table.add_row(
            model.provider,
            model.id,
            model.api,
            f"{model.context_window // 1000}K",
            "✓" if model.reasoning else "",
        )

    console.print(table)


@app.command()
def list_sessions() -> None:
    """List saved sessions."""
    from rich.table import Table
    import datetime
    from .core.session_manager import SessionManager

    manager = SessionManager()
    sessions = manager.list_sessions()

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Sessions")
    table.add_column("ID")
    table.add_column("Label")
    table.add_column("Messages")
    table.add_column("Updated")

    for s in sessions[:20]:  # Show last 20
        updated = datetime.datetime.fromtimestamp(s.updated_at / 1000).strftime("%Y-%m-%d %H:%M")
        table.add_row(
            s.session_id[:8] + "...",
            s.label or "(no label)",
            str(s.entry_count),
            updated,
        )

    console.print(table)


def main() -> None:
    """Main CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
