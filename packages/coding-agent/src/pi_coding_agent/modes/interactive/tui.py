"""
TUI runner for interactive mode using pi_tui.

Wires up a ProcessTerminal + TUI + Editor to give a full interactive
experience aligned with the TypeScript coding agent interactive mode.
Falls back to a readline loop if pi_tui cannot start (e.g. not a TTY).
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pi_coding_agent.core.agent_session import AgentSession


async def run_tui(session: "AgentSession", initial_messages: list[str] | None = None) -> None:
    """
    Run the interactive TUI using pi_tui.

    Falls back to readline if pi_tui is unavailable or not in a TTY.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        from .mode import _run_readline_fallback
        await _run_readline_fallback(session, initial_messages)
        return

    try:
        await _run_pi_tui(session, initial_messages)
    except Exception as exc:
        import traceback
        from rich.console import Console
        console = Console()
        console.print(f"[yellow]TUI error ({exc}), falling back to readline mode.[/yellow]")
        console.print(traceback.format_exc(), style="dim")
        from .mode import _run_readline_fallback
        await _run_readline_fallback(session, initial_messages)


async def _run_pi_tui(session: "AgentSession", initial_messages: list[str] | None) -> None:
    """Set up and run the pi_tui interactive loop."""
    from pi_tui import (
        TUI,
        Editor,
        EditorTheme,
        MarkdownTheme,
        Text,
        Spacer,
        SelectListTheme,
        ProcessTerminal,
    )
    from pi_tui import CombinedAutocompleteProvider, SlashCommand
    from pi_ai.types import TextContent, UserMessage

    trace_path = os.environ.get("PI_INTERACTIVE_TRACE_LOG", "").strip()

    def trace(msg: str) -> None:
        if not trace_path:
            return
        try:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    terminal = ProcessTerminal()
    tui = TUI(terminal)

    # ── ANSI theme helpers ────────────────────────────────────────────────────
    def dim(s: str) -> str:
        return f"\x1b[2m{s}\x1b[22m"

    def bold(s: str) -> str:
        return f"\x1b[1m{s}\x1b[22m"

    def cyan(s: str) -> str:
        return f"\x1b[36m{s}\x1b[39m"

    def green(s: str) -> str:
        return f"\x1b[32m{s}\x1b[39m"

    def yellow(s: str) -> str:
        return f"\x1b[33m{s}\x1b[39m"

    def red(s: str) -> str:
        return f"\x1b[31m{s}\x1b[39m"

    select_theme = SelectListTheme(
        selected_text=cyan,
        description=dim,
        scroll_info=dim,
        no_match=dim,
    )

    editor_theme = EditorTheme(
        border_color=dim,
        select_list=select_theme,
    )

    markdown_theme = MarkdownTheme(
        heading=bold,
        bold=bold,
        code=lambda s: f"\x1b[33m{s}\x1b[39m",
        code_block=lambda s: f"\x1b[33m{s}\x1b[39m",
        code_block_border=dim,
        list_bullet=cyan,
    )

    # ── Output area ──────────────────────────────────────────────────────────
    # history_text: all completed exchanges
    # stream_text:  current streaming assistant response (updated in-place)
    history_text = Text("", padding_x=1, padding_y=0)
    stream_text = Text("", padding_x=1, padding_y=0)
    tui.add_child(history_text)
    tui.add_child(stream_text)
    tui.add_child(Spacer(1))

    def append_history(line: str) -> None:
        """Append a completed line to history."""
        trace(f"append_history: {line[:120]!r}")
        current = history_text._text
        history_text.set_text((current + "\n" + line).lstrip("\n"))
        history_text.invalidate()

    def assistant_text_from_message(message: object) -> str:
        """
        Extract full assistant text from an assistant message snapshot.
        Mirrors TS behavior where UI updates from full message state, not only deltas.
        """
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if getattr(item, "type", None) == "text":
                text = getattr(item, "text", "")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts)

    def set_stream(text: str) -> None:
        """Update the current streaming response line."""
        trace(f"set_stream: {text[:120]!r}")
        stream_text.set_text(text)
        stream_text.invalidate()
        tui.request_render()

    # ── Editor ───────────────────────────────────────────────────────────────
    editor = Editor(tui, editor_theme)
    slash_commands = [
        SlashCommand(name="exit", description="Exit the agent"),
        SlashCommand(name="clear", description="Clear conversation history"),
        SlashCommand(name="help", description="Show help"),
    ]
    autocomplete = CombinedAutocompleteProvider(commands=slash_commands)
    editor.set_autocomplete_provider(autocomplete)
    tui.add_child(editor)
    tui.set_focus(editor)

    # ── Submit handler ────────────────────────────────────────────────────────
    is_busy = False

    async def handle_submit(text: str) -> None:
        nonlocal is_busy
        trace(f"handle_submit: raw={text!r}")
        stripped = text.strip()
        if not stripped:
            trace("handle_submit: empty, return")
            return

        # Slash commands
        if stripped in ("/exit", "exit", "quit"):
            trace("handle_submit: exit command")
            tui.stop()
            return
        if stripped == "/clear":
            trace("handle_submit: clear command")
            history_text.set_text("")
            history_text.invalidate()
            set_stream("")
            return

        if is_busy:
            trace("handle_submit: busy, queue follow-up")
            queued = UserMessage(
                role="user",
                content=[TextContent(type="text", text=stripped)],
                timestamp=int(time.time() * 1000),
            )
            await session.follow_up(queued)
            append_history(f"{dim('Queued follow-up:')} {stripped}")
            tui.request_render()
            return

        is_busy = True
        trace("handle_submit: busy=true")

        # Show user message in history
        append_history(f"{bold('You:')} {stripped}")
        tui.request_render()

        # Collect streaming text and done signal
        collected: list[str] = []
        rendered_response = ""
        pending_tools: dict[str, str] = {}
        def on_event(event) -> None:
            """
            Handle AgentEvent (Pydantic models) aligned with TS lifecycle:
              message_start → initialize assistant streaming row
              message_update → render from full assistant snapshot when possible
              message_end → fallback finalization (for non-delta providers)
              agent_end → complete request
              turn_end → surface error message
            """
            nonlocal rendered_response
            try:
                etype = event.type
                trace(f"on_event: {etype}")

                if etype == "message_start":
                    msg = getattr(event, "message", None)
                    if getattr(msg, "role", None) == "assistant":
                        if not stream_text._text:
                            set_stream(f"{bold('Assistant:')} ")

                elif etype == "message_update":
                    msg = getattr(event, "message", None)
                    if getattr(msg, "role", None) != "assistant":
                        return

                    # TS parity: prefer full message snapshot over raw deltas.
                    snapshot_text = assistant_text_from_message(msg)
                    if snapshot_text:
                        if snapshot_text != rendered_response:
                            trace(f"on_event: snapshot_text len={len(snapshot_text)}")
                            rendered_response = snapshot_text
                            set_stream(f"{bold('Assistant:')} {snapshot_text}")
                        return

                    ae = getattr(event, "assistant_message_event", None)
                    if getattr(ae, "type", None) == "text_delta":
                        collected.append(getattr(ae, "delta", ""))
                        response_so_far = "".join(collected)
                        if response_so_far != rendered_response:
                            trace(f"on_event: delta_text len={len(response_so_far)}")
                            rendered_response = response_so_far
                            set_stream(f"{bold('Assistant:')} {response_so_far}")

                elif etype == "message_end":
                    msg = getattr(event, "message", None)
                    if getattr(msg, "role", None) == "assistant":
                        final_text = assistant_text_from_message(msg)
                        if final_text and final_text != rendered_response:
                            trace(f"on_event: final_text len={len(final_text)}")
                            rendered_response = final_text
                            set_stream(f"{bold('Assistant:')} {final_text}")

                elif etype == "agent_end":
                    # Fallback path: if agent crashed before normal assistant message
                    # lifecycle completed, surface the terminal agent_end payload.
                    if not rendered_response:
                        terminal_messages = getattr(event, "messages", None)
                        if isinstance(terminal_messages, list):
                            for msg in terminal_messages:
                                if getattr(msg, "role", None) != "assistant":
                                    continue
                                err = getattr(msg, "error_message", None)
                                if isinstance(err, str) and err.strip():
                                    trace(f"on_event: agent_end error={err!r}")
                                    set_stream(f"{red('Error:')} {err}")
                                    rendered_response = err
                                    break
                                fallback_text = assistant_text_from_message(msg)
                                if fallback_text:
                                    trace(f"on_event: agent_end fallback_text len={len(fallback_text)}")
                                    set_stream(f"{bold('Assistant:')} {fallback_text}")
                                    rendered_response = fallback_text
                                    break

                elif etype == "turn_end":
                    msg = getattr(event, "message", None)
                    err = getattr(msg, "error_message", None)
                    if err:
                        trace(f"on_event: turn_end error={err!r}")
                        set_stream(f"{red('Error:')} {err}")

                elif etype == "tool_execution_start":
                    tool_call_id = getattr(event, "tool_call_id", "")
                    tool_name = getattr(event, "tool_name", "tool")
                    pending_tools[tool_call_id] = tool_name
                    append_history(f"{yellow('Tool start:')} {tool_name}")
                    tui.request_render()

                elif etype == "tool_execution_end":
                    tool_call_id = getattr(event, "tool_call_id", "")
                    tool_name = pending_tools.pop(tool_call_id, getattr(event, "tool_name", "tool"))
                    is_error = bool(getattr(event, "is_error", False))
                    result = getattr(event, "result", None)
                    status = red("error") if is_error else green("ok")
                    line = f"{yellow('Tool end:')} {tool_name} ({status})"
                    content = getattr(result, "content", None)
                    if isinstance(content, list):
                        text_parts: list[str] = []
                        for c in content:
                            if getattr(c, "type", None) == "text":
                                t = getattr(c, "text", "")
                                if isinstance(t, str) and t:
                                    text_parts.append(t)
                        if text_parts:
                            snippet = " ".join(" ".join(text_parts).split())
                            if len(snippet) > 160:
                                snippet = snippet[:157] + "..."
                            line += f" - {snippet}"
                    append_history(line)
                    tui.request_render()
            except Exception as exc:
                trace(f"on_event: exception={exc!r}")

        unsub = session.subscribe(on_event)
        try:
            trace("handle_submit: before session.prompt")
            await asyncio.wait_for(session.prompt(stripped, source="interactive"), timeout=300.0)
            trace("handle_submit: after session.prompt")
        except asyncio.TimeoutError:
            trace("handle_submit: timeout; aborting active turn")
            abort = getattr(session, "abort", None)
            if callable(abort):
                abort()
            set_stream(f"{yellow('Response timed out and was aborted')}")
        except Exception as exc:
            trace(f"handle_submit: exception={exc!r}")
            set_stream(f"{red('Error:')} {exc}")
        finally:
            trace("handle_submit: finally begin")
            unsub()
            # Move the completed response from stream_text to history
            final = stream_text._text
            if final:
                append_history(final)
                set_stream("")
            is_busy = False
            trace("handle_submit: busy=false")
            tui.request_render()

    # Capture the main event loop BEFORE TUI starts threads.
    # on_submit_sync is called from the terminal read thread, so we must
    # use run_coroutine_threadsafe instead of ensure_future.
    main_loop = asyncio.get_running_loop()

    def on_submit_sync(text: str) -> None:
        asyncio.run_coroutine_threadsafe(handle_submit(text), main_loop)

    editor.on_submit = on_submit_sync

    # ── Start TUI (also starts the terminal read thread) ─────────────────────
    trace("tui: start")
    tui.start()

    if initial_messages:
        trace(f"tui: initial_messages={initial_messages!r}")
        for msg in initial_messages:
            await handle_submit(msg)

    # ── Main loop ─────────────────────────────────────────────────────────────
    # Poll until the TUI is stopped (by /exit, Ctrl+C, or tui.stop())
    try:
        while not tui.stopped:
            await asyncio.sleep(0.05)
    except (KeyboardInterrupt, asyncio.CancelledError):
        trace("tui: keyboard/cancelled")
        pass
    finally:
        if not tui.stopped:
            trace("tui: stop in finally")
            tui.stop()
