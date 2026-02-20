"""
TUI runner for interactive mode using pi_tui.

Wires up a ProcessTerminal + TUI + Editor to give a full interactive
experience aligned with the TypeScript coding agent interactive mode.
Falls back to a readline loop if pi_tui cannot start (e.g. not a TTY).

Slash commands: /exit /clear /help /model /compact /thinking /session /tools
Footer:         model | thinking: off | ctx: 12% | tokens: 8k/64k
Ctrl+P:         cycle model
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
    history_text = Text("", padding_x=1, padding_y=0)
    stream_text = Text("", padding_x=1, padding_y=0)
    footer_text = Text("", padding_x=1, padding_y=0)
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

    # ── Footer ────────────────────────────────────────────────────────────────
    tui.add_child(footer_text)

    def _fmt_tokens(n: int) -> str:
        if n >= 1000:
            return f"{n // 1000}k"
        return str(n)

    def update_footer() -> None:
        """Refresh footer: model | thinking: off | ctx: 12% | tokens: 8k/64k"""
        model = session.model
        model_str = model.id if model else "no model"
        thinking = getattr(session, "thinking_level", "off") or "off"
        parts = [model_str, f"thinking: {thinking}"]
        ctx = session.get_context_usage()
        if ctx and ctx.get("percent") is not None:
            pct = ctx["percent"]
            tkn = _fmt_tokens(ctx.get("tokens", 0))
            cw = _fmt_tokens(ctx.get("contextWindow", 0))
            parts.append(f"ctx: {pct:.0f}% ({tkn}/{cw})")
        footer_text.set_text(dim("  " + " | ".join(parts)))
        footer_text.invalidate()

    update_footer()

    # ── Editor ───────────────────────────────────────────────────────────────
    editor = Editor(tui, editor_theme)
    slash_commands = [
        SlashCommand(name="exit",     description="Exit the agent"),
        SlashCommand(name="clear",    description="Clear conversation history"),
        SlashCommand(name="help",     description="Show help"),
        SlashCommand(name="model",    description="List or switch models"),
        SlashCommand(name="compact",  description="Compact conversation context"),
        SlashCommand(name="thinking", description="Cycle thinking level (off/minimal/low/medium/high)"),
        SlashCommand(name="session",  description="Show session statistics"),
        SlashCommand(name="tools",    description="List active tools"),
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

        # ── Slash commands ────────────────────────────────────────────────────
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

        if stripped == "/help":
            lines = [
                bold("Available commands:"),
                f"  {cyan('/exit')}     — Exit the agent",
                f"  {cyan('/clear')}    — Clear conversation history",
                f"  {cyan('/model')}    — List available models / switch model",
                f"  {cyan('/compact')}  — Compact context to free tokens",
                f"  {cyan('/thinking')} — Cycle thinking level",
                f"  {cyan('/session')}  — Show session statistics",
                f"  {cyan('/tools')}    — List active tools",
                f"  {cyan('Ctrl+P')}    — Cycle to next model",
            ]
            append_history("\n".join(lines))
            tui.request_render()
            return

        if stripped == "/tools":
            names = session.get_active_tool_names()
            if names:
                append_history(bold("Active tools:") + "\n" + "\n".join(f"  - {n}" for n in names))
            else:
                append_history(dim("No active tools."))
            tui.request_render()
            return

        if stripped == "/session":
            stats = session.get_session_stats()
            lines = [
                bold("Session stats:"),
                f"  Session ID:   {stats.get('sessionId', '?')}",
                f"  User msgs:    {stats.get('userMessages', 0)}",
                f"  Asst msgs:    {stats.get('assistantMessages', 0)}",
                f"  Tool calls:   {stats.get('toolCalls', 0)}",
                f"  Total tokens: {_fmt_tokens(stats.get('tokens', {}).get('total', 0))}",
                f"  Cost:         ${stats.get('cost', 0.0):.4f}",
            ]
            append_history("\n".join(lines))
            tui.request_render()
            return

        if stripped == "/thinking":
            new_level = session.cycle_thinking_level()
            if new_level:
                append_history(f"{cyan('Thinking level:')} {new_level}")
            else:
                append_history(dim("Thinking not supported by current model."))
            update_footer()
            tui.request_render()
            return

        if stripped == "/compact":
            append_history(dim("Compacting context..."))
            tui.request_render()
            try:
                summary = await session.compact()
                if summary:
                    short = summary[:400] + "..." if len(summary) > 400 else summary
                    append_history(f"{green('Compaction complete.')}\n{dim(short)}")
                else:
                    append_history(dim("Compaction complete (nothing to summarize)."))
            except Exception as exc:
                append_history(f"{red('Compaction error:')} {exc}")
            update_footer()
            tui.request_render()
            return

        if stripped == "/model" or stripped.startswith("/model "):
            await _handle_model_command(stripped, session,
                                        append_history, update_footer, tui,
                                        cyan, dim, red, bold, green)
            return

        # ── Busy guard ────────────────────────────────────────────────────────
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
              auto_retry_start/end → show retry indicator
              auto_compaction_start/end → show compaction indicator
            """
            nonlocal rendered_response
            try:
                if isinstance(event, dict):
                    etype = event.get("type", "") or ""
                else:
                    etype = getattr(event, "type", None) or ""
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

                elif etype == "auto_retry_start":
                    attempt = event.get("attempt", 0) if isinstance(event, dict) else getattr(event, "attempt", 0)
                    max_a = event.get("maxAttempts", 3) if isinstance(event, dict) else getattr(event, "maxAttempts", 3)
                    delay = event.get("delayMs", 0) if isinstance(event, dict) else getattr(event, "delayMs", 0)
                    err = event.get("errorMessage", "") if isinstance(event, dict) else getattr(event, "errorMessage", "")
                    append_history(f"{yellow(f'Retry {attempt}/{max_a}:')} {dim(str(err))} (wait {delay // 1000}s)")
                    tui.request_render()

                elif etype == "auto_retry_end":
                    success = event.get("success", True) if isinstance(event, dict) else getattr(event, "success", True)
                    if not success:
                        err = event.get("finalError", "") if isinstance(event, dict) else getattr(event, "finalError", "")
                        append_history(f"{red('Retry failed:')} {err}")
                        tui.request_render()

                elif etype == "auto_compaction_start":
                    append_history(dim("Auto-compacting context..."))
                    tui.request_render()

                elif etype == "auto_compaction_end":
                    result = event.get("result") if isinstance(event, dict) else getattr(event, "result", None)
                    err_msg = event.get("errorMessage") if isinstance(event, dict) else getattr(event, "errorMessage", None)
                    if err_msg:
                        append_history(f"{red('Compaction error:')} {err_msg}")
                    else:
                        append_history(dim("Context compacted."))
                    update_footer()
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
            if hasattr(session, "abort"):
                asyncio.ensure_future(session.abort())
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
            update_footer()
            trace("handle_submit: busy=false")
            tui.request_render()

    # ── Capture main loop for cross-thread callbacks ──────────────────────────
    main_loop = asyncio.get_running_loop()

    async def _cycle_model_async() -> None:
        """Ctrl+P handler: cycle to next available model."""
        try:
            result = await session.cycle_model("forward")
            if result:
                m = result["model"]
                append_history(f"{cyan('Model:')} {m.id} ({m.provider})")
            else:
                append_history(dim("Only one model available."))
            update_footer()
            tui.request_render()
        except Exception as exc:
            append_history(f"{red('Model switch failed:')} {exc}")
            tui.request_render()

    def on_submit_sync(text: str) -> None:
        asyncio.run_coroutine_threadsafe(handle_submit(text), main_loop)

    def on_keydown_sync(key: str) -> None:
        """Handle special key sequences (called from terminal thread)."""
        # Ctrl+P = '\x10'
        if key == "\x10":
            asyncio.run_coroutine_threadsafe(_cycle_model_async(), main_loop)

    editor.on_submit = on_submit_sync

    # Register keydown handler if supported by the editor
    if hasattr(editor, "on_keydown"):
        editor.on_keydown = on_keydown_sync

    # ── Start TUI ─────────────────────────────────────────────────────────────
    trace("tui: start")
    tui.start()

    if initial_messages:
        trace(f"tui: initial_messages={initial_messages!r}")
        for msg in initial_messages:
            await handle_submit(msg)
            # Yield to event loop so pending render ticks can fire before the
            # next message is processed.
            await asyncio.sleep(0)

    # ── Main loop ─────────────────────────────────────────────────────────────
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


async def _handle_model_command(
    stripped: str,
    session: "AgentSession",
    append_history,
    update_footer,
    tui,
    cyan, dim, red, bold, green,
) -> None:
    """Handle /model and /model <id> commands."""
    parts = stripped.split(None, 1)
    model_arg = parts[1].strip() if len(parts) > 1 else None

    if model_arg:
        # /model <id> — switch to named model
        available = await session.model_registry.get_available()
        target = next(
            (m for m in available
             if m.id == model_arg or m.id.lower() == model_arg.lower()),
            None,
        )
        if target is None:
            append_history(f"{red('Unknown model:')} {model_arg}")
            tui.request_render()
            return
        try:
            await session.set_model(target)
            append_history(f"{cyan('Switched to model:')} {target.id} ({target.provider})")
            update_footer()
        except Exception as exc:
            append_history(f"{red('Model switch failed:')} {exc}")
        tui.request_render()
        return

    # /model — list available models
    available = await session.model_registry.get_available()
    current = session.model
    if not available:
        append_history(dim("No models available."))
        tui.request_render()
        return
    lines = [bold("Available models:")]
    for m in available:
        marker = cyan("→") if (current and m.id == current.id) else " "
        lines.append(f"  {marker} {m.id} ({m.provider})")
    lines.append(dim("Use /model <id> to switch."))
    append_history("\n".join(lines))
    tui.request_render()
