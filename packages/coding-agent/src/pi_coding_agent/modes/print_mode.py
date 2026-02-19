"""
Print mode — mirrors packages/coding-agent/src/modes/print-mode.ts

Non-interactive (single-shot) mode: sends prompts to agent, outputs result.
Used for `pi -p "prompt"` (text) and `pi --mode json "prompt"` (JSON event stream).
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from pi_agent.types import (
    AgentEvent,
    AgentEventAgentEnd,
    AgentEventMessageEnd,
    AgentEventMessageStart,
    AgentEventToolEnd,
    AgentEventToolStart,
)
from pi_ai.types import AssistantMessage, ImageContent, TextContent, ToolCall, ToolResultMessage

from ..core.agent_session import AgentSession


class PrintModeOptions:
    """Options for print mode."""
    def __init__(
        self,
        mode: str = "text",              # "text" | "json"
        messages: list[str] | None = None,
        initial_message: str | None = None,
        initial_images: list[Any] | None = None,  # list[ImageContent]
    ) -> None:
        self.mode = mode
        self.messages = messages or []
        self.initial_message = initial_message
        self.initial_images = initial_images or []


async def run_print_mode(
    session: AgentSession,
    prompt: str | None = None,
    show_thinking: bool = False,
    json_output: bool = False,
    # New parity params
    options: PrintModeOptions | None = None,
) -> int:
    """
    Run in print (single-shot) mode.
    Mirrors runPrintMode() in TypeScript.

    Supports:
    - Multiple messages (messages[] array)
    - Initial images
    - JSON mode: outputs session header + all events as newline-delimited JSON
    - Error exit: returns 1 if stop_reason == "error" or "aborted"
    - Explicit stdout flush

    Returns exit code (0 = success, 1 = error).
    """
    # Build options object (backwards-compat with old positional API)
    if options is None:
        options = PrintModeOptions(
            mode="json" if json_output else "text",
            initial_message=prompt,
        )

    mode = options.mode

    # JSON mode: output session header first
    if mode == "json":
        try:
            sm = session.session_manager if hasattr(session, "session_manager") else None
            header = sm.get_header() if sm and hasattr(sm, "get_header") else None
            if header:
                print(json.dumps(header), flush=True)
        except Exception:
            pass

    # Subscribe to events
    def on_event(event: AgentEvent) -> None:
        if mode == "json":
            try:
                obj = _event_to_dict(event)
                print(json.dumps(obj, default=str), flush=True)
            except Exception:
                pass
        else:
            _handle_print_event(event, show_thinking=show_thinking)

    unsub = session.subscribe(on_event)

    try:
        # Send initial message (with optional images)
        if options.initial_message:
            if options.initial_images:
                await session.prompt(options.initial_message, images=options.initial_images)
            else:
                await session.prompt(options.initial_message)

        # Send remaining messages in sequence
        for msg in options.messages:
            await session.prompt(msg)

        # In text mode, output final assistant response
        if mode == "text":
            state = session.state
            msgs = state.messages if hasattr(state, "messages") else []
            last = msgs[-1] if msgs else None

            if last and isinstance(last, AssistantMessage):
                # Check for error/aborted
                stop = getattr(last, "stop_reason", None) or getattr(last, "stopReason", None)
                if stop in ("error", "aborted"):
                    err_msg = getattr(last, "error_message", None) or getattr(last, "errorMessage", f"Request {stop}")
                    print(err_msg or f"Request {stop}", file=sys.stderr, flush=True)
                    return 1

                # Output text content
                for block in last.content:
                    if isinstance(block, TextContent) and block.text:
                        print(block.text, flush=True)

        # Check state for errors
        if hasattr(session.state, "error") and session.state.error:
            return 1

        return 0

    finally:
        unsub()
        # Explicit stdout flush (mirrors TS process.stdout.write("", resolve))
        sys.stdout.flush()


def _handle_print_event(event: AgentEvent, show_thinking: bool = False) -> None:
    """Handle a single agent event in text print mode."""
    if event.type == "message_end":
        msg = event.message
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextContent) and block.text.strip():
                    print(block.text, flush=True)
                elif hasattr(block, "thinking") and show_thinking and block.thinking.strip():
                    print(f"[Thinking: {block.thinking[:200]}...]", flush=True)

    elif event.type == "tool_execution_start":
        name = getattr(event, "tool_name", "")
        args = getattr(event, "args", {})
        print(f"  → {name}({_format_args(args)})", flush=True)

    elif event.type == "tool_execution_end":
        if getattr(event, "is_error", False):
            print("  ✗ Tool error", file=sys.stderr, flush=True)


def _format_args(args: Any) -> str:
    """Format tool arguments for display."""
    if not isinstance(args, dict):
        return str(args)
    parts = []
    for k, v in list(args.items())[:3]:
        v_str = str(v)
        if len(v_str) > 50:
            v_str = v_str[:47] + "..."
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)


def _event_to_dict(event: AgentEvent) -> dict[str, Any]:
    """Convert an agent event to a JSON-serializable dict."""
    base: dict[str, Any] = {"type": event.type}

    if event.type == "message_end":
        msg = event.message
        if isinstance(msg, AssistantMessage):
            base["role"] = "assistant"
            base["content"] = [
                {"type": "text", "text": b.text}
                for b in msg.content
                if isinstance(b, TextContent)
            ]
            stop = getattr(msg, "stop_reason", None) or getattr(msg, "stopReason", None)
            if stop:
                base["stopReason"] = stop
            usage = msg.usage
            if usage:
                base["usage"] = {
                    "input": getattr(usage, "input", 0),
                    "output": getattr(usage, "output", 0),
                }
        elif isinstance(msg, ToolResultMessage):
            base["role"] = "tool_result"
            base["toolCallId"] = msg.tool_call_id
            base["content"] = [
                {"type": "text", "text": b.text}
                for b in msg.content
                if isinstance(b, TextContent)
            ]
            base["isError"] = msg.is_error

    elif event.type == "message_start":
        base["role"] = getattr(event, "role", "")

    elif event.type == "tool_execution_start":
        base["toolName"] = getattr(event, "tool_name", "")
        args = getattr(event, "args", {})
        base["args"] = args

    elif event.type == "tool_execution_end":
        base["toolCallId"] = getattr(event, "tool_call_id", "")
        base["toolName"] = getattr(event, "tool_name", "")
        base["isError"] = getattr(event, "is_error", False)

    elif event.type == "agent_end":
        reason = getattr(event, "reason", "") or getattr(event, "stop_reason", "")
        base["reason"] = reason

    return base
