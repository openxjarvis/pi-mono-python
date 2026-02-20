"""
AgentSession — mirrors packages/coding-agent/src/core/agent-session.ts

Central class managing agent lifecycle, session persistence, tools, and events.
Full parity with TypeScript: auto-retry, overflow compaction, tool management,
model/thinking cycling, context usage, session stats, and queue management.
"""
from __future__ import annotations

import asyncio
import html
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from pi_agent import Agent, AgentOptions
from pi_agent.types import (
    AgentEvent,
    AgentMessage,
    AgentTool,
    ThinkingLevel,
)
from pi_ai import get_model, is_context_overflow
from pi_ai.types import AssistantMessage, ImageContent, Model, TextContent, UserMessage

from .auth_storage import AuthStorage
from .compaction import compact_context, should_compact
from .model_registry import ModelRegistry
from .session_manager import SessionManager
from .settings_manager import Settings, SettingsManager
from .system_prompt import build_system_prompt
from .tools import (
    create_bash_tool,
    create_edit_tool,
    create_find_tool,
    create_grep_tool,
    create_ls_tool,
    create_read_tool,
    create_write_tool,
)

# ── Thinking levels (mirrors TS constants) ────────────────────────────────────
_THINKING_LEVELS: list[ThinkingLevel] = ["off", "minimal", "low", "medium", "high"]
_THINKING_LEVELS_WITH_XHIGH: list[ThinkingLevel] = ["off", "minimal", "low", "medium", "high", "xhigh"]

# ── Retry error pattern (mirrors TS _isRetryableError regex) ─────────────────
_RETRY_PATTERN = re.compile(
    r"overloaded|rate.?limit|too many requests|429|500|502|503|504|"
    r"service.?unavailable|server error|internal error|connection.?error|"
    r"connection.?refused|other side closed|fetch failed|upstream.?connect|"
    r"reset before headers|terminated|retry delay",
    re.IGNORECASE,
)


class AgentSession:
    """
    Manages an agent session with persistence, tools, and events.
    Mirrors AgentSession in TypeScript.

    Key features vs. old version:
    - Per-message session persistence (message_end, not agent_end)
    - Auto-retry with exponential backoff
    - Overflow-aware auto-compaction (two paths: overflow vs threshold)
    - Tool registry with set_active_tools_by_name()
    - Model cycling (cycle_model), thinking cycling (cycle_thinking_level)
    - Context usage and session statistics
    - Queue management (clear_queue, pending_message_count)
    """

    def __init__(
        self,
        cwd: str | None = None,
        model: Model | None = None,
        settings: Settings | None = None,
        session_id: str | None = None,
        session_manager: SessionManager | None = None,
        auth_storage: AuthStorage | None = None,
        model_registry: ModelRegistry | None = None,
        settings_manager: SettingsManager | None = None,
    ) -> None:
        self.cwd = cwd or os.getcwd()
        self._settings = settings or Settings()
        self._auth_storage = auth_storage or AuthStorage()
        self._model_registry = model_registry or ModelRegistry()
        self._settings_manager = settings_manager or SettingsManager.create(cwd=self.cwd)

        if session_manager is not None:
            self._session_manager = session_manager
        else:
            self._session_manager = SessionManager.create(cwd=self.cwd)

        self.session_id = self._session_manager.get_session_id()

        # Build all tools; keep registry for set_active_tools_by_name
        self._all_tools: list[AgentTool] = self._build_tools()
        active_tools = list(self._all_tools)  # start with all tools active

        # Resolve model
        resolved_model = model or self._resolve_default_model()

        # Build system prompt (stored as _base_system_prompt so it can be rebuilt)
        self._base_system_prompt = build_system_prompt(
            self.cwd, selected_tools=[t.name for t in active_tools]
        )

        opts = AgentOptions(get_api_key=self._resolve_api_key)
        self._agent = Agent(opts)
        self._agent.set_model(resolved_model)
        self._agent.set_system_prompt(self._base_system_prompt)
        self._agent.set_tools(active_tools)
        self._agent.set_thinking_level(self._settings.thinking_level)

        self._listeners: list[Callable[[AgentEvent], None]] = []
        self._agent.subscribe(self._on_agent_event)

        # ── Auto-retry state ──────────────────────────────────────────────────
        self._retry_attempt: int = 0
        self._retry_event: asyncio.Event | None = None      # set when retry resolves/fails
        self._retry_success: bool = False

        # ── Auto-compaction abort ─────────────────────────────────────────────
        self._compaction_abort: asyncio.Event = asyncio.Event()
        self._compaction_running: bool = False

        # ── Last assistant message tracker (for auto-compaction/retry check) ──
        self._last_assistant_msg: AssistantMessage | None = None

    # ── Tool construction ─────────────────────────────────────────────────────

    def _build_tools(self) -> list[AgentTool]:
        """Create all default coding tools."""
        return [
            create_read_tool(self.cwd),
            create_write_tool(self.cwd),
            create_edit_tool(self.cwd),
            create_bash_tool(self.cwd),
            create_grep_tool(self.cwd),
            create_find_tool(self.cwd),
            create_ls_tool(self.cwd),
        ]

    # ── Model resolution ──────────────────────────────────────────────────────

    def _resolve_default_model(self) -> Model:
        """Resolve the default model from settings."""
        try:
            resolved = self._model_registry.resolve_model(
                model_id=self._settings.model_id,
                provider=self._settings.provider,
            )
            explicit_requested = bool(self._settings.model_id or self._settings.provider)
            has_auth = bool(self._model_registry.get_api_key(resolved.provider))
            if explicit_requested and not has_auth:
                for prov, mid in (
                    ("google", "gemini-2.0-flash"),
                    ("anthropic", "claude-3-5-sonnet-20241022"),
                    ("openai", "gpt-4o"),
                ):
                    if self._model_registry.get_api_key(prov):
                        fallback = self._model_registry.find(prov, mid)
                        if fallback:
                            return fallback
            return resolved
        except Exception:
            if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
                return get_model("google", "gemini-2.0-flash")
            return get_model("anthropic", "claude-3-5-sonnet-20241022")

    async def _resolve_api_key(self, provider: str) -> str | None:
        return self._auth_storage.resolve_api_key(provider)

    # ── Event handling ────────────────────────────────────────────────────────

    def _on_agent_event(self, event: AgentEvent) -> None:
        """Handle agent events — persist messages and notify listeners."""
        # ── 2a: Persist messages on message_end (not agent_end) ──────────────
        if event.type == "message_end":
            msg = getattr(event, "message", None)
            if msg is not None:
                role = getattr(msg, "role", "")
                if role in ("user", "assistant", "toolResult"):
                    self._session_manager.append_message(_message_to_dict(msg))
                # Track last assistant message for retry/compaction
                if role == "assistant":
                    self._last_assistant_msg = msg
                    # Reset retry on successful non-error response
                    stop_reason = getattr(msg, "stop_reason", "")
                    if stop_reason != "error" and self._retry_attempt > 0:
                        self._emit({"type": "auto_retry_end", "success": True,
                                    "attempt": self._retry_attempt})
                        self._retry_attempt = 0
                        self._resolve_retry(success=True)

        # ── agent_end: check retry and compaction ─────────────────────────────
        if event.type == "agent_end":
            if self._last_assistant_msg is not None:
                msg = self._last_assistant_msg
                self._last_assistant_msg = None
                # Schedule retry / compaction check asynchronously
                try:
                    loop = asyncio.get_running_loop()
                    loop.call_soon(lambda: asyncio.ensure_future(
                        self._post_turn_checks(msg)
                    ))
                except RuntimeError:
                    pass  # no running loop in sync context

        # Notify external listeners
        for listener in list(self._listeners):
            listener(event)

    async def _post_turn_checks(self, msg: AssistantMessage) -> None:
        """Check retry and compaction after a turn completes (mirrors TS _handleAgentEvent)."""
        # Retry takes priority over compaction
        if self._is_retryable_error(msg):
            did_retry = await self._handle_retryable_error(msg)
            if did_retry:
                return
        await self._check_compaction(msg)

    def _emit(self, event: dict | Any) -> None:
        """Emit a synthetic session event to all listeners."""
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                pass

    # ── Subscription ──────────────────────────────────────────────────────────

    def subscribe(self, fn: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """Subscribe to session events. Returns unsubscribe function."""
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn) if fn in self._listeners else None

    # ── Agent control ─────────────────────────────────────────────────────────

    async def prompt(
        self,
        message: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
        source: str | None = None,
    ) -> None:
        """Send a prompt to the agent and wait for completion (including retries)."""
        # Reset retry state for new prompt
        self._retry_event = asyncio.Event()
        self._retry_success = False
        self._retry_attempt = 0

        await self._agent.prompt(message, images)
        # Wait for any pending retries to complete
        await self._wait_for_retry()

    async def steer(self, message: AgentMessage) -> None:
        """Queue a steering message."""
        self._agent.steer(message)

    async def follow_up(self, message: AgentMessage) -> None:
        """Queue a follow-up message."""
        self._agent.follow_up(message)

    async def abort(self) -> None:
        """Abort current operation and wait for agent to become idle."""
        self._abort_retry()
        self._agent.abort()
        await self._agent.wait_for_idle()

    async def wait_for_idle(self) -> None:
        await self._agent.wait_for_idle()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self):
        return self._agent.state

    @property
    def model(self) -> Model | None:
        return self._agent.state.model

    @property
    def thinking_level(self) -> ThinkingLevel:
        return self._agent.state.thinking_level

    @property
    def is_streaming(self) -> bool:
        return self._agent.state.is_streaming

    @property
    def is_compacting(self) -> bool:
        return self._compaction_running

    @property
    def is_retrying(self) -> bool:
        return self._retry_attempt > 0

    @property
    def retry_attempt(self) -> int:
        return self._retry_attempt

    @property
    def model_registry(self) -> ModelRegistry:
        return self._model_registry

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    @property
    def system_prompt(self) -> str:
        return self._agent.state.system_prompt

    # ── Queue management ──────────────────────────────────────────────────────

    @property
    def pending_message_count(self) -> int:
        return (len(self._agent._steering_queue)
                + len(self._agent._follow_up_queue))

    def get_steering_messages(self) -> list[str]:
        return [getattr(m, "content", str(m)) for m in self._agent._steering_queue]

    def get_follow_up_messages(self) -> list[str]:
        return [getattr(m, "content", str(m)) for m in self._agent._follow_up_queue]

    def clear_queue(self) -> dict[str, list]:
        steering = list(self._agent._steering_queue)
        follow_up = list(self._agent._follow_up_queue)
        self._agent.clear_all_queues()
        return {"steering": steering, "followUp": follow_up}

    # ── Tool management (2d) ──────────────────────────────────────────────────

    def get_active_tool_names(self) -> list[str]:
        """Get names of currently active tools."""
        return [t.name for t in self._agent.state.tools]

    def get_all_tool_names(self) -> list[str]:
        """Get names of all registered tools."""
        return [t.name for t in self._all_tools]

    def set_active_tools_by_name(self, tool_names: list[str]) -> None:
        """
        Set active tools by name. Rebuilds system prompt to reflect new tool set.
        Mirrors setActiveToolsByName() in TypeScript.
        """
        name_set = set(tool_names)
        active = [t for t in self._all_tools if t.name in name_set]
        self._agent.set_tools(active)
        valid_names = [t.name for t in active]
        self._base_system_prompt = build_system_prompt(
            self.cwd, selected_tools=valid_names
        )
        self._agent.set_system_prompt(self._base_system_prompt)

    # ── Model management (2g) ─────────────────────────────────────────────────

    async def set_model(self, model: Model) -> None:
        """
        Switch the active model with API key validation.
        Mirrors setModel() in TypeScript.
        """
        api_key = self._model_registry.get_api_key(model.provider)
        if not api_key:
            raise RuntimeError(f"No API key for {model.provider}/{model.id}")
        self._agent.set_model(model)
        self._session_manager.append_model_change(model.id, model.provider)
        # Re-clamp thinking level for new model
        self.set_thinking_level(self.thinking_level)

    async def cycle_model(self, direction: str = "forward") -> dict | None:
        """
        Cycle to next/previous available model.
        Mirrors cycleModel() in TypeScript.
        Returns new model info or None if only one model available.
        """
        available = await self._model_registry.get_available()
        if len(available) <= 1:
            return None
        current = self._agent.state.model
        current_idx = next(
            (i for i, m in enumerate(available)
             if m.provider == getattr(current, "provider", "") and m.id == getattr(current, "id", "")),
            0,
        )
        n = len(available)
        next_idx = (current_idx + 1) % n if direction == "forward" else (current_idx - 1 + n) % n
        next_model = available[next_idx]
        await self.set_model(next_model)
        return {"model": next_model}

    # ── Thinking level management (2h) ────────────────────────────────────────

    def get_available_thinking_levels(self) -> list[ThinkingLevel]:
        """Get thinking levels available for current model."""
        from pi_ai import supports_xhigh
        model = self._agent.state.model
        if model and supports_xhigh(model):
            return list(_THINKING_LEVELS_WITH_XHIGH)
        return list(_THINKING_LEVELS)

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        """Set thinking level, clamped to model capabilities. Persists to session."""
        available = self.get_available_thinking_levels()
        effective = level if level in available else _clamp_thinking_level(level, available)
        is_changing = effective != self._agent.state.thinking_level
        self._agent.set_thinking_level(effective)
        if is_changing:
            self._session_manager.append_thinking_level_change(effective)

    def cycle_thinking_level(self) -> ThinkingLevel | None:
        """
        Cycle to next thinking level.
        Mirrors cycleThinkingLevel() in TypeScript.
        Returns new level or None if model doesn't support thinking.
        """
        available = self.get_available_thinking_levels()
        if available == ["off"]:
            return None
        current = self._agent.state.thinking_level
        idx = available.index(current) if current in available else 0
        next_level = available[(idx + 1) % len(available)]
        self.set_thinking_level(next_level)
        return next_level

    # ── Session statistics (2f) ───────────────────────────────────────────────

    def get_session_stats(self) -> dict[str, Any]:
        """
        Get session statistics (message counts, token totals, cost).
        Mirrors getSessionStats() in TypeScript.
        """
        msgs = self._agent.state.messages
        user_messages = sum(1 for m in msgs if getattr(m, "role", "") == "user")
        assistant_messages = sum(1 for m in msgs if getattr(m, "role", "") == "assistant")
        tool_results = sum(1 for m in msgs if getattr(m, "role", "") == "toolResult")
        tool_calls = 0
        total_input = 0
        total_output = 0
        total_cache_read = 0
        total_cache_write = 0
        total_cost = 0.0

        for m in msgs:
            if getattr(m, "role", "") == "assistant":
                content = getattr(m, "content", [])
                tool_calls += sum(
                    1 for c in content if getattr(c, "type", "") == "toolCall"
                )
                usage = getattr(m, "usage", None)
                if usage:
                    total_input += getattr(usage, "input", 0)
                    total_output += getattr(usage, "output", 0)
                    total_cache_read += getattr(usage, "cache_read", 0)
                    total_cache_write += getattr(usage, "cache_write", 0)
                    cost = getattr(usage, "cost", None)
                    if cost:
                        total_cost += getattr(cost, "total", 0) or 0.0

        return {
            "sessionId": self.session_id,
            "sessionFile": self._session_manager.get_session_file(),
            "userMessages": user_messages,
            "assistantMessages": assistant_messages,
            "toolCalls": tool_calls,
            "toolResults": tool_results,
            "totalMessages": len(msgs),
            "tokens": {
                "input": total_input,
                "output": total_output,
                "cacheRead": total_cache_read,
                "cacheWrite": total_cache_write,
                "total": total_input + total_output + total_cache_read + total_cache_write,
            },
            "cost": total_cost,
        }

    # ── Context usage (2e) ────────────────────────────────────────────────────

    def get_context_usage(self) -> dict | None:
        """
        Get current context window usage.
        Mirrors getContextUsage() in TypeScript.
        """
        model = self._agent.state.model
        if not model:
            return None
        context_window = getattr(model, "context_window", 0) or 0
        if context_window <= 0:
            return None
        tokens = self._estimate_context_tokens()
        percent = (tokens / context_window * 100) if context_window else 0
        return {"tokens": tokens, "contextWindow": context_window, "percent": percent}

    def _estimate_context_tokens(self) -> int:
        """Estimate current context size from last assistant usage or message lengths."""
        msgs = self._agent.state.messages
        # Walk backwards to find last assistant message with usage
        for m in reversed(msgs):
            if getattr(m, "role", "") == "assistant":
                usage = getattr(m, "usage", None)
                if usage:
                    inp = getattr(usage, "input", 0) or 0
                    out = getattr(usage, "output", 0) or 0
                    cr = getattr(usage, "cache_read", 0) or 0
                    if inp + out + cr > 0:
                        return inp + cr  # context tokens = input + cache_read
        # Fallback: estimate from character count
        total_chars = sum(
            len(str(getattr(m, "content", ""))) for m in msgs
        )
        return total_chars // 4

    # ── Session management ────────────────────────────────────────────────────

    def fork(self) -> "AgentSession":
        """Create a fork of the current session."""
        sessions_dir = self._session_manager.get_session_dir()
        src_path = self._session_manager.get_session_file()
        forked_sm = SessionManager.fork_from(src_path, self.cwd, sessions_dir)
        forked = AgentSession(
            cwd=self.cwd,
            model=self._agent.state.model,
            settings=self._settings,
            session_manager=forked_sm,
            auth_storage=self._auth_storage,
            model_registry=self._model_registry,
        )
        forked._agent.replace_messages(list(self._agent.state.messages))
        return forked

    def get_session_info(self) -> dict[str, Any]:
        """Get basic session information (backwards compat)."""
        return {
            "session_id": self.session_id,
            "cwd": self.cwd,
            "model": self._agent.state.model.id if self._agent.state.model else None,
            "message_count": len(self._agent.state.messages),
            "is_streaming": self._agent.state.is_streaming,
        }

    # ── Compaction ────────────────────────────────────────────────────────────

    async def compact(self, custom_instructions: str | None = None) -> str:
        """Manually compact the context. Returns the summary."""
        messages = self._agent.state.messages
        model = self._agent.state.model
        if not model:
            return ""

        from pi_ai import stream_simple
        new_messages, summary = await compact_context(
            messages,
            self._agent.state.system_prompt,
            stream_simple,
            model,
        )
        self._agent.replace_messages(new_messages)

        if summary:
            first_kept_id = str(len(messages)) if messages else "0"
            self._session_manager.append_compaction(summary, first_kept_id)

        return summary

    async def _check_compaction(self, msg: AssistantMessage, skip_aborted: bool = True) -> None:
        """
        Check if compaction is needed and run it.
        Mirrors _checkCompaction() in TypeScript with two cases:
        1. Overflow — LLM returned context overflow error → compact + retry
        2. Threshold — context over threshold → compact (no auto-retry)
        """
        settings = self._settings_manager.get_compaction_settings()
        if not settings.get("enabled", True):
            return
        if skip_aborted and getattr(msg, "stop_reason", "") == "aborted":
            return

        model = self._agent.state.model
        context_window = getattr(model, "context_window", 0) if model else 0

        # Case 1: Overflow
        same_model = (
            model and
            getattr(msg, "provider", None) == model.provider and
            getattr(msg, "model", None) == model.id
        )
        if same_model and is_context_overflow(msg, context_window):
            # Remove the error message from agent state (keep in session history)
            messages = self._agent.state.messages
            if messages and getattr(messages[-1], "role", "") == "assistant":
                self._agent.replace_messages(messages[:-1])
            await self._run_auto_compaction("overflow", will_retry=True)
            return

        # Case 2: Threshold
        if getattr(msg, "stop_reason", "") == "error":
            return  # non-overflow errors have no usage data
        tokens = self._estimate_context_tokens()
        reserve = settings.get("reserveTokens", 16384)
        if context_window > 0 and should_compact(
            self._agent.state.messages,
            context_window,
            (context_window - reserve) / context_window,
        ):
            await self._run_auto_compaction("threshold", will_retry=False)

    async def _run_auto_compaction(self, reason: str, will_retry: bool) -> None:
        """Run auto-compaction with events (mirrors _runAutoCompaction in TS)."""
        self._compaction_running = True
        self._emit({"type": "auto_compaction_start", "reason": reason})
        try:
            model = self._agent.state.model
            if not model:
                self._emit({"type": "auto_compaction_end", "result": None,
                            "aborted": False, "willRetry": False})
                return

            from pi_ai import stream_simple
            messages = self._agent.state.messages
            new_messages, summary = await compact_context(
                messages,
                self._agent.state.system_prompt,
                stream_simple,
                model,
            )

            self._agent.replace_messages(new_messages)

            result = {"summary": summary, "tokensBefore": self._estimate_context_tokens()}
            if summary:
                first_kept_id = str(len(messages)) if messages else "0"
                self._session_manager.append_compaction(summary, first_kept_id)

            self._emit({"type": "auto_compaction_end", "result": result,
                        "aborted": False, "willRetry": will_retry})

            if will_retry:
                # Schedule agent.continue() via call_soon to break out of event chain
                try:
                    loop = asyncio.get_running_loop()
                    loop.call_soon(lambda: asyncio.ensure_future(
                        self._agent.continue_from_context()
                    ))
                except RuntimeError:
                    pass
            elif self._agent.has_queued_messages():
                try:
                    loop = asyncio.get_running_loop()
                    loop.call_soon(lambda: asyncio.ensure_future(
                        self._agent.continue_from_context()
                    ))
                except RuntimeError:
                    pass

        except Exception as e:
            err_msg = str(e)
            prefix = "Context overflow recovery failed" if reason == "overflow" else "Auto-compaction failed"
            self._emit({"type": "auto_compaction_end", "result": None, "aborted": False,
                        "willRetry": False, "errorMessage": f"{prefix}: {err_msg}"})
        finally:
            self._compaction_running = False

    def abort_compaction(self) -> None:
        self._compaction_abort.set()

    # ── Auto-retry (2b) ───────────────────────────────────────────────────────

    def _is_retryable_error(self, msg: AssistantMessage) -> bool:
        """
        Check if error is retryable (rate limit, overloaded, server errors).
        Context overflow is NOT retryable — handled by compaction.
        Mirrors _isRetryableError() in TypeScript.
        """
        if getattr(msg, "stop_reason", "") != "error":
            return False
        model = self._agent.state.model
        context_window = getattr(model, "context_window", 0) if model else 0
        if is_context_overflow(msg, context_window):
            return False
        err = getattr(msg, "error_message", "") or ""
        return bool(_RETRY_PATTERN.search(err))

    async def _handle_retryable_error(self, msg: AssistantMessage) -> bool:
        """
        Handle retryable errors with exponential backoff.
        Mirrors _handleRetryableError() in TypeScript.
        Returns True if retry was initiated.
        """
        settings = self._settings_manager.get_retry_settings()
        if not settings.get("enabled", True):
            return False

        self._retry_attempt += 1
        max_retries = settings.get("maxRetries", 3)
        base_delay_ms = settings.get("baseDelayMs", 2000)

        if self._retry_attempt > max_retries:
            self._emit({
                "type": "auto_retry_end",
                "success": False,
                "attempt": self._retry_attempt - 1,
                "finalError": getattr(msg, "error_message", "Unknown error"),
            })
            self._retry_attempt = 0
            self._resolve_retry(success=False)
            return False

        delay_ms = base_delay_ms * (2 ** (self._retry_attempt - 1))

        self._emit({
            "type": "auto_retry_start",
            "attempt": self._retry_attempt,
            "maxAttempts": max_retries,
            "delayMs": delay_ms,
            "errorMessage": getattr(msg, "error_message", "Unknown error"),
        })

        # Remove error message from state (keep in session for history)
        messages = self._agent.state.messages
        if messages and getattr(messages[-1], "role", "") == "assistant":
            self._agent.replace_messages(messages[:-1])

        # Wait with exponential backoff
        try:
            await asyncio.sleep(delay_ms / 1000.0)
        except asyncio.CancelledError:
            self._emit({"type": "auto_retry_end", "success": False,
                        "attempt": self._retry_attempt, "finalError": "Retry cancelled"})
            self._retry_attempt = 0
            self._resolve_retry(success=False)
            return False

        # Schedule retry via call_soon
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon(lambda: asyncio.ensure_future(
                self._agent.continue_from_context()
            ))
        except RuntimeError:
            pass

        return True

    def _resolve_retry(self, success: bool) -> None:
        """Resolve the pending retry event."""
        self._retry_success = success
        if self._retry_event:
            self._retry_event.set()

    def _abort_retry(self) -> None:
        """Cancel in-progress retry."""
        if self._retry_event:
            self._retry_event.set()

    async def _wait_for_retry(self) -> None:
        """Wait for any in-progress retry to complete (mirrors waitForRetry in TS)."""
        if self._retry_attempt == 0:
            return
        if self._retry_event and not self._retry_event.is_set():
            await self._retry_event.wait()

    # ── HTML export ───────────────────────────────────────────────────────────

    async def export_to_html(self, output_path: str | None = None) -> str:
        """Export current session messages to a basic HTML transcript."""
        messages = self._session_manager.get_messages()
        if not output_path:
            output_path = os.path.join(self.cwd, f"{self.session_id}.html")
        rows: list[str] = []
        for msg in messages:
            role = html.escape(str(msg.get("role", "unknown")))
            content = html.escape(str(msg.get("content", "")))
            rows.append(f"<div><strong>{role}</strong>: <pre>{content}</pre></div>")
        body = "\n".join(rows) or "<div><em>No messages</em></div>"
        html_doc = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Session Export</title></head><body>"
            f"{body}</body></html>"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        return output_path

    def get_last_assistant_text(self) -> str | None:
        """Get text of last assistant message (for /copy command)."""
        for m in reversed(self._agent.state.messages):
            if getattr(m, "role", "") != "assistant":
                continue
            stop = getattr(m, "stop_reason", "")
            content = getattr(m, "content", [])
            if stop == "aborted" and not content:
                continue
            text = "".join(
                getattr(c, "text", "")
                for c in content
                if getattr(c, "type", "") == "text"
            )
            return text.strip() or None
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp_thinking_level(level: ThinkingLevel, available: list[ThinkingLevel]) -> ThinkingLevel:
    """Clamp a thinking level to the closest available level."""
    ordered = _THINKING_LEVELS_WITH_XHIGH
    avail_set = set(available)
    idx = ordered.index(level) if level in ordered else 0
    for i in range(idx, len(ordered)):
        if ordered[i] in avail_set:
            return ordered[i]
    for i in range(idx - 1, -1, -1):
        if ordered[i] in avail_set:
            return ordered[i]
    return available[0] if available else "off"


def _message_to_dict(msg: Any) -> dict[str, Any]:
    """Convert a message to a dict for persistence."""
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    return {"role": getattr(msg, "role", "unknown")}
