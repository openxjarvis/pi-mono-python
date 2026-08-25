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
import sys
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
from .compaction import compact_context, estimate_context_tokens
from .messages import wrap_convert_to_llm
from .model_registry import ModelRegistry
from .session_manager import SessionManager
from .settings_manager import Settings, SettingsManager
from .system_prompt import build_system_prompt
from .tools import (
    create_bash_tool,
    create_edit_tool,
    create_powershell_tool,
    create_find_tool,
    create_grep_tool,
    create_ls_tool,
    create_read_tool,
    create_write_tool,
)

# ── Thinking levels (mirrors TS constants) ────────────────────────────────────
_THINKING_LEVELS: list[ThinkingLevel] = ["off", "minimal", "low", "medium", "high"]
_THINKING_LEVELS_WITH_XHIGH: list[ThinkingLevel] = ["off", "minimal", "low", "medium", "high", "xhigh"]
_THINKING_LEVELS_WITH_MAX: list[ThinkingLevel] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"]

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

        # Create convertToLlm wrapper with blockImages support
        convert_to_llm_fn = wrap_convert_to_llm(self._settings_manager.get_block_images())

        # Create Agent with convert_to_llm and transform_context
        opts = AgentOptions(
            get_api_key=self._resolve_api_key,
            convert_to_llm=convert_to_llm_fn,
            transform_context=self._transform_context,
        )
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
        self._overflow_recovery_attempted: bool = False  # one-shot guard against infinite overflow loops

        # ── Last assistant message tracker (for auto-compaction/retry check) ──
        self._last_assistant_msg: AssistantMessage | None = None

        # ── Bash execution state ──────────────────────────────────────────────
        self._pending_bash_messages: list[AgentMessage] = []
        self._pending_next_turn_messages: list[AgentMessage] = []

        # ── Scoped models (for cycling) ───────────────────────────────────────
        self._scoped_models: list[dict[str, Model | ThinkingLevel | None]] | None = None

        # ── Queue modes (RPC get_state / set_*_mode) ──────────────────────────
        self._steering_mode: str = self._settings_manager.get_steering_mode() or "one-at-a-time"
        self._follow_up_mode: str = self._settings_manager.get_follow_up_mode() or "one-at-a-time"

    # ── Tool construction ─────────────────────────────────────────────────────

    def _build_tools(self) -> list[AgentTool]:
        """Create all default coding tools."""
        tools = [
            create_read_tool(self.cwd),
            create_write_tool(self.cwd),
            create_edit_tool(self.cwd),
            create_bash_tool(self.cwd),
            create_grep_tool(self.cwd),
            create_find_tool(self.cwd),
            create_ls_tool(self.cwd),
        ]
        if sys.platform == "win32":
            try:
                tools.append(create_powershell_tool(self.cwd))
            except RuntimeError:
                pass
        return tools

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

    async def _transform_context(
        self, 
        messages: list[AgentMessage], 
        signal: asyncio.Event | None = None
    ) -> list[AgentMessage]:
        """
        Transform context before convert_to_llm.
        
        Currently returns messages unchanged. 
        Will be connected to ExtensionRunner.emit_context when extension support is added.
        Mirrors the transform_context callback in TypeScript SDK.
        """
        # TODO: Connect to ExtensionRunner.emit_context when available
        # if self._extension_runner:
        #     return await self._extension_runner.emit_context(messages)
        return messages

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
        # Reset overflow recovery on successful turns
        if getattr(msg, "stop_reason", "") not in ("error", "aborted"):
            self._overflow_recovery_attempted = False
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
        expand_prompt_templates: bool = True,
        streaming_behavior: str | None = None,
    ) -> None:
        """
        Send a prompt to the agent and wait for completion (including retries).
        Mirrors prompt() in TypeScript with:
        - Model/API key validation
        - Pre-prompt compaction check
        - Bash flush
        - pendingNextTurnMessages injection
        """
        current_text: str | None = None
        current_images = images

        # Normalize input
        if isinstance(message, str):
            current_text = message
        elif isinstance(message, list):
            pass  # already message list
        else:
            pass  # single message

        # If streaming, queue via steer/followUp
        if self._agent.state.is_streaming:
            if not streaming_behavior:
                raise RuntimeError(
                    "Agent is already processing. Specify streaming_behavior ('steer' or 'followUp') to queue."
                )
            if current_text is not None:
                user_msg = UserMessage(
                    role="user",
                    content=[TextContent(type="text", text=current_text)],
                    timestamp=int(time.time() * 1000),
                )
                if streaming_behavior == "followUp":
                    self._agent.follow_up(user_msg)
                else:
                    self._agent.steer(user_msg)
            return

        # Reset overflow recovery flag for each new user-initiated turn
        self._overflow_recovery_attempted = False

        # Flush pending bash messages
        self._flush_pending_bash_messages()

        # Validate model
        if not self._agent.state.model:
            raise RuntimeError("No model selected. Use /login or set an API key environment variable.")

        # Validate API key
        model = self._agent.state.model
        api_key = await self._resolve_api_key(model.provider)
        if not api_key:
            raise RuntimeError(
                f"No API key found for {model.provider}. "
                "Use /login or set an API key environment variable."
            )

        # Pre-compaction check on last assistant message
        last_assistant = self._find_last_assistant_message()
        if last_assistant:
            await self._check_compaction(last_assistant, skip_aborted=False)

        # Build messages array
        msgs: list[AgentMessage]
        if isinstance(message, list):
            msgs = message
        elif current_text is not None:
            content_parts: list[TextContent | ImageContent] = [TextContent(type="text", text=current_text)]
            if current_images:
                content_parts.extend(current_images)
            msgs = [UserMessage(
                role="user",
                content=content_parts,
                timestamp=int(time.time() * 1000),
            )]
        else:
            msgs = [message]

        # Inject pending next-turn messages
        if self._pending_next_turn_messages:
            msgs.extend(self._pending_next_turn_messages)
            self._pending_next_turn_messages = []

        # Reset system prompt to base
        self._agent.set_system_prompt(self._base_system_prompt)

        # Reset retry state
        self._retry_event = asyncio.Event()
        self._retry_success = False
        self._retry_attempt = 0

        await self._agent.prompt(msgs)
        await self._wait_for_retry()

    def _flush_pending_bash_messages(self) -> None:
        """Flush pending bash messages into agent state and session."""
        if not self._pending_bash_messages:
            return
        for bash_msg in self._pending_bash_messages:
            self._agent.append_message(bash_msg)
            self._session_manager.append_message(_message_to_dict(bash_msg))
        self._pending_bash_messages = []

    def _find_last_assistant_message(self) -> AssistantMessage | None:
        """Find the last assistant message in the current context."""
        for m in reversed(self._agent.state.messages):
            if getattr(m, "role", "") == "assistant":
                return m
        return None

    def record_bash_result(
        self,
        tool_call_id: str,
        output: str,
        exit_code: int,
    ) -> None:
        """
        Record a bash execution result.
        If streaming, queues for later flush; otherwise adds immediately.
        """
        from pi_ai.types import ToolResultMessage
        bash_msg = ToolResultMessage(
            role="toolResult",
            tool_call_id=tool_call_id,
            tool_name="bash",
            content=[TextContent(type="text", text=output)],
            is_error=exit_code != 0,
            timestamp=int(time.time() * 1000),
        )
        if self._agent.state.is_streaming:
            self._pending_bash_messages.append(bash_msg)
        else:
            self._agent.append_message(bash_msg)
            self._session_manager.append_message(_message_to_dict(bash_msg))

    # ── Session switching / tree navigation ────────────────────────────────────

    async def switch_session(self, session_path: str) -> bool:
        """
        Switch to a different session file.
        Mirrors switchSession() in TypeScript.
        Returns True if switch succeeded.
        """
        self._agent.abort()
        await self._agent.wait_for_idle()

        self._agent.clear_all_queues()
        self._pending_bash_messages = []
        self._pending_next_turn_messages = []

        new_sm = SessionManager.open(session_path)
        self._session_manager = new_sm
        self.session_id = new_sm.get_session_id()

        # Restore context from session
        context = new_sm.build_context()
        self._agent.replace_messages(context.messages)

        # Restore model/thinking from session
        if context.model:
            try:
                model = get_model(context.model["provider"], context.model["model_id"])
                self._agent.set_model(model)
            except Exception:
                pass
        if context.thinking_level:
            self._agent.set_thinking_level(context.thinking_level)

        return True

    async def new_session(self, options: dict[str, Any] | None = None) -> bool:
        """
        Start a new session, replacing the current one.

        Mirrors AgentSessionRuntime.newSession() / RPC ``new_session``.
        Returns True if the session was created (not cancelled).
        """
        await self.abort()
        self._agent.clear_all_queues()
        self._pending_bash_messages = []
        self._pending_next_turn_messages = []
        self._retry_attempt = 0
        self._overflow_recovery_attempted = False

        parent: str | None = None
        if options:
            parent = options.get("parentSession") or options.get("parent_session")

        session_dir = self._session_manager.get_session_dir()
        new_sm = SessionManager.create(
            self.cwd,
            session_dir=session_dir,
            parent_session=parent,
        )
        self._session_manager = new_sm
        self.session_id = new_sm.get_session_id()
        self._agent.replace_messages([])
        return True

    async def clone(self) -> dict[str, Any]:
        """
        Duplicate the current session at the current leaf.
        Mirrors RPC ``clone`` (fork at current entry).
        """
        leaf_id = self._session_manager.get_leaf_id()
        if not leaf_id:
            raise RuntimeError("Cannot clone session: no current entry selected")

        await self.abort()
        self._agent.clear_all_queues()
        self._pending_bash_messages = []
        self._pending_next_turn_messages = []

        new_sm = self._session_manager.branch(leaf_id, self.cwd)
        self._session_manager = new_sm
        self.session_id = new_sm.get_session_id()
        context = new_sm.build_context()
        self._agent.replace_messages(context.messages)
        return {"cancelled": False}

    async def import_from_jsonl(
        self,
        input_path: str,
        cwd_override: str | None = None,
    ) -> dict[str, Any]:
        """
        Import a session JSONL file and switch to it.
        Mirrors AgentSessionRuntime.importFromJsonl().
        """
        import shutil

        from .session_cwd import MissingSessionCwdError, assert_session_cwd_exists

        resolved = os.path.abspath(os.path.expanduser(input_path))
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"File not found: {resolved}")

        session_dir = self._session_manager.get_session_dir()
        os.makedirs(session_dir, exist_ok=True)
        destination = os.path.join(session_dir, os.path.basename(resolved))
        if os.path.abspath(destination) != resolved:
            shutil.copy2(resolved, destination)

        new_sm = SessionManager.open(destination)
        if cwd_override:
            if new_sm._header is not None:
                new_sm._header["cwd"] = cwd_override
            new_sm._cwd = cwd_override
        try:
            assert_session_cwd_exists(new_sm, cwd_override or self.cwd)
        except MissingSessionCwdError:
            raise

        await self.abort()
        self._session_manager = new_sm
        self.session_id = new_sm.get_session_id()
        if cwd_override:
            self.cwd = cwd_override
        elif new_sm.get_cwd():
            self.cwd = new_sm.get_cwd()
        context = new_sm.build_context()
        self._agent.replace_messages(context.messages)
        return {"cancelled": False}

    async def navigate_tree(
        self,
        target_id: str,
        summarize: bool = False,
    ) -> dict[str, Any]:
        """
        Navigate to a different point in the session tree.
        Mirrors navigateTree() in TypeScript.
        """
        old_leaf_id = self._session_manager.get_leaf_id()
        if target_id == old_leaf_id:
            return {"cancelled": False}

        self._agent.abort()
        await self._agent.wait_for_idle()

        # Optionally summarize the branch being left
        if summarize and old_leaf_id:
            from .compaction.branch_summarization import summarize_branch
            try:
                summary_result = await summarize_branch(
                    self._session_manager,
                    old_leaf_id,
                    target_id,
                    self._agent.state.model,
                )
                if summary_result and summary_result.summary:
                    self._session_manager.append_branch_summary(
                        summary_result.summary,
                        from_id=old_leaf_id,
                    )
            except Exception:
                pass

        self._session_manager.set_leaf_id(target_id)

        # Rebuild context from new position
        context = self._session_manager.build_context(target_id)
        self._agent.replace_messages(context.messages)

        if context.model:
            try:
                model = get_model(context.model["provider"], context.model["model_id"])
                self._agent.set_model(model)
            except Exception:
                pass
        if context.thinking_level:
            self._agent.set_thinking_level(context.thinking_level)

        # Check if target is a user message (return its text for editor)
        target_entry = self._session_manager.get_entry(target_id)
        editor_text: str | None = None
        if target_entry and target_entry.type == "message":
            msg_data = target_entry.data.get("message", {})
            if isinstance(msg_data, dict) and msg_data.get("role") == "user":
                content = msg_data.get("content", [])
                if isinstance(content, str):
                    editor_text = content
                elif isinstance(content, list):
                    editor_text = "".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )

        return {"cancelled": False, "editorText": editor_text}

    async def create_branched_session(self, branch_point_id: str) -> "AgentSession":
        """
        Create a new session branching from a specific entry.
        Mirrors createBranchedSession() via SessionManager.branch().
        """
        new_sm = self._session_manager.branch(branch_point_id, self.cwd)
        branched = AgentSession(
            cwd=self.cwd,
            model=self._agent.state.model,
            settings=self._settings,
            session_manager=new_sm,
            auth_storage=self._auth_storage,
            model_registry=self._model_registry,
        )
        context = new_sm.build_context()
        branched._agent.replace_messages(context.messages)
        return branched

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
    def settings_manager(self) -> SettingsManager:
        return self._settings_manager

    @property
    def scoped_models(self) -> list[Any]:
        return list(self._scoped_models or [])

    @property
    def resource_loader(self) -> Any:
        return getattr(self, "_resource_loader", None)

    @property
    def extension_runner(self) -> Any:
        return getattr(self, "_extension_runner", None)

    @property
    def model_runtime(self) -> Any:
        return getattr(self, "_model_runtime", None)

    @property
    def prompt_templates(self) -> list[Any]:
        return list(getattr(self, "_prompt_templates", []) or [])

    @property
    def system_prompt(self) -> str:
        return self._agent.state.system_prompt

    @property
    def agent(self):
        """Underlying Agent instance (used by RPC extension bindings)."""
        return self._agent

    @property
    def messages(self) -> list[Any]:
        """All messages including custom types. Mirrors TS AgentSession.messages."""
        return list(self._agent.state.messages)

    @property
    def steering_mode(self) -> str:
        return self._steering_mode

    @property
    def follow_up_mode(self) -> str:
        return self._follow_up_mode

    @property
    def session_file(self) -> str | None:
        return self._session_manager.get_session_file()

    @property
    def session_name(self) -> str | None:
        return self._session_manager.get_session_name()

    def set_steering_mode(self, mode: str) -> None:
        """Set steering queue mode and persist to settings."""
        if mode not in ("all", "one-at-a-time"):
            raise ValueError(f"Invalid steering mode: {mode}")
        self._steering_mode = mode
        self._agent.set_steering_mode(mode)
        if hasattr(self._settings_manager, "set_steering_mode"):
            self._settings_manager.set_steering_mode(mode)
        else:
            self._settings_manager.save_global("steeringMode", mode)

    def set_follow_up_mode(self, mode: str) -> None:
        """Set follow-up queue mode and persist to settings."""
        if mode not in ("all", "one-at-a-time"):
            raise ValueError(f"Invalid follow-up mode: {mode}")
        self._follow_up_mode = mode
        self._agent.set_follow_up_mode(mode)
        if hasattr(self._settings_manager, "set_follow_up_mode"):
            self._settings_manager.set_follow_up_mode(mode)
        else:
            self._settings_manager.save_global("followUpMode", mode)

    def abort_retry(self) -> None:
        """Public alias for cancelling in-progress retry (RPC abort_retry)."""
        self._abort_retry()

    # ── Queue management ──────────────────────────────────────────────────────

    @property
    def pending_message_count(self) -> int:
        return (
            len(self._agent.peek_steering_messages())
            + len(self._agent.peek_follow_up_messages())
        )

    def get_steering_messages(self) -> list[str]:
        return [getattr(m, "content", str(m)) for m in self._agent.peek_steering_messages()]

    def get_follow_up_messages(self) -> list[str]:
        return [getattr(m, "content", str(m)) for m in self._agent.peek_follow_up_messages()]

    def clear_queue(self) -> dict[str, list]:
        steering = self._agent.peek_steering_messages()
        follow_up = self._agent.peek_follow_up_messages()
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
        self._session_manager.append_model_change(model.provider, model.id)
        # Per-model default takes priority; otherwise keep/clamp current level
        self.set_thinking_level(self._get_thinking_level_for_model_switch(model))

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

    def set_thinking_level(self, level: ThinkingLevel, persist: bool = False) -> None:
        """Set thinking level, clamped to model capabilities.

        Always writes a session transcript entry when the level changes.
        When persist=True, also saves the requested level as the global default.
        """
        available = self.get_available_thinking_levels()
        effective = level if level in available else _clamp_thinking_level(level, available)
        is_changing = effective != self._agent.state.thinking_level
        self._agent.set_thinking_level(effective)
        if persist:
            self._settings_manager.set_default_thinking_level(level)
        if is_changing:
            self._session_manager.append_thinking_level_change(effective)

    def _get_thinking_level_for_model_switch(
        self,
        target_model: Model | None = None,
        explicit_level: ThinkingLevel | None = None,
    ) -> ThinkingLevel:
        """Resolve thinking level when switching models. Mirrors TS _getThinkingLevelForModelSwitch."""
        if explicit_level is not None:
            return explicit_level
        if target_model is not None:
            per_model = self._settings_manager.get_model_thinking_level(
                target_model.provider, target_model.id
            )
            if per_model is not None:
                return per_model  # type: ignore[return-value]
        return (
            self._settings_manager.get_default_thinking_level()
            or self.thinking_level
            or "medium"
        )

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
        """Estimate current context size from last valid assistant usage or message lengths."""
        return int(estimate_context_tokens(self._agent.state.messages)["tokens"])

    # ── Session management ────────────────────────────────────────────────────

    async def fork(self, entry_id: str | None = None) -> "AgentSession":
        """
        Fork the session from a specific entry (or current leaf).
        Mirrors fork() in TypeScript.
        """
        branch_point = entry_id
        if not branch_point:
            leaf = self._session_manager.get_leaf_entry()
            branch_point = leaf.id if leaf else None

        if branch_point:
            # Check if entry has a parent; if not, create new session
            entry = self._session_manager.get_entry(branch_point)
            if entry and entry.parent_id:
                return await self.create_branched_session(entry.parent_id)

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
            will_retry = getattr(msg, "stop_reason", "") != "stop"
            if not will_retry:
                # Successful overflow response: compact but do not retry
                await self._run_auto_compaction("overflow", will_retry=False)
                return
            if self._overflow_recovery_attempted:
                # Already tried once — don't loop infinitely
                self._emit({
                    "type": "auto_compaction_end",
                    "result": None,
                    "aborted": False,
                    "willRetry": False,
                    "errorMessage": (
                        "Context overflow recovery failed after one compact-and-retry attempt. "
                        "Try reducing context or switching to a larger-context model."
                    ),
                })
                return
            self._overflow_recovery_attempted = True
            # Remove the error message from agent state (keep in session history)
            messages = self._agent.state.messages
            if messages and getattr(messages[-1], "role", "") == "assistant":
                self._agent.replace_messages(messages[:-1])
            await self._run_auto_compaction("overflow", will_retry=True)
            return

        # Case 3: threshold compaction without retry.
        # For error messages or all-zero usage, estimate from the last valid
        # response so persistent API errors still compact.
        from .compaction.compaction import calculate_context_tokens

        usage = getattr(msg, "usage", None)
        direct = calculate_context_tokens(usage) if usage else 0
        if getattr(msg, "stop_reason", "") == "error" or direct == 0:
            # Without provider usage, estimate.tokens is the pure message-size estimate.
            context_tokens = estimate_context_tokens(self._agent.state.messages)["tokens"]
        else:
            context_tokens = direct
        reserve = settings.get("reserveTokens", 16384)
        if context_window > 0 and context_tokens > context_window - reserve:
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

    def export_to_jsonl(self, output_path: str | None = None) -> str:
        from .session_export import export_session_to_jsonl

        return export_session_to_jsonl(self._session_manager, output_path)

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

    # ── Additional session methods (mirrors TS) ───────────────────────────────

    def dispose(self) -> None:
        """Clean up session resources and event listeners."""
        self._listeners = []

    def set_scoped_models(self, scoped_models: list[dict[str, Model | ThinkingLevel | None]]) -> None:
        """
        Set models available for cycling (Ctrl+P).
        Mirrors setScopedModels() in TypeScript.
        """
        self._scoped_models = scoped_models

    def set_auto_compaction_enabled(self, enabled: bool) -> None:
        """Enable or disable auto-compaction."""
        # SettingsManager doesn't have set_compaction_enabled
        # This is a read-only property from settings files
        # In TS version this calls settingsManager.setCompactionEnabled
        # For now, this is a no-op as settings are file-based
        pass

    @property
    def auto_compaction_enabled(self) -> bool:
        """Whether auto-compaction is currently enabled."""
        return self._settings_manager.get_compaction_settings().get("enabled", True)

    def set_auto_retry_enabled(self, enabled: bool) -> None:
        """Enable or disable auto-retry."""
        # SettingsManager doesn't have set_retry_enabled
        # This is a read-only property from settings files
        # In TS version this calls settingsManager.setRetryEnabled
        # For now, this is a no-op as settings are file-based
        pass

    @property
    def auto_retry_enabled(self) -> bool:
        """Whether auto-retry is currently enabled."""
        return self._settings_manager.get_retry_settings().get("enabled", True)

    async def bind_extensions(self, bindings: dict[str, Any]) -> None:
        """
        Bind extension UI context and handlers.
        Mirrors bindExtensions() in TypeScript.
        This is a stub for future extension support.
        """
        # TODO: Implement extension bindings when extension system is added
        pass

    async def reload(self) -> None:
        """
        Reload configuration and resources.
        Mirrors reload() in TypeScript.
        """
        # Reload settings
        await self._settings_manager.reload()
        # Reset API providers if needed
        # TODO: Add resetApiProviders when available
        pass

    async def execute_bash(
        self,
        command: str,
        on_chunk: Callable[[str], None] | None = None,
        exclude_from_context: bool = False,
    ) -> dict[str, Any]:
        """
        Execute a bash command and return the result.
        Mirrors executeBash() in TypeScript.
        
        Returns dict with:
            - output: str
            - exit_code: int
            - cancelled: bool
            - truncated: bool
            - full_output_path: str | None
        """
        from pi_ai.types import ToolResultMessage
        
        # Apply command prefix if configured
        prefix = self._settings_manager.get_shell_command_prefix()
        resolved_command = f"{prefix}\n{command}" if prefix else command
        
        # Execute bash command
        # TODO: Integrate with actual bash executor when available
        import subprocess
        try:
            proc = subprocess.run(
                resolved_command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.cwd,
                timeout=300,  # 5 minute timeout
            )
            output = proc.stdout + proc.stderr
            exit_code = proc.returncode
            cancelled = False
            truncated = False
            full_output_path = None
            
            if on_chunk:
                on_chunk(output)
        except subprocess.TimeoutExpired:
            output = "Command timed out after 5 minutes"
            exit_code = -1
            cancelled = True
            truncated = False
            full_output_path = None
        except Exception as e:
            output = f"Error executing command: {str(e)}"
            exit_code = -1
            cancelled = False
            truncated = False
            full_output_path = None
        
        result = {
            "output": output,
            "exit_code": exit_code,
            "cancelled": cancelled,
            "truncated": truncated,
            "full_output_path": full_output_path,
        }
        
        # Record result in session
        self.record_bash_result(
            tool_call_id="manual_exec",
            output=output,
            exit_code=exit_code,
        )
        
        return result

    def abort_bash(self) -> None:
        """
        Abort current bash execution.
        Mirrors abortBash() in TypeScript.
        """
        # TODO: Integrate with bash executor abort mechanism
        pass

    def set_session_name(self, name: str) -> None:
        """
        Set a human-readable name for the session.
        Mirrors setSessionName() in TypeScript.
        """
        self._session_manager.append_session_info(name)

    def get_user_messages_for_forking(self) -> list[dict[str, str]]:
        """
        Get all user messages suitable for forking.
        Mirrors getUserMessagesForForking() in TypeScript.
        
        Returns list of dicts with:
            - entry_id: str
            - text: str
        """
        entries = self._session_manager.get_entries()
        result: list[dict[str, str]] = []
        
        for entry in entries:
            if entry.type != "message":
                continue
            msg_data = entry.data.get("message", {})
            if isinstance(msg_data, dict) and msg_data.get("role") == "user":
                text = self._extract_user_message_text(msg_data.get("content", ""))
                if text:
                    result.append({"entry_id": entry.id, "text": text})
        
        return result

    def _extract_user_message_text(self, content: str | list[dict] | Any) -> str:
        """Extract text from user message content."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        return ""

    def has_extension_handlers(self, event_type: str) -> bool:
        """
        Check if any extensions handle the given event type.
        Mirrors hasExtensionHandlers() in TypeScript.
        This is a stub for future extension support.
        """
        # TODO: Implement when extension system is added
        return False


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
