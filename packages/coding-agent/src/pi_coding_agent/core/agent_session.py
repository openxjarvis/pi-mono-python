"""
AgentSession — mirrors packages/coding-agent/src/core/agent-session.ts

Central class managing agent lifecycle, session persistence, tools, and events.
"""
from __future__ import annotations

import asyncio
import html
import os
import time
from typing import Any, Callable

from pi_agent import Agent, AgentOptions
from pi_agent.types import (
    AgentEvent,
    AgentMessage,
    AgentTool,
    ThinkingLevel,
)
from pi_ai import get_model
from pi_ai.types import ImageContent, Model, UserMessage

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


class AgentSession:
    """
    Manages an agent session with persistence, tools, and events.
    Mirrors AgentSession in TypeScript.
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
    ) -> None:
        self.cwd = cwd or os.getcwd()
        self._settings = settings or Settings()
        self._auth_storage = auth_storage or AuthStorage()
        self._model_registry = model_registry or ModelRegistry()

        # Use provided session_manager or create a new one
        if session_manager is not None:
            self._session_manager = session_manager
        else:
            self._session_manager = SessionManager.create(cwd=self.cwd)

        # Session ID is always derived from the manager
        self.session_id = self._session_manager.get_session_id()

        # Build tools
        tools = self._build_tools()

        # Resolve model
        resolved_model = model or self._resolve_default_model()

        # Build system prompt
        system_prompt = build_system_prompt(self.cwd, selected_tools=[t.name for t in tools])

        # Create the underlying Agent
        opts = AgentOptions(
            get_api_key=self._resolve_api_key,
        )
        self._agent = Agent(opts)
        self._agent.set_model(resolved_model)
        self._agent.set_system_prompt(system_prompt)
        self._agent.set_tools(tools)
        self._agent.set_thinking_level(self._settings.thinking_level)

        self._listeners: list[Callable[[AgentEvent], None]] = []
        self._agent.subscribe(self._on_agent_event)

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

    def _resolve_default_model(self) -> Model:
        """Resolve the default model from settings."""
        try:
            resolved = self._model_registry.resolve_model(
                model_id=self._settings.model_id,
                provider=self._settings.provider,
            )
            # If settings pin a model/provider without usable auth, prefer an
            # actually-authenticated provider to avoid silent "no response" UX.
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
            # Emergency fallback: prefer Google if key is available, else Anthropic
            import os
            if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
                return get_model("google", "gemini-2.0-flash")
            return get_model("anthropic", "claude-3-5-sonnet-20241022")

    async def _resolve_api_key(self, provider: str) -> str | None:
        """Resolve API key for a provider."""
        return self._auth_storage.resolve_api_key(provider)

    def _on_agent_event(self, event: AgentEvent) -> None:
        """Handle agent events — persist messages and notify listeners."""
        # Persist messages on agent_end
        if event.type == "agent_end":
            for msg in event.messages:
                if hasattr(msg, "role"):
                    self._session_manager.append_message(_message_to_dict(msg))

        # Notify external listeners
        for listener in self._listeners:
            listener(event)

    def subscribe(self, fn: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """Subscribe to session events. Returns unsubscribe function."""
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn) if fn in self._listeners else None

    # ── Agent control ─────────────────────────────────────────────────────────

    async def prompt(
        self,
        message: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
        source: str | None = None,  # "interactive" | "print" | etc. (ignored, for compat)
    ) -> None:
        """Send a prompt to the agent."""
        # Check for auto-compaction
        if self._settings.auto_compact:
            await self._maybe_compact()
        await self._agent.prompt(message, images)

    async def steer(self, message: AgentMessage) -> None:
        """Queue a steering message."""
        self._agent.steer(message)

    async def follow_up(self, message: AgentMessage) -> None:
        """Queue a follow-up message."""
        self._agent.follow_up(message)

    def abort(self) -> None:
        """Abort the current operation."""
        self._agent.abort()

    async def wait_for_idle(self) -> None:
        """Wait until the agent is idle."""
        await self._agent.wait_for_idle()

    # ── Model management ──────────────────────────────────────────────────────

    def set_model(self, model: Model) -> None:
        """Switch the active model."""
        self._agent.set_model(model)
        self._session_manager.append_model_change(model.id, model.provider)

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        """Set the thinking/reasoning level."""
        self._agent.set_thinking_level(level)
        self._session_manager.append_thinking_level_change(level)

    # ── Session management ────────────────────────────────────────────────────

    def fork(self) -> "AgentSession":
        """Create a fork of the current session."""
        # Fork the session using SessionManager.fork_from
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
        """Get information about the current session."""
        return {
            "session_id": self.session_id,
            "cwd": self.cwd,
            "model": self._agent.state.model.id if self._agent.state.model else None,
            "message_count": len(self._agent.state.messages),
            "is_streaming": self._agent.state.is_streaming,
        }

    # ── Compaction ────────────────────────────────────────────────────────────

    async def compact(self) -> str:
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
            # Store compaction with a reference to the first kept entry
            first_kept_id = str(len(messages)) if messages else "0"
            self._session_manager.append_compaction(summary, first_kept_id)

        return summary

    async def _maybe_compact(self) -> None:
        """Auto-compact if context is getting full."""
        model = self._agent.state.model
        if not model:
            return
        messages = self._agent.state.messages
        if should_compact(
            messages,
            model.context_window,
            self._settings.compact_threshold,
        ):
            await self.compact()

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

    @property
    def state(self):
        """Get the underlying agent state."""
        return self._agent.state

    @property
    def model_registry(self):
        """Expose model registry for CLI/model listing."""
        return self._model_registry

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    @property
    def model(self) -> Model | None:
        return self._agent.state.model


def _message_to_dict(msg: Any) -> dict[str, Any]:
    """Convert a message to a dict for persistence."""
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    return {"role": getattr(msg, "role", "unknown")}
