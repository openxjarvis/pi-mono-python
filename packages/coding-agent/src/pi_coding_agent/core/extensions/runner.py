"""
Extension runner - executes extensions and manages lifecycle.

Mirrors core/extensions/runner.ts
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from pi_coding_agent.core.extensions.types import (
    Extension,
    ExtensionActions,
    ExtensionContext,
    ExtensionError,
    ExtensionRuntime,
    InputEvent,
    InputEventResult,
    LoadExtensionsResult,
    ResourcesDiscoverEvent,
    ResourcesDiscoverResult,
    ToolCallEvent,
    ToolCallEventResult,
    ToolResultEvent,
    ToolResultEventResult,
)


_SENTINEL = object()


class ExtensionRunner:
    """Executes extensions and manages their event handler callbacks."""

    def __init__(
        self,
        extensions_result: LoadExtensionsResult,
        context_actions: Any,
        command_context_actions: Any = None,
        on_error: Callable[[ExtensionError], None] | None = None,
    ) -> None:
        self._extensions = extensions_result.extensions
        self._runtime = extensions_result.runtime
        self._context_actions = context_actions
        self._command_context_actions = command_context_actions
        self._on_error = on_error
        self._errors: list[ExtensionError] = []

    def initialize(self, actions: ExtensionActions) -> None:
        """Bind runtime actions so extensions can call API methods."""
        self._runtime.send_message = actions.send_message
        self._runtime.send_user_message = actions.send_user_message
        self._runtime.append_entry = actions.append_entry
        self._runtime.set_session_name = actions.set_session_name
        self._runtime.get_session_name = actions.get_session_name
        self._runtime.set_label = actions.set_label
        self._runtime.get_active_tools = actions.get_active_tools
        self._runtime.get_all_tools = actions.get_all_tools
        self._runtime.set_active_tools = actions.set_active_tools
        self._runtime.get_commands = actions.get_commands
        self._runtime.set_model = actions.set_model
        self._runtime.get_thinking_level = actions.get_thinking_level
        self._runtime.set_thinking_level = actions.set_thinking_level

    def create_context(self) -> ExtensionContext:
        """Create an ExtensionContext bound to current context_actions."""
        return ExtensionContext(
            ui=getattr(self._context_actions, "ui", None),
            has_ui=getattr(self._context_actions, "has_ui", False),
            cwd=getattr(self._context_actions, "cwd", ""),
            session_manager=getattr(self._context_actions, "session_manager", None),
            model_registry=getattr(self._context_actions, "model_registry", None),
            model=getattr(self._context_actions, "get_model", lambda: None)(),
            is_idle=getattr(self._context_actions, "is_idle", lambda: True),
            abort=getattr(self._context_actions, "abort", lambda: None),
            has_pending_messages=getattr(self._context_actions, "has_pending_messages", lambda: False),
            shutdown=getattr(self._context_actions, "shutdown", lambda: None),
            get_context_usage=getattr(self._context_actions, "get_context_usage", lambda: None),
            compact=getattr(self._context_actions, "compact", lambda **kw: None),
            get_system_prompt=getattr(self._context_actions, "get_system_prompt", lambda: ""),
        )

    def has_handlers(self, event: str) -> bool:
        return any(event in ext.handlers for ext in self._extensions)

    def get_all_tools(self) -> dict[str, Any]:
        tools: dict[str, Any] = {}
        for ext in self._extensions:
            for name, tool in ext.tools.items():
                if name not in tools:
                    tools[name] = tool
        return tools

    def get_all_commands(self) -> dict[str, Any]:
        cmds: dict[str, Any] = {}
        for ext in self._extensions:
            for name, cmd in ext.commands.items():
                if name not in cmds:
                    cmds[name] = cmd
        return cmds

    def get_all_flags(self) -> dict[str, Any]:
        flags: dict[str, Any] = {}
        for ext in self._extensions:
            for name, flag in ext.flags.items():
                if name not in flags:
                    flags[name] = flag
        return flags

    async def emit(self, event: Any) -> None:
        """Emit an event to all handler for that event type."""
        event_type = getattr(event, "type", None) or event.get("type", "")
        ctx = self.create_context()
        for ext in self._extensions:
            handlers = ext.handlers.get(event_type, [])
            for handler in handlers:
                try:
                    result = handler(event, ctx)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    self._record_error(ext.path, event_type, e)

    async def emit_tool_call(self, event: ToolCallEvent) -> ToolCallEventResult | None:
        ctx = self.create_context()
        combined: ToolCallEventResult | None = None
        for ext in self._extensions:
            handlers = ext.handlers.get("tool_call", [])
            for handler in handlers:
                try:
                    result = handler(event, ctx)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if result and getattr(result, "block", False):
                        return result
                    if result:
                        combined = result
                except Exception as e:
                    self._record_error(ext.path, "tool_call", e)
        return combined

    async def emit_tool_result(self, event: ToolResultEvent) -> ToolResultEventResult | None:
        ctx = self.create_context()
        combined: ToolResultEventResult | None = None
        for ext in self._extensions:
            handlers = ext.handlers.get("tool_result", [])
            for handler in handlers:
                try:
                    result = handler(event, ctx)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if result:
                        combined = result
                except Exception as e:
                    self._record_error(ext.path, "tool_result", e)
        return combined

    async def emit_input(self, event: InputEvent) -> InputEventResult:
        ctx = self.create_context()
        for ext in self._extensions:
            handlers = ext.handlers.get("input", [])
            for handler in handlers:
                try:
                    result = handler(event, ctx)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, dict):
                        action = result.get("action")
                        if action in ("handled", "transform"):
                            return result
                except Exception as e:
                    self._record_error(ext.path, "input", e)
        return {"action": "continue"}

    async def emit_resources_discover(
        self, event: ResourcesDiscoverEvent
    ) -> ResourcesDiscoverResult:
        ctx = self.create_context()
        combined = ResourcesDiscoverResult()
        for ext in self._extensions:
            handlers = ext.handlers.get("resources_discover", [])
            for handler in handlers:
                try:
                    result = handler(event, ctx)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if result:
                        combined.skill_paths.extend(getattr(result, "skill_paths", []) or [])
                        combined.prompt_paths.extend(getattr(result, "prompt_paths", []) or [])
                        combined.theme_paths.extend(getattr(result, "theme_paths", []) or [])
                except Exception as e:
                    self._record_error(ext.path, "resources_discover", e)
        return combined

    def _record_error(self, extension_path: str, event: str, error: Exception) -> None:
        import traceback
        err = ExtensionError(
            extension_path=extension_path,
            event=event,
            error=str(error),
            stack=traceback.format_exc(),
        )
        self._errors.append(err)
        if self._on_error:
            self._on_error(err)

    @property
    def errors(self) -> list[ExtensionError]:
        return list(self._errors)
