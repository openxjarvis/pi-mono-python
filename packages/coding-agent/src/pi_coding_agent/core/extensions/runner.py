"""
Extension runner — mirrors packages/coding-agent/src/core/extensions/runner.ts

Manages loaded extensions and dispatches events to their handlers.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

from .types import (
    Extension,
    ExtensionContext,
    RegisteredCommand,
    RegisteredTool,
)


class ExtensionRunner:
    """
    Runs extensions and dispatches events.
    Mirrors ExtensionRunner in TypeScript.
    """

    def __init__(
        self,
        extensions: list[Extension],
        cwd: str = "",
        session_id: str = "",
    ) -> None:
        self._extensions = extensions
        self._cwd = cwd
        self._session_id = session_id

    @property
    def extensions(self) -> list[Extension]:
        return self._extensions

    def has_handlers(self, event_type: str) -> bool:
        """Check if any extension has handlers for this event type."""
        return any(
            event_type in ext.handlers and len(ext.handlers[event_type]) > 0
            for ext in self._extensions
        )

    async def emit(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Emit an event to all extensions."""
        event_type = event.get("type", "")
        combined_result: dict[str, Any] | None = None

        for ext in self._extensions:
            handlers = ext.handlers.get(event_type, [])
            for handler in handlers:
                try:
                    ctx = ExtensionContext(
                        cwd=self._cwd,
                        session_id=self._session_id,
                    )
                    result = handler(ctx, event)
                    if inspect.isawaitable(result):
                        result = await result
                    if isinstance(result, dict):
                        if combined_result is None:
                            combined_result = {}
                        combined_result.update(result)
                except Exception:
                    pass

        return combined_result

    async def emit_input(
        self,
        text: str,
        images: list[Any] | None = None,
        source: str = "interactive",
    ) -> dict[str, Any]:
        """Emit input event. Returns {action, text, images}."""
        result = await self.emit({
            "type": "input",
            "text": text,
            "images": images,
            "source": source,
        })
        if result and result.get("action") in ("handled", "transform"):
            return result
        return {"action": "pass", "text": text, "images": images}

    async def emit_before_agent_start(
        self,
        prompt: str,
        images: list[Any] | None,
        system_prompt: str,
    ) -> dict[str, Any] | None:
        """Emit before_agent_start event."""
        return await self.emit({
            "type": "before_agent_start",
            "prompt": prompt,
            "images": images,
            "systemPrompt": system_prompt,
        })

    async def emit_context(self, messages: list[Any]) -> list[Any]:
        """
        Emit context event for message modification.
        Uses chain passing - each handler receives the output of the previous handler.
        Mirrors ExtensionRunner.emitContext() in TypeScript.
        """
        current_messages = messages
        
        for ext in self._extensions:
            handlers = ext.handlers.get("context", [])
            for handler in handlers:
                try:
                    ctx = ExtensionContext(
                        cwd=self._cwd,
                        session_id=self._session_id,
                    )
                    # Pass current_messages to handler
                    result = handler(ctx, {"type": "context", "messages": current_messages})
                    if inspect.isawaitable(result):
                        result = await result
                    
                    # Update current_messages with handler result (chain passing)
                    if isinstance(result, dict) and "messages" in result:
                        current_messages = result["messages"]
                except Exception:
                    # On error, skip this handler but continue chain
                    pass
        
        return current_messages

    async def emit_tool_call(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Emit tool_call event (before tool execution)."""
        return await self.emit({"type": "tool_call", **event})

    async def emit_tool_result(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Emit tool_result event (after tool execution)."""
        return await self.emit({"type": "tool_result", **event})

    async def emit_resources_discover(
        self,
        cwd: str,
        reason: str = "init",
    ) -> dict[str, list[str]]:
        """Emit resources_discover event."""
        result = await self.emit({
            "type": "resources_discover",
            "cwd": cwd,
            "reason": reason,
        })
        return {
            "skillPaths": (result or {}).get("skillPaths", []),
            "promptPaths": (result or {}).get("promptPaths", []),
            "themePaths": (result or {}).get("themePaths", []),
        }

    def get_all_registered_tools(self) -> list[RegisteredTool]:
        """Get all tools registered by extensions."""
        tools: list[RegisteredTool] = []
        for ext in self._extensions:
            tools.extend(ext.tools.values())
        return tools

    def get_tool_definition(self, tool_name: str) -> RegisteredTool | None:
        """Get a specific tool's definition."""
        for ext in self._extensions:
            if tool_name in ext.tools:
                return ext.tools[tool_name]
        return None

    def get_all_commands(self) -> list[RegisteredCommand]:
        """Get all commands registered by extensions."""
        commands: list[RegisteredCommand] = []
        for ext in self._extensions:
            commands.extend(ext.commands.values())
        return commands

    def get_command(self, name: str) -> RegisteredCommand | None:
        """Get a specific command."""
        for ext in self._extensions:
            if name in ext.commands:
                return ext.commands[name]
        return None

    async def execute_command(self, name: str, args: str = "") -> Any:
        """Execute a registered command."""
        cmd = self.get_command(name)
        if not cmd:
            raise ValueError(f"Command not found: {name}")
        result = cmd.execute(args)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def emit_before_provider_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Emit before_provider_headers. Returns transformed headers."""
        result = await self.emit({"type": "before_provider_headers", "headers": headers})
        if isinstance(result, dict) and isinstance(result.get("headers"), dict):
            return result["headers"]
        return headers

    async def emit_before_provider_request(self, payload: Any) -> Any:
        """Emit before_provider_request. Returns transformed payload."""
        result = await self.emit({"type": "before_provider_request", "payload": payload})
        if isinstance(result, dict) and "payload" in result:
            return result["payload"]
        return payload

    async def emit_after_provider_response(self, status: Any, headers: dict[str, str] | None = None) -> None:
        """Emit after_provider_response."""
        await self.emit({
            "type": "after_provider_response",
            "status": status,
            "headers": headers or {},
        })

    async def shutdown(self) -> None:
        """Emit session_shutdown to all extensions."""
        await self.emit({"type": "session_shutdown"})
