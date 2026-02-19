"""
RPC mode: Headless operation with JSON stdin/stdout protocol.

Used for embedding the agent in other applications.
Receives commands as JSON on stdin, outputs events and responses as JSON on stdout.

Mirrors packages/coding-agent/src/modes/rpc/rpc-mode.ts
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from typing import TYPE_CHECKING, Any

from .types import (
    RpcCommand,
    RpcExtensionUIRequest,
    RpcExtensionUIRequestConfirm,
    RpcExtensionUIRequestEditor,
    RpcExtensionUIRequestInput,
    RpcExtensionUIRequestNotify,
    RpcExtensionUIRequestSelect,
    RpcExtensionUIRequestSetEditorText,
    RpcExtensionUIRequestSetStatus,
    RpcExtensionUIRequestSetTitle,
    RpcExtensionUIRequestSetWidget,
    RpcExtensionUIResponse,
    RpcResponse,
    RpcResponseError,
    RpcResponseSuccess,
    RpcSessionState,
    RpcSlashCommand,
)

if TYPE_CHECKING:
    from pi_coding_agent.core.agent_session import AgentSession
    from pi_coding_agent.core.extensions.types import ExtensionUIContext


def _output(obj: dict[str, Any]) -> None:
    print(json.dumps(obj), flush=True)


def _success(cmd_id: str | None, command: str, data: Any = None) -> RpcResponseSuccess:
    return RpcResponseSuccess(id=cmd_id, command=command, data=data)


def _error(cmd_id: str | None, command: str, message: str) -> RpcResponseError:
    return RpcResponseError(id=cmd_id, command=command, error=message)


def _create_extension_ui_context(
    pending_requests: dict[str, asyncio.Future[Any]],
    output_fn: Any,
) -> "ExtensionUIContext":
    """Create an ExtensionUIContext that uses the RPC protocol."""
    from pi_coding_agent.core.extensions.types import ExtensionUIContext

    class RpcExtensionUIContextImpl(ExtensionUIContext):
        async def select(self, title: str, options: list[str], opts: Any = None) -> str | None:
            if opts and getattr(opts, "signal", None) and getattr(opts.signal, "aborted", False):
                return None
            req_id = str(uuid.uuid4())
            future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
            pending_requests[req_id] = future
            output_fn({
                "type": "extension_ui_request", "id": req_id,
                "method": "select", "title": title, "options": options,
                "timeout": getattr(opts, "timeout", None),
            })
            try:
                response = await asyncio.wait_for(future, timeout=getattr(opts, "timeout", None))
                if isinstance(response, dict) and response.get("cancelled"):
                    return None
                return response.get("value")
            except asyncio.TimeoutError:
                pending_requests.pop(req_id, None)
                return None

        async def confirm(self, title: str, message: str, opts: Any = None) -> bool:
            req_id = str(uuid.uuid4())
            future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
            pending_requests[req_id] = future
            output_fn({
                "type": "extension_ui_request", "id": req_id,
                "method": "confirm", "title": title, "message": message,
                "timeout": getattr(opts, "timeout", None),
            })
            try:
                response = await asyncio.wait_for(future, timeout=getattr(opts, "timeout", None))
                if isinstance(response, dict) and response.get("cancelled"):
                    return False
                return bool(response.get("confirmed", False))
            except asyncio.TimeoutError:
                pending_requests.pop(req_id, None)
                return False

        async def input(self, title: str, placeholder: str | None = None, opts: Any = None) -> str | None:
            req_id = str(uuid.uuid4())
            future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
            pending_requests[req_id] = future
            output_fn({
                "type": "extension_ui_request", "id": req_id,
                "method": "input", "title": title, "placeholder": placeholder,
                "timeout": getattr(opts, "timeout", None),
            })
            try:
                response = await asyncio.wait_for(future, timeout=getattr(opts, "timeout", None))
                if isinstance(response, dict) and response.get("cancelled"):
                    return None
                return response.get("value")
            except asyncio.TimeoutError:
                pending_requests.pop(req_id, None)
                return None

        def notify(self, message: str, notify_type: str | None = None) -> None:
            output_fn({
                "type": "extension_ui_request",
                "id": str(uuid.uuid4()),
                "method": "notify", "message": message, "notifyType": notify_type,
            })

        def on_terminal_input(self, handler: Any) -> Any:
            return lambda: None

        def set_status(self, key: str, text: str | None) -> None:
            output_fn({
                "type": "extension_ui_request",
                "id": str(uuid.uuid4()),
                "method": "setStatus", "statusKey": key, "statusText": text,
            })

        def set_working_message(self, message: str | None = None) -> None:
            pass

        def set_widget(self, key: str, content: Any, options: Any = None) -> None:
            if content is None or isinstance(content, list):
                output_fn({
                    "type": "extension_ui_request",
                    "id": str(uuid.uuid4()),
                    "method": "setWidget", "widgetKey": key,
                    "widgetLines": content,
                    "widgetPlacement": getattr(options, "placement", None) if options else None,
                })

        def set_footer(self, factory: Any) -> None:
            pass

        def set_header(self, factory: Any) -> None:
            pass

        def set_title(self, title: str) -> None:
            output_fn({
                "type": "extension_ui_request",
                "id": str(uuid.uuid4()),
                "method": "setTitle", "title": title,
            })

        async def custom(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def paste_to_editor(self, text: str) -> None:
            self.set_editor_text(text)

        def set_editor_text(self, text: str) -> None:
            output_fn({
                "type": "extension_ui_request",
                "id": str(uuid.uuid4()),
                "method": "set_editor_text", "text": text,
            })

        def get_editor_text(self) -> str:
            return ""

        async def editor(self, title: str, prefill: str | None = None) -> str | None:
            req_id = str(uuid.uuid4())
            future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
            pending_requests[req_id] = future
            output_fn({"type": "extension_ui_request", "id": req_id, "method": "editor", "title": title, "prefill": prefill})
            response = await future
            if isinstance(response, dict) and response.get("cancelled"):
                return None
            return response.get("value")

        def set_editor_component(self, *args: Any, **kwargs: Any) -> None:
            pass

        @property
        def theme(self) -> Any:
            return {}

        def get_all_themes(self) -> list[Any]:
            return []

        def get_theme(self, name: str) -> Any:
            return None

        def set_theme(self, theme_val: Any) -> dict[str, Any]:
            return {"success": False, "error": "Theme switching not supported in RPC mode"}

        def get_tools_expanded(self) -> bool:
            return False

        def set_tools_expanded(self, expanded: bool) -> None:
            pass

    return RpcExtensionUIContextImpl()


async def run_rpc_mode(session: "AgentSession") -> None:
    """
    Run in RPC mode.
    Listens for JSON commands on stdin, outputs events and responses on stdout.
    """
    pending_extension_requests: dict[str, asyncio.Future[Any]] = {}
    shutdown_requested = False

    def output(obj: Any) -> None:
        if hasattr(obj, "model_dump"):
            _output(obj.model_dump(exclude_none=True))
        elif isinstance(obj, dict):
            _output(obj)
        else:
            _output(obj)

    ui_ctx = _create_extension_ui_context(pending_extension_requests, _output)

    await session.bind_extensions({
        "uiContext": ui_ctx,
        "commandContextActions": {
            "waitForIdle": lambda: session.agent.wait_for_idle(),
            "newSession": lambda opts=None: session.new_session(opts),
            "fork": lambda entry_id: session.fork(entry_id),
            "navigateTree": lambda target_id, opts=None: session.navigate_tree(target_id, opts),
            "switchSession": lambda path: session.switch_session(path),
            "reload": lambda: session.reload(),
        },
        "shutdownHandler": lambda: None,  # shutdown_requested set below
        "onError": lambda err: output({"type": "extension_error", **err}),
    })

    # Forward all agent events as JSON
    session.subscribe(lambda event: output(event))

    async def handle_command(command: dict[str, Any]) -> RpcResponse:
        cmd_id = command.get("id")
        cmd_type = command.get("type", "")

        if cmd_type == "prompt":
            asyncio.ensure_future(
                session.prompt(
                    command["message"],
                    images=command.get("images"),
                    streaming_behavior=command.get("streamingBehavior"),
                    source="rpc",
                )
            )
            return _success(cmd_id, "prompt")

        elif cmd_type == "steer":
            await session.steer(command["message"], command.get("images"))
            return _success(cmd_id, "steer")

        elif cmd_type == "follow_up":
            await session.follow_up(command["message"], command.get("images"))
            return _success(cmd_id, "follow_up")

        elif cmd_type == "abort":
            await session.abort()
            return _success(cmd_id, "abort")

        elif cmd_type == "new_session":
            opts = {"parentSession": command["parentSession"]} if command.get("parentSession") else None
            cancelled = not await session.new_session(opts)
            return _success(cmd_id, "new_session", {"cancelled": cancelled})

        elif cmd_type == "get_state":
            state = RpcSessionState(
                model=session.model,
                thinkingLevel=session.thinking_level,
                isStreaming=session.is_streaming,
                isCompacting=session.is_compacting,
                steeringMode=session.steering_mode,
                followUpMode=session.follow_up_mode,
                sessionFile=session.session_file,
                sessionId=session.session_id,
                sessionName=session.session_name,
                autoCompactionEnabled=session.auto_compaction_enabled,
                messageCount=len(session.messages),
                pendingMessageCount=session.pending_message_count,
            )
            return _success(cmd_id, "get_state", state.model_dump())

        elif cmd_type == "set_model":
            models = await session.model_registry.get_available()
            model = next(
                (m for m in models if m.get("provider") == command["provider"] and m.get("id") == command["modelId"]),
                None,
            )
            if not model:
                return _error(cmd_id, "set_model", f"Model not found: {command['provider']}/{command['modelId']}")
            await session.set_model(model)
            return _success(cmd_id, "set_model", model)

        elif cmd_type == "cycle_model":
            result = await session.cycle_model()
            return _success(cmd_id, "cycle_model", result)

        elif cmd_type == "get_available_models":
            models = await session.model_registry.get_available()
            return _success(cmd_id, "get_available_models", {"models": models})

        elif cmd_type == "set_thinking_level":
            session.set_thinking_level(command["level"])
            return _success(cmd_id, "set_thinking_level")

        elif cmd_type == "cycle_thinking_level":
            level = session.cycle_thinking_level()
            return _success(cmd_id, "cycle_thinking_level", {"level": level} if level else None)

        elif cmd_type == "set_steering_mode":
            session.set_steering_mode(command["mode"])
            return _success(cmd_id, "set_steering_mode")

        elif cmd_type == "set_follow_up_mode":
            session.set_follow_up_mode(command["mode"])
            return _success(cmd_id, "set_follow_up_mode")

        elif cmd_type == "compact":
            result = await session.compact(command.get("customInstructions"))
            return _success(cmd_id, "compact", result)

        elif cmd_type == "set_auto_compaction":
            session.set_auto_compaction_enabled(command["enabled"])
            return _success(cmd_id, "set_auto_compaction")

        elif cmd_type == "set_auto_retry":
            session.set_auto_retry_enabled(command["enabled"])
            return _success(cmd_id, "set_auto_retry")

        elif cmd_type == "abort_retry":
            session.abort_retry()
            return _success(cmd_id, "abort_retry")

        elif cmd_type == "bash":
            result = await session.execute_bash(command["command"])
            return _success(cmd_id, "bash", result)

        elif cmd_type == "abort_bash":
            session.abort_bash()
            return _success(cmd_id, "abort_bash")

        elif cmd_type == "get_session_stats":
            stats = session.get_session_stats()
            return _success(cmd_id, "get_session_stats", stats)

        elif cmd_type == "export_html":
            path = await session.export_to_html(command.get("outputPath"))
            return _success(cmd_id, "export_html", {"path": path})

        elif cmd_type == "switch_session":
            cancelled = not await session.switch_session(command["sessionPath"])
            return _success(cmd_id, "switch_session", {"cancelled": cancelled})

        elif cmd_type == "fork":
            result = await session.fork(command["entryId"])
            return _success(cmd_id, "fork", {"text": result.get("selectedText", ""), "cancelled": result.get("cancelled", False)})

        elif cmd_type == "get_fork_messages":
            messages = session.get_user_messages_for_forking()
            return _success(cmd_id, "get_fork_messages", {"messages": messages})

        elif cmd_type == "get_last_assistant_text":
            text = session.get_last_assistant_text()
            return _success(cmd_id, "get_last_assistant_text", {"text": text})

        elif cmd_type == "set_session_name":
            name = command.get("name", "").strip()
            if not name:
                return _error(cmd_id, "set_session_name", "Session name cannot be empty")
            session.set_session_name(name)
            return _success(cmd_id, "set_session_name")

        elif cmd_type == "get_messages":
            return _success(cmd_id, "get_messages", {"messages": session.messages})

        elif cmd_type == "get_commands":
            commands: list[RpcSlashCommand] = []
            runner = getattr(session, "extension_runner", None)
            if runner:
                for cmd_info in runner.get_registered_commands_with_paths():
                    commands.append(RpcSlashCommand(
                        name=cmd_info["command"]["name"],
                        description=cmd_info["command"].get("description"),
                        source="extension",
                        path=cmd_info.get("extensionPath"),
                    ))
            for template in getattr(session, "prompt_templates", []):
                commands.append(RpcSlashCommand(
                    name=template.name,
                    description=getattr(template, "description", None),
                    source="prompt",
                    location=getattr(template, "source", None),
                    path=getattr(template, "file_path", None),
                ))
            resource_loader = getattr(session, "resource_loader", None)
            if resource_loader:
                for skill in resource_loader.get_skills().skills:
                    commands.append(RpcSlashCommand(
                        name=f"skill:{skill.name}",
                        description=getattr(skill, "description", None),
                        source="skill",
                        location=getattr(skill, "source", None),
                        path=getattr(skill, "file_path", None),
                    ))
            return _success(cmd_id, "get_commands", {"commands": [c.model_dump(exclude_none=True) for c in commands]})

        else:
            return _error(None, cmd_type, f"Unknown command: {cmd_type}")

    # Read lines from stdin asynchronously
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        try:
            line_bytes = await reader.readline()
            if not line_bytes:
                break
            line = line_bytes.decode().strip()
            if not line:
                continue

            parsed = json.loads(line)

            # Handle extension UI responses
            if parsed.get("type") == "extension_ui_response":
                req_id = parsed.get("id")
                if req_id and req_id in pending_extension_requests:
                    fut = pending_extension_requests.pop(req_id)
                    if not fut.done():
                        fut.set_result(parsed)
                continue

            response = await handle_command(parsed)
            output(response)

        except json.JSONDecodeError as e:
            output(_error(None, "parse", f"Failed to parse command: {e}"))
        except Exception as e:  # noqa: BLE001
            output(_error(None, "error", str(e)))
