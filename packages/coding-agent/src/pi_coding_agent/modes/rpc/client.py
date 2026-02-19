"""
RPC Client for programmatic access to the coding agent.

Spawns the agent in RPC mode and provides a typed API for all operations.

Mirrors packages/coding-agent/src/modes/rpc/rpc-client.ts
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from typing import Any, Callable

from .types import RpcResponse, RpcSessionState, RpcSlashCommand


RpcEventListener = Callable[[dict[str, Any]], None]


class RpcClientOptions:
    def __init__(
        self,
        cli_path: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        provider: str | None = None,
        model: str | None = None,
        args: list[str] | None = None,
    ) -> None:
        self.cli_path = cli_path
        self.cwd = cwd
        self.env = env or {}
        self.provider = provider
        self.model = model
        self.args = args or []


class RpcClient:
    """
    RPC Client that spawns the coding agent in RPC mode and provides
    a typed async API for all operations.
    """

    def __init__(self, options: RpcClientOptions | None = None) -> None:
        self._options = options or RpcClientOptions()
        self._process: subprocess.Popen | None = None
        self._event_listeners: list[RpcEventListener] = []
        self._pending_requests: dict[str, asyncio.Future[RpcResponse]] = {}
        self._request_id = 0
        self._stderr = ""
        self._reader_task: asyncio.Task | None = None
        self._stdin_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the RPC agent process."""
        if self._process:
            raise RuntimeError("Client already started")

        import os

        cli_path = self._options.cli_path or "dist/cli.py"
        args = [sys.executable, cli_path, "--mode", "rpc"]

        if self._options.provider:
            args += ["--provider", self._options.provider]
        if self._options.model:
            args += ["--model", self._options.model]
        args += self._options.args

        env = {**os.environ, **self._options.env}

        self._process = subprocess.Popen(
            args,
            cwd=self._options.cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Start reader task
        loop = asyncio.get_event_loop()
        self._reader_task = loop.create_task(self._read_loop())

        await asyncio.sleep(0.1)

        if self._process.poll() is not None:
            raise RuntimeError(
                f"Agent process exited immediately with code {self._process.returncode}."
                f" Stderr: {self._stderr}"
            )

    async def stop(self) -> None:
        """Stop the RPC agent process."""
        if not self._process:
            return

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        self._process.terminate()
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self._process.wait),
                timeout=1.0,
            )
        except asyncio.TimeoutError:
            self._process.kill()

        self._process = None
        self._pending_requests.clear()

    def on_event(self, listener: RpcEventListener) -> Callable[[], None]:
        """Subscribe to agent events. Returns unsubscribe function."""
        self._event_listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self._event_listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    def get_stderr(self) -> str:
        return self._stderr

    # =========================================================================
    # Command Methods
    # =========================================================================

    async def prompt(self, message: str, images: list[dict] | None = None) -> None:
        await self._send({"type": "prompt", "message": message, "images": images})

    async def steer(self, message: str, images: list[dict] | None = None) -> None:
        await self._send({"type": "steer", "message": message, "images": images})

    async def follow_up(self, message: str, images: list[dict] | None = None) -> None:
        await self._send({"type": "follow_up", "message": message, "images": images})

    async def abort(self) -> None:
        await self._send({"type": "abort"})

    async def new_session(self, parent_session: str | None = None) -> dict[str, bool]:
        response = await self._send({"type": "new_session", "parentSession": parent_session})
        return self._get_data(response)

    async def get_state(self) -> dict[str, Any]:
        response = await self._send({"type": "get_state"})
        return self._get_data(response)

    async def set_model(self, provider: str, model_id: str) -> dict[str, Any]:
        response = await self._send({"type": "set_model", "provider": provider, "modelId": model_id})
        return self._get_data(response)

    async def cycle_model(self) -> dict[str, Any] | None:
        response = await self._send({"type": "cycle_model"})
        return self._get_data(response)

    async def get_available_models(self) -> list[dict[str, Any]]:
        response = await self._send({"type": "get_available_models"})
        data = self._get_data(response)
        return data.get("models", [])

    async def set_thinking_level(self, level: str) -> None:
        await self._send({"type": "set_thinking_level", "level": level})

    async def cycle_thinking_level(self) -> dict[str, str] | None:
        response = await self._send({"type": "cycle_thinking_level"})
        return self._get_data(response)

    async def set_steering_mode(self, mode: str) -> None:
        await self._send({"type": "set_steering_mode", "mode": mode})

    async def set_follow_up_mode(self, mode: str) -> None:
        await self._send({"type": "set_follow_up_mode", "mode": mode})

    async def compact(self, custom_instructions: str | None = None) -> dict[str, Any]:
        response = await self._send({"type": "compact", "customInstructions": custom_instructions})
        return self._get_data(response)

    async def set_auto_compaction(self, enabled: bool) -> None:
        await self._send({"type": "set_auto_compaction", "enabled": enabled})

    async def set_auto_retry(self, enabled: bool) -> None:
        await self._send({"type": "set_auto_retry", "enabled": enabled})

    async def abort_retry(self) -> None:
        await self._send({"type": "abort_retry"})

    async def bash(self, command: str) -> dict[str, Any]:
        response = await self._send({"type": "bash", "command": command})
        return self._get_data(response)

    async def abort_bash(self) -> None:
        await self._send({"type": "abort_bash"})

    async def get_session_stats(self) -> dict[str, Any]:
        response = await self._send({"type": "get_session_stats"})
        return self._get_data(response)

    async def export_html(self, output_path: str | None = None) -> dict[str, str]:
        response = await self._send({"type": "export_html", "outputPath": output_path})
        return self._get_data(response)

    async def switch_session(self, session_path: str) -> dict[str, bool]:
        response = await self._send({"type": "switch_session", "sessionPath": session_path})
        return self._get_data(response)

    async def fork(self, entry_id: str) -> dict[str, Any]:
        response = await self._send({"type": "fork", "entryId": entry_id})
        return self._get_data(response)

    async def get_fork_messages(self) -> list[dict[str, str]]:
        response = await self._send({"type": "get_fork_messages"})
        data = self._get_data(response)
        return data.get("messages", [])

    async def get_last_assistant_text(self) -> str | None:
        response = await self._send({"type": "get_last_assistant_text"})
        data = self._get_data(response)
        return data.get("text")

    async def set_session_name(self, name: str) -> None:
        await self._send({"type": "set_session_name", "name": name})

    async def get_messages(self) -> list[dict[str, Any]]:
        response = await self._send({"type": "get_messages"})
        data = self._get_data(response)
        return data.get("messages", [])

    async def get_commands(self) -> list[dict[str, Any]]:
        response = await self._send({"type": "get_commands"})
        data = self._get_data(response)
        return data.get("commands", [])

    # =========================================================================
    # Helpers
    # =========================================================================

    async def wait_for_idle(self, timeout: float = 60.0) -> None:
        """Wait for agent to become idle (agent_end event)."""
        future: asyncio.Future[None] = asyncio.get_event_loop().create_future()

        def listener(event: dict[str, Any]) -> None:
            if event.get("type") == "agent_end" and not future.done():
                future.set_result(None)

        unsubscribe = self.on_event(listener)
        try:
            await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Timeout waiting for agent to become idle. Stderr: {self._stderr}"
            )
        finally:
            unsubscribe()

    async def collect_events(self, timeout: float = 60.0) -> list[dict[str, Any]]:
        """Collect events until agent becomes idle."""
        events: list[dict[str, Any]] = []
        future: asyncio.Future[list] = asyncio.get_event_loop().create_future()

        def listener(event: dict[str, Any]) -> None:
            events.append(event)
            if event.get("type") == "agent_end" and not future.done():
                future.set_result(events)

        unsubscribe = self.on_event(listener)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Timeout collecting events. Stderr: {self._stderr}"
            )
        finally:
            unsubscribe()

    async def prompt_and_wait(
        self,
        message: str,
        images: list[dict] | None = None,
        timeout: float = 60.0,
    ) -> list[dict[str, Any]]:
        """Send prompt and wait for completion, returning all events."""
        events_coro = self.collect_events(timeout)
        await self.prompt(message, images)
        return await events_coro

    # =========================================================================
    # Internal
    # =========================================================================

    async def _read_loop(self) -> None:
        """Background task reading stdout from the agent process."""
        loop = asyncio.get_event_loop()
        while self._process and self._process.stdout:
            line_bytes = await loop.run_in_executor(None, self._process.stdout.readline)
            if not line_bytes:
                break
            line = line_bytes.decode().strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                self._handle_line(data)
            except json.JSONDecodeError:
                pass

            # Collect stderr
            if self._process and self._process.stderr:
                try:
                    chunk = self._process.stderr.read1(4096)  # type: ignore[attr-defined]
                    if chunk:
                        self._stderr += chunk.decode(errors="replace")
                except Exception:
                    pass

    def _handle_line(self, data: dict[str, Any]) -> None:
        if data.get("type") == "response" and data.get("id") and data["id"] in self._pending_requests:
            future = self._pending_requests.pop(data["id"])
            if not future.done():
                future.set_result(data)
            return

        for listener in self._event_listeners:
            listener(data)

    async def _send(self, command: dict[str, Any]) -> dict[str, Any]:
        if not self._process or not self._process.stdin:
            raise RuntimeError("Client not started")

        self._request_id += 1
        req_id = f"req_{self._request_id}"
        full_command = {**command, "id": req_id}

        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()

        async def _do_send() -> dict[str, Any]:
            self._pending_requests[req_id] = future
            line = json.dumps(full_command) + "\n"
            await loop.run_in_executor(None, self._process.stdin.write, line.encode())
            await loop.run_in_executor(None, self._process.stdin.flush)
            try:
                return await asyncio.wait_for(future, timeout=30.0)
            except asyncio.TimeoutError:
                self._pending_requests.pop(req_id, None)
                raise TimeoutError(
                    f"Timeout waiting for response to {command['type']}. Stderr: {self._stderr}"
                )

        return await _do_send()

    def _get_data(self, response: dict[str, Any]) -> Any:
        if not response.get("success"):
            raise RuntimeError(response.get("error", "Unknown error"))
        return response.get("data")
