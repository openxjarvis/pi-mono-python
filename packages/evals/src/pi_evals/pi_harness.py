from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pi_ai.utils.text import content_text
from pi_coding_agent.core.agent_session_services import create_agent_session_from_services, create_agent_session_services
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager

from .pytest_evals.artifacts import PI_SESSION_SNAPSHOT_ARTIFACT

PiCodingAgentInput = str | list[dict[str, str]]


def resolve_model_selection(
    explicit_model: dict[str, str] | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    env = environment if environment is not None else dict(os.environ)
    provider = ((explicit_model or {}).get("provider") or env.get("PI_PROVIDER") or "").strip()
    model_id = ((explicit_model or {}).get("id") or env.get("PI_MODEL") or "").strip()
    if not provider or not model_id:
        raise RuntimeError("Select a harness model explicitly or set both PI_PROVIDER and PI_MODEL as defaults.")
    return {"provider": provider, "id": model_id}


def to_transcript_events(messages: list[Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for message in messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role == "user":
            events.append({"type": "message", "role": "user", "content": content_text(content)})
        elif role == "assistant":
            text = content_text(content)
            if text:
                events.append({"type": "message", "role": "assistant", "content": text})
            for part in content or []:
                kind = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                if kind == "toolCall":
                    events.append(
                        {
                            "type": "tool_call",
                            "id": part.get("id") if isinstance(part, dict) else part.id,
                            "name": part.get("name") if isinstance(part, dict) else part.name,
                            "arguments": part.get("arguments") if isinstance(part, dict) else getattr(part, "arguments", {}),
                        }
                    )
        elif role in {"tool", "toolResult"}:
            text = content_text(content)
            events.append(
                {
                    "type": "tool_result",
                    "toolCallId": getattr(message, "tool_call_id", getattr(message, "toolCallId", None)),
                    "name": getattr(message, "tool_name", getattr(message, "toolName", None)),
                    "content": text,
                }
            )
    return events


async def _prompt_agent(session: Any, text: str) -> str:
    previous = len(session.messages)
    await session.prompt(text)
    assistant = next((message for message in reversed(session.messages[previous:]) if getattr(message, "role", None) == "assistant"), None)
    if assistant is None:
        raise RuntimeError("Agent run completed without an assistant message.")
    output = session.get_last_assistant_text() if hasattr(session, "get_last_assistant_text") else content_text(assistant.content)
    if not output:
        raise RuntimeError("Agent run produced no assistant text.")
    return output


async def run_pi_coding_agent(
    input_value: PiCodingAgentInput,
    set_artifact: Callable[[str, Any], None],
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = options or {}
    started = time.perf_counter()
    selection = resolve_model_selection(options.get("model"))
    model_runtime = await ModelRuntime.create()
    model = model_runtime.get_model(selection["provider"], selection["id"])
    if model is None:
        raise RuntimeError(f"Eval model not found: {selection['provider']}/{selection['id']}")

    root = Path(tempfile.mkdtemp(prefix="pi-eval-"))
    cwd = root / "workspace"
    agent_dir = root / "agent"
    session = None
    session_manager = None
    try:
        cwd.mkdir()
        agent_dir.mkdir()
        services = await create_agent_session_services(
            {
                "cwd": str(cwd),
                "agentDir": str(agent_dir),
                "modelRuntime": model_runtime,
                "settingsManager": SettingsManager.in_memory() if hasattr(SettingsManager, "in_memory") else SettingsManager.inMemory(),
            }
        )
        session_manager = SessionManager.create(str(cwd), str(root / "sessions"))
        set_artifact("runId", session_manager.get_session_id() if hasattr(session_manager, "get_session_id") else getattr(session_manager, "session_id", "eval"))
        created = await create_agent_session_from_services(
            {
                "services": services,
                "sessionManager": session_manager,
                "model": model,
                "thinkingLevel": "off",
                "noTools": options.get("noTools") or options.get("no_tools"),
            }
        )
        session = created["session"] if isinstance(created, dict) else created.session
        steps = [{"type": "prompt", "content": input_value}] if isinstance(input_value, str) else input_value
        response = None
        for step in steps:
            if step["type"] == "prompt":
                response = await _prompt_agent(session, step["content"])
            else:
                await session.reload()
        if response is None:
            raise RuntimeError("Pi eval input must include at least one prompt step.")
        output = await options["output"]({"response": response, "session": session}) if "output" in options else response
        stats = session.get_session_stats() if hasattr(session, "get_session_stats") else None
        return {
            "output": output,
            "events": to_transcript_events(session.messages),
            "usage": {
                "provider": model.provider,
                "model": model.id,
                "inputTokens": getattr(getattr(stats, "tokens", None), "input", 0) if stats else 0,
                "outputTokens": getattr(getattr(stats, "tokens", None), "output", 0) if stats else 0,
                "totalTokens": getattr(getattr(stats, "tokens", None), "total", 0) if stats else 0,
            },
            "timings": {"totalMs": (time.perf_counter() - started) * 1000},
        }
    finally:
        if session_manager is not None:
            session_path = session_manager.get_session_file() if hasattr(session_manager, "get_session_file") else None
            if session_path and Path(session_path).exists():
                set_artifact(PI_SESSION_SNAPSHOT_ARTIFACT, Path(session_path).read_text(encoding="utf-8"))
        if session is not None and hasattr(session, "dispose"):
            session.dispose()
        shutil.rmtree(root, ignore_errors=True)


def create_pi_coding_agent_harness(options: dict[str, Any] | None = None):
    options = options or {}

    async def run(input_value: PiCodingAgentInput, set_artifact: Callable[[str, Any], None] | None = None):
        artifacts: dict[str, Any] = {}

        def store(name: str, value: Any) -> None:
            artifacts[name] = value
            if set_artifact:
                set_artifact(name, value)

        result = await run_pi_coding_agent(input_value, store, options)
        result["artifacts"] = artifacts
        return result

    return {"name": options.get("name") or "pi-coding-agent", "run": run}
