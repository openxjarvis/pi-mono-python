"""Coding-agent harness factory — mirrors packages/coding-agent/src/server/create-harness.ts."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pi_agent.harness.agent_harness import AgentHarness, AgentHarnessOptions, HarnessTool
from pi_agent.harness.tools import (
    ExecutionToolContext,
    create_bash_tool,
    create_edit_tool,
    create_read_tool,
    create_write_tool,
)
from pi_agent.harness.types import ExecutionEnv
from pi_coding_agent.core.experimental import get_experimental_tool_sampling
from pi_coding_agent.core.system_prompt import build_system_prompt

READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION = {
    "snippet": "Read file contents",
    "guidelines": ["Use read to examine files instead of cat or sed."],
}

BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION = {
    "snippet": "Execute bash commands (ls, grep, find, etc.)",
    "guidelines": ["You can inspect PI_* environment variables for current model and session details."],
}

EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION = {
    "snippet": "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
    "guidelines": [
        "Use edit for precise changes (edits[].oldText must match exactly)",
        "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
        "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
        "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
    ],
}

WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION = {
    "snippet": "Create or overwrite files",
    "guidelines": ["Use write only for new files or complete rewrites."],
}


class CodingAgentHarnessTool:
    """Harness tool plus optional system-prompt contribution fields."""

    def __init__(
        self,
        tool: Any,
        prompt_snippet: str | None = None,
        prompt_guidelines: list[str] | None = None,
    ) -> None:
        self._tool = tool
        self.prompt_snippet = prompt_snippet
        self.prompt_guidelines = list(prompt_guidelines or [])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    @property
    def name(self) -> str:
        return self._tool.name


@dataclass
class CreateCodingAgentHarnessOptions:
    session: Any = None
    models: Any = None
    model: Any = None
    env: ExecutionEnv | None = None
    cwd: str | None = None
    bash_command_prefix: str | None = None
    session_file: str | None = None
    tools: list[Any] | None = None
    active_tool_names: list[str] | None = None
    system_prompt: str | Callable[[], str | Awaitable[str]] | None = None
    system_prompt_options: dict[str, Any] | None = None
    thinking_level: Any = None
    resources: Any = None
    stream_options: Any = None
    retry: Any = None
    compaction: Any = None
    steering_mode: Any = None
    follow_up_mode: Any = None
    tool_execution: Any = None
    drive: Any = None
    to_provider_messages: Any = None
    entry_projectors: Any = None
    context: Any = None


def _tool_name(tool: Any) -> str:
    return tool["name"] if isinstance(tool, dict) else tool.name


def _tool_snippet(tool: Any) -> str | None:
    if isinstance(tool, dict):
        return tool.get("prompt_snippet")
    return getattr(tool, "prompt_snippet", None)


def _tool_guidelines(tool: Any) -> list[str]:
    if isinstance(tool, dict):
        return list(tool.get("prompt_guidelines") or [])
    return list(getattr(tool, "prompt_guidelines", None) or [])


def build_coding_agent_harness_system_prompt(
    *,
    cwd: str,
    tools: list[Any],
    active_tool_names: list[str],
    system_prompt_options: dict[str, Any] | None = None,
) -> str:
    active = []
    for name in active_tool_names:
        tool = next((candidate for candidate in tools if _tool_name(candidate) == name), None)
        if tool is not None:
            active.append(tool)
    tool_snippets: dict[str, str] = {}
    for tool in active:
        snippet = _tool_snippet(tool)
        if snippet:
            normalized = " ".join(snippet.replace("\r", " ").replace("\n", " ").split())
            if normalized:
                tool_snippets[_tool_name(tool)] = normalized
    prompt_guidelines = [item for tool in active for item in _tool_guidelines(tool)]
    return build_system_prompt(
        cwd,
        selected_tools=[_tool_name(tool) for tool in active],
        tool_snippets=tool_snippets,
        prompt_guidelines=prompt_guidelines,
        **(system_prompt_options or {}),
    )


def _attach_prompt(tool: Any, prompt: dict[str, Any]) -> Any:
    object.__setattr__(tool, "prompt_snippet", prompt["snippet"])
    object.__setattr__(tool, "prompt_guidelines", list(prompt["guidelines"]))
    sampling = get_experimental_tool_sampling()
    if sampling is not None and hasattr(tool, "constrained_sampling"):
        tool.constrained_sampling = sampling
    return tool


def create_coding_agent_harness_tool(tool: HarnessTool, context: ExecutionToolContext, prompt: dict[str, Any]) -> Any:
    original = tool.execute

    async def execute(tool_call_id, params, abort=None, on_update=None):
        return await original(tool_call_id, params, abort, on_update, context)

    wrapped = tool.model_copy() if hasattr(tool, "model_copy") else tool
    wrapped.execute = execute
    return _attach_prompt(wrapped, prompt)


def _as_options(options: CreateCodingAgentHarnessOptions | dict[str, Any]) -> CreateCodingAgentHarnessOptions:
    if isinstance(options, CreateCodingAgentHarnessOptions):
        return options
    return CreateCodingAgentHarnessOptions(**options)


async def _ensure_harness_dependencies(options: CreateCodingAgentHarnessOptions) -> CreateCodingAgentHarnessOptions:
    cwd = options.cwd or (getattr(options.env, "cwd", None) if options.env is not None else None) or os.getcwd()
    if options.env is None:
        from pi_agent.harness.env.python import create_python_execution_env

        options.env = create_python_execution_env(cwd)
    if options.session is None:
        from pi_agent.harness.session.memory import InMemorySessionStorage
        from pi_agent.harness.session.session import Session
        from pi_ai.utils.uuid import uuidv7

        storage = InMemorySessionStorage({"id": uuidv7()})
        options.session = Session(storage)
    if options.models is None or options.model is None:
        from pi_ai import create_models

        models = options.models or create_models()
        options.models = models
        if options.model is None:
            all_models = models.get_models()
            options.model = all_models[0] if all_models else None
    return options


async def create_coding_agent_harness(options: CreateCodingAgentHarnessOptions | dict[str, Any]) -> dict[str, Any]:
    options = await _ensure_harness_dependencies(_as_options(options))
    harness: AgentHarness | None = None

    def get_harness() -> AgentHarness:
        if harness is None:
            raise RuntimeError("Coding-agent Harness callback ran before Harness initialization")
        return harness

    tools = options.tools
    if tools is None:
        metadata = await options.session.get_metadata()
        tool_context = ExecutionToolContext(env=options.env)

        async def prepare(execution: dict[str, Any], _context: ExecutionToolContext, _abort=None) -> None:
            current = get_harness()
            model = await current.get_model()
            thinking_level = await current.get_thinking_level()
            execution["env"]["PI_SESSION_ID"] = metadata["id"] if isinstance(metadata, dict) else metadata.id
            execution["env"]["PI_SESSION_FILE"] = options.session_file or ""
            execution["env"]["PI_PROVIDER"] = model.provider
            execution["env"]["PI_MODEL"] = model.id
            execution["env"]["PI_REASONING_LEVEL"] = thinking_level

        tools = [
            create_coding_agent_harness_tool(
                create_read_tool(),
                tool_context,
                READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION,
            ),
            create_coding_agent_harness_tool(
                create_bash_tool(
                    {
                        "command_prefix": options.bash_command_prefix,
                        "prepare": prepare,
                    }
                ),
                tool_context,
                BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION,
            ),
            create_coding_agent_harness_tool(
                create_edit_tool(),
                tool_context,
                EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION,
            ),
            create_coding_agent_harness_tool(
                create_write_tool(),
                tool_context,
                WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION,
            ),
        ]

    active_tool_names = list(options.active_tool_names or [_tool_name(tool) for tool in tools])
    system_prompt = options.system_prompt
    if system_prompt is None:

        async def default_system_prompt() -> str:
            current = get_harness()
            current_tools = await current.get_tools()
            current_active = await current.get_active_tools()
            return build_coding_agent_harness_system_prompt(
                cwd=options.env.cwd,
                tools=current_tools,
                active_tool_names=current_active,
                system_prompt_options=options.system_prompt_options,
            )

        system_prompt = default_system_prompt

    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=options.session,
            models=options.models,
            model=options.model,
            thinking_level=options.thinking_level,
            active_tool_names=active_tool_names,
            tools=tools,
            system_prompt=system_prompt,
            resources=options.resources,
            stream_options=options.stream_options,
            retry=options.retry,
            compaction=options.compaction,
            steering_mode=options.steering_mode,
            follow_up_mode=options.follow_up_mode,
            tool_execution=options.tool_execution,
            drive=options.drive,
            to_provider_messages=options.to_provider_messages,
            entry_projectors=options.entry_projectors,
            context=options.context,
        )
    )
    harness = created["harness"]
    harness.env = options.env
    if not hasattr(harness, "agent") or getattr(harness, "agent", None) is None:
        harness.agent = getattr(harness, "_agent", harness)
    return harness
