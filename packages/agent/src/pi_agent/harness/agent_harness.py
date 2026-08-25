"""AgentHarness v4 scaffold — mirrors harness/agent-harness.ts."""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Literal

from pi_ai.types import ImageContent, Message, Model, SimpleStreamOptions, Usage
from pi_ai.models_runtime import Models
from pi_ai.utils.retry import RetryPolicy

from pi_agent.harness.compaction.compaction import CompactionSettings
from pi_agent.harness.result import Result, tagged_error
from pi_agent.harness.session.session import Session
from pi_agent.harness.types import AgentHarnessResources, PromptTemplate, Skill
from pi_agent.types import AgentMessage, AgentTool, QueueMode, ThinkingLevel

LaneBusy = tagged_error("LaneBusy")
MissingIdentities = tagged_error("MissingIdentities")
NoActiveRun = tagged_error("NoActiveRun")
NoActiveOperation = tagged_error("NoActiveOperation")
NothingToResume = tagged_error("NothingToResume")
InvalidMessage = tagged_error("InvalidMessage")
UnknownSkill = tagged_error("UnknownSkill")
UnknownTemplate = tagged_error("UnknownTemplate")
UnknownTarget = tagged_error("UnknownTarget")
UnknownQueueItem = tagged_error("UnknownQueueItem")
LaneExists = tagged_error("LaneExists")
InvalidLane = tagged_error("InvalidLane")
NothingToCompact = tagged_error("NothingToCompact")
Closed = tagged_error("Closed")


class HarnessFault(Exception):
    def __init__(self, message: str, cause: object) -> None:
        super().__init__(message)
        self.name = "HarnessFault"
        self.cause = cause


class HarnessClosed(Exception):
    def __init__(self) -> None:
        super().__init__("AgentHarness was closed while the operation was active")
        self.name = "HarnessClosed"


class HarnessNotImplemented(Exception):
    def __init__(self, operation: str) -> None:
        super().__init__(f"AgentHarness.{operation} is not implemented yet")
        self.name = "HarnessNotImplemented"
        self.operation = operation


class HarnessTool(AgentTool):
    replay: Literal["never", "safe"] | None = None


Resources = AgentHarnessResources
StreamOptions = SimpleStreamOptions
StreamOptionsPatch = dict[str, Any]
EntryProjector = Callable[[Any], list[AgentMessage] | Awaitable[list[AgentMessage]]]

HookName = Literal[
    "before_run",
    "before_resume",
    "before_run_end",
    "transform_context",
    "before_request",
    "before_payload",
    "after_response",
    "before_tool",
    "after_tool",
    "before_compaction",
    "before_navigation",
]


class UnavailableRegistry:
    def __init__(self, operation: str, is_closed: Callable[[], bool]) -> None:
        self._operation = operation
        self._is_closed = is_closed

    def on(self, _name: str, _handler: Callable, _options: dict[str, Any] | None = None) -> Callable[[], None]:
        if self._is_closed():
            raise HarnessClosed()
        raise HarnessNotImplemented(self._operation)


class AgentHarnessOptions:
    def __init__(
        self,
        session: Session,
        models: Models,
        model: Model,
        thinking_level: ThinkingLevel | None = None,
        active_tool_names: list[str] | None = None,
        tools: list[HarnessTool] | None = None,
        tool_context: object | None = None,
        system_prompt: str | Callable[[], str | Awaitable[str]] | None = None,
        resources: Resources | None = None,
        stream_options: StreamOptions | None = None,
        retry: RetryPolicy | None = None,
        compaction: CompactionSettings | None = None,
        steering_mode: QueueMode | None = None,
        follow_up_mode: QueueMode | None = None,
        tool_execution: Literal["sequential", "parallel"] | None = None,
        drive: Literal["automatic", "manual"] | None = None,
        to_provider_messages: Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]] | None = None,
        entry_projectors: dict[str, EntryProjector] | None = None,
        context: Any = None,
        **_extra: Any,
    ) -> None:
        self.session = session
        self.models = models
        self.model = model
        self.thinking_level = thinking_level
        self.active_tool_names = active_tool_names
        self.tools = tools
        self.tool_context = tool_context
        self.system_prompt = system_prompt
        self.resources = resources
        self.stream_options = stream_options
        self.retry = retry
        self.compaction = compaction
        self.steering_mode = steering_mode
        self.follow_up_mode = follow_up_mode
        self.tool_execution = tool_execution
        self.drive = drive
        self.to_provider_messages = to_provider_messages
        self.entry_projectors = entry_projectors
        self.context = context


class AgentHarness:
    name = "main"

    def __init__(self, options: AgentHarnessOptions | dict[str, Any]) -> None:
        if isinstance(options, dict):
            options = AgentHarnessOptions(**options)
        self._durable_session = options.session
        self.session = options.session
        self.hooks = UnavailableRegistry("hooks.on", lambda: self._closed)
        self.events = UnavailableRegistry("events.on", lambda: self._closed)
        self._model = options.model
        self._thinking_level: ThinkingLevel = options.thinking_level or "off"
        self._active_tool_names = list(
            options.active_tool_names
            if options.active_tool_names is not None
            else ([tool.name for tool in options.tools] if options.tools else [])
        )
        self._tools = list(options.tools or [])
        self._resources: Resources = {
            "skills": list(options.resources["skills"]) if options.resources and options.resources.get("skills") else None,
            "prompt_templates": (
                list(options.resources["prompt_templates"])
                if options.resources and options.resources.get("prompt_templates")
                else None
            ),
        }
        self._stream_options = dict(options.stream_options or {}) if not isinstance(options.stream_options, SimpleStreamOptions) else options.stream_options.model_dump()
        if options.stream_options is None:
            self._stream_options = {}
        elif isinstance(options.stream_options, SimpleStreamOptions):
            self._stream_options = options.stream_options.model_copy()
        else:
            self._stream_options = {**options.stream_options}
        self._retry_policy = options.retry or RetryPolicy(enabled=False, max_retries=0, base_delay_ms=1000)
        self._compaction_settings = options.compaction or {
            "enabled": True,
            "reserve_tokens": 16384,
            "keep_recent_tokens": 20000,
        }
        self._steering_mode: QueueMode = options.steering_mode or "one-at-a-time"
        self._follow_up_mode: QueueMode = options.follow_up_mode or "one-at-a-time"
        self._closed = False

    @staticmethod
    async def create(options: AgentHarnessOptions | dict[str, Any]) -> dict[str, Any]:
        if isinstance(options, dict):
            options = AgentHarnessOptions(**options)
        records = await options.session.find_records({"limit": 1})
        if records:
            raise HarnessNotImplemented("create.restore")
        return {"harness": AgentHarness(options), "suspended": []}

    def _unavailable(self, operation: str) -> Any:
        if self._closed:
            raise HarnessClosed()
        raise HarnessNotImplemented(operation)

    async def get_leaf_id(self) -> str | None:
        return await self._durable_session.get_leaf_id()

    async def prompt(self, *_args: Any, **_kwargs: Any) -> Result:
        return self._unavailable("prompt")

    async def skill(self, _name: str, _additional_instructions: str | None = None) -> Result:
        return self._unavailable("skill")

    async def prompt_from_template(self, _name: str, _args: list[str] | None = None) -> Result:
        return self._unavailable("promptFromTemplate")

    async def compact(self, _options: dict[str, Any] | None = None) -> Result:
        return self._unavailable("compact")

    async def navigate_tree(self, _target_id: str | None, _options: dict[str, Any] | None = None) -> Result:
        return self._unavailable("navigateTree")

    async def resume(self) -> Result:
        return self._unavailable("resume")

    async def abort(self) -> Result:
        return self._unavailable("abort")

    async def steer(self, *_args: Any, **_kwargs: Any) -> Result:
        return self._unavailable("steer")

    async def follow_up(self, *_args: Any, **_kwargs: Any) -> Result:
        return self._unavailable("followUp")

    async def next_run(self, *_args: Any, **_kwargs: Any) -> Result:
        return self._unavailable("nextRun")

    async def cancel_queued(self, _entry_id: str) -> Result:
        return self._unavailable("cancelQueued")

    async def record_usage(self, _usage: Usage, _options: dict[str, Any] | None = None) -> Result:
        return self._unavailable("recordUsage")

    async def wait_for_idle(self) -> None:
        return self._unavailable("waitForIdle")

    async def run_when_idle(self, _callback: Callable[[], Any]) -> None:
        return self._unavailable("runWhenIdle")

    async def peek_action(self) -> Any:
        return self._unavailable("peekAction")

    async def execute_action(self) -> Any:
        return self._unavailable("executeAction")

    async def run_to_completion(self) -> None:
        return self._unavailable("runToCompletion")

    async def get_model(self) -> Model:
        return self._model

    async def set_model(self, model: Model) -> None:
        self._model = model

    async def get_thinking_level(self) -> ThinkingLevel:
        return self._thinking_level

    async def set_thinking_level(self, level: ThinkingLevel) -> None:
        self._thinking_level = level

    async def get_active_tools(self) -> list[str]:
        return list(self._active_tool_names)

    async def set_active_tools(self, names: list[str]) -> None:
        self._active_tool_names = list(names)

    async def watch(self) -> Any:
        return self._unavailable("watch")

    async def lane(self, _name: str) -> Any:
        return self._unavailable("lane")

    async def create_lane(self, _name: str, _at: str | None) -> Result:
        return self._unavailable("createLane")

    async def lanes(self) -> list[Any]:
        return self._unavailable("lanes")

    async def get_tools(self) -> list[HarnessTool]:
        return list(self._tools)

    async def set_tools(self, tools: list[HarnessTool], active_names: list[str] | None = None) -> None:
        self._tools = list(tools)
        self._active_tool_names = list(active_names if active_names is not None else [tool.name for tool in tools])

    async def get_resources(self) -> Resources:
        return {
            "skills": list(self._resources["skills"]) if self._resources.get("skills") else None,
            "prompt_templates": list(self._resources["prompt_templates"]) if self._resources.get("prompt_templates") else None,
        }

    async def set_resources(self, resources: Resources) -> None:
        self._resources = {
            "skills": list(resources["skills"]) if resources.get("skills") else None,
            "prompt_templates": list(resources["prompt_templates"]) if resources.get("prompt_templates") else None,
        }

    async def get_stream_options(self) -> Any:
        if isinstance(self._stream_options, SimpleStreamOptions):
            return self._stream_options.model_copy()
        return {**self._stream_options}

    async def set_stream_options(self, options: Any) -> None:
        if isinstance(options, SimpleStreamOptions):
            self._stream_options = options.model_copy()
        else:
            self._stream_options = {**options}

    async def get_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            enabled=self._retry_policy.enabled,
            max_retries=self._retry_policy.max_retries,
            base_delay_ms=self._retry_policy.base_delay_ms,
        )

    async def set_retry_policy(self, policy: RetryPolicy | dict[str, Any]) -> None:
        if isinstance(policy, dict):
            policy = RetryPolicy(
                enabled=policy.get("enabled", False),
                max_retries=policy.get("max_retries", policy.get("maxRetries", 0)),
                base_delay_ms=policy.get("base_delay_ms", policy.get("baseDelayMs", 1000)),
            )
        self._retry_policy = RetryPolicy(
            enabled=policy.enabled,
            max_retries=policy.max_retries,
            base_delay_ms=policy.base_delay_ms,
        )

    async def get_compaction_settings(self) -> CompactionSettings:
        return {**self._compaction_settings}

    async def set_compaction_settings(self, settings: CompactionSettings) -> None:
        self._compaction_settings = {**settings}

    async def get_steering_mode(self) -> QueueMode:
        return self._steering_mode

    async def set_steering_mode(self, mode: QueueMode) -> None:
        self._steering_mode = mode

    async def get_follow_up_mode(self) -> QueueMode:
        return self._follow_up_mode

    async def set_follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_mode = mode

    async def watch_session(self) -> Any:
        return self._unavailable("watchSession")

    async def close(self) -> None:
        self._closed = True
