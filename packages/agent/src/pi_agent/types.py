"""
Agent types — mirrors packages/agent/src/types.ts
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Literal,
    Protocol,
    Union,
)

from pydantic import BaseModel, Field, field_validator

from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
)

# ─── ThinkingLevel ────────────────────────────────────────────────────────────

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]

# ─── StreamFn ─────────────────────────────────────────────────────────────────


class StreamFn(Protocol):
    """
    Stream function used by the agent loop. ``stream_simple`` satisfies this shape.

    Contract:
    - Must not raise for request/model/runtime failures.
    - Must return an async iterator of AssistantMessageEvent (or an EventStream).
    - Failures must be encoded in the stream via protocol events and a final
      AssistantMessage with stop_reason "error" or "aborted" and error_message.
    """

    def __call__(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> (
        AsyncGenerator[AssistantMessageEvent, None]
        | Awaitable[AsyncGenerator[AssistantMessageEvent, None]]
        | Any
    ):
        ...


# ─── Execution / queue modes ──────────────────────────────────────────────────

ToolExecutionMode = Literal["sequential", "parallel"]
QueueMode = Literal["all", "one-at-a-time"]

AgentToolCall = ToolCall

# ─── AgentMessage ─────────────────────────────────────────────────────────────

# Applications and the harness extend this with custom role objects (see harness.messages).
CustomAgentMessages = Any
AgentMessage = Union[Message, CustomAgentMessages]


# ─── Hook results / contexts ──────────────────────────────────────────────────


@dataclass
class BeforeToolCallResult:
    """Result returned from ``before_tool_call``."""

    block: bool = False
    reason: str | None = None
    # Early termination only happens when every finalized tool result in the
    # batch sets terminate to True.
    terminate: bool | None = None


@dataclass
class AfterToolCallResult:
    """
    Partial override returned from ``after_tool_call``.

    Merge is field-by-field. Omitted / None fields keep the original values.
    There is no deep merge for content, details, or usage.
    """

    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = None
    usage: Usage | None = None
    terminate: bool | None = None


@dataclass
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    context: "AgentContext"


@dataclass
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    result: "AgentToolResult"
    is_error: bool
    context: "AgentContext"


@dataclass
class ShouldStopAfterTurnContext:
    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: "AgentContext"
    new_messages: list[AgentMessage]


@dataclass
class AgentLoopTurnUpdate:
    context: "AgentContext | None" = None
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None


PrepareNextTurnContext = ShouldStopAfterTurnContext

AgentEventSink = Callable[[Any], Awaitable[None] | None]


# ─── AgentLoopConfig ──────────────────────────────────────────────────────────


class AgentLoopConfig(SimpleStreamOptions):
    """
    Configuration for the agent loop — mirrors AgentLoopConfig in TypeScript.
    """

    model: Model

    # Converts AgentMessage[] to LLM-compatible Message[]
    convert_to_llm: Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]

    # Optional transform applied to context before convert_to_llm
    transform_context: Callable[[list[AgentMessage], asyncio.Event | None], Awaitable[list[AgentMessage]]] | None = None

    # Resolves API key dynamically per call
    get_api_key: Callable[[str], str | None | Awaitable[str | None]] | None = None

    # Called after turn_end; if True, emit agent_end and exit before queue polls
    should_stop_after_turn: Callable[[ShouldStopAfterTurnContext], bool | Awaitable[bool]] | None = None

    # Return replacement context/model/thinking before the next provider request
    prepare_next_turn: Callable[
        [PrepareNextTurnContext],
        AgentLoopTurnUpdate | dict[str, Any] | None | Awaitable[AgentLoopTurnUpdate | dict[str, Any] | None],
    ] | None = None

    # Returns steering messages to inject mid-run (after tools, not during)
    get_steering_messages: Callable[[], Awaitable[list[AgentMessage]] | list[AgentMessage]] | None = None

    # Returns follow-up messages after the agent would otherwise stop
    get_follow_up_messages: Callable[[], Awaitable[list[AgentMessage]] | list[AgentMessage]] | None = None

    # "sequential" or "parallel" (default). A single sequential tool forces sequential.
    tool_execution: ToolExecutionMode | None = None

    before_tool_call: Callable[
        [BeforeToolCallContext, asyncio.Event | None],
        Awaitable[BeforeToolCallResult | dict[str, Any] | None] | BeforeToolCallResult | dict[str, Any] | None,
    ] | None = None

    after_tool_call: Callable[
        [AfterToolCallContext, asyncio.Event | None],
        Awaitable[AfterToolCallResult | dict[str, Any] | None] | AfterToolCallResult | dict[str, Any] | None,
    ] | None = None

    model_config = {"arbitrary_types_allowed": True}


# ─── AgentTool ────────────────────────────────────────────────────────────────


class AgentToolResult(BaseModel):
    """Result of a tool execution."""

    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    usage: Usage | None = None
    added_tool_names: list[str] | None = None
    terminate: bool | None = None


AgentToolUpdateCallback = Callable[["AgentToolResult"], None]


class AgentTool(Tool):
    """
    An agent tool with an execute function.
    Mirrors AgentTool<TParameters> interface in TypeScript.
    """

    label: str
    prepare_arguments: Callable[[Any], Any] | None = None
    execute: Callable[
        [str, dict[str, Any], asyncio.Event | None, AgentToolUpdateCallback | None],
        Awaitable["AgentToolResult"],
    ]
    execution_mode: ToolExecutionMode | None = None

    model_config = {"arbitrary_types_allowed": True}


# ─── AgentContext ─────────────────────────────────────────────────────────────


class AgentContext(BaseModel):
    """Context for agent operations."""

    system_prompt: str = ""
    messages: list[AgentMessage] = Field(default_factory=list)
    tools: list[AgentTool] | None = None

    model_config = {"arbitrary_types_allowed": True}


# ─── AgentState ───────────────────────────────────────────────────────────────


class AgentState(BaseModel):
    """Complete agent state."""

    system_prompt: str = ""
    model: Model | None = None
    thinking_level: ThinkingLevel = "off"
    tools: list[AgentTool] = Field(default_factory=list)
    messages: list[AgentMessage] = Field(default_factory=list)
    is_streaming: bool = False
    streaming_message: AgentMessage | None = None
    pending_tool_calls: set[str] = Field(default_factory=set)
    error_message: str | None = None

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    @field_validator("tools", "messages", mode="before")
    @classmethod
    def _copy_assigned_list(cls, value: Any) -> Any:
        if value is None:
            return value
        return list(value)

    # Backward-compatible aliases used by existing Python callers
    @property
    def stream_message(self) -> AgentMessage | None:
        return self.streaming_message

    @stream_message.setter
    def stream_message(self, value: AgentMessage | None) -> None:
        self.streaming_message = value

    @property
    def error(self) -> str | None:
        return self.error_message

    @error.setter
    def error(self, value: str | None) -> None:
        self.error_message = value


# ─── AgentEvent ───────────────────────────────────────────────────────────────


class AgentEventAgentStart(BaseModel):
    type: Literal["agent_start"] = "agent_start"


class AgentEventAgentEnd(BaseModel):
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage]

    model_config = {"arbitrary_types_allowed": True}


class AgentEventTurnStart(BaseModel):
    type: Literal["turn_start"] = "turn_start"


class AgentEventTurnEnd(BaseModel):
    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage
    tool_results: list[ToolResultMessage]

    model_config = {"arbitrary_types_allowed": True}


class AgentEventMessageStart(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: AgentMessage

    model_config = {"arbitrary_types_allowed": True}


class AgentEventMessageUpdate(BaseModel):
    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent

    model_config = {"arbitrary_types_allowed": True}


class AgentEventMessageEnd(BaseModel):
    type: Literal["message_end"] = "message_end"
    message: AgentMessage

    model_config = {"arbitrary_types_allowed": True}


class AgentEventToolStart(BaseModel):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: Any


class AgentEventToolUpdate(BaseModel):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any


class AgentEventToolEnd(BaseModel):
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool


AgentEvent = Union[
    AgentEventAgentStart,
    AgentEventAgentEnd,
    AgentEventTurnStart,
    AgentEventTurnEnd,
    AgentEventMessageStart,
    AgentEventMessageUpdate,
    AgentEventMessageEnd,
    AgentEventToolStart,
    AgentEventToolUpdate,
    AgentEventToolEnd,
]
