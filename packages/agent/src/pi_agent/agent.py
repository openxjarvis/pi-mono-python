"""
Agent class — mirrors packages/agent/src/agent.ts

Stateful wrapper around the agent loop.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Callable

from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    Message,
    Model,
    ModelCost,
    TextContent,
    ThinkingBudgets,
    Transport,
    Usage,
    UserMessage,
)

from .agent_loop import run_agent_loop, run_agent_loop_continue
from .stream_fn import get_default_stream_fn
from .types import (
    AgentContext,
    AgentEvent,
    AgentEventAgentEnd,
    AgentEventMessageEnd,
    AgentEventMessageStart,
    AgentEventTurnEnd,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    PrepareNextTurnContext,
    QueueMode,
    ShouldStopAfterTurnContext,
    StreamFn,
    ThinkingLevel,
    ToolExecutionMode,
)

EMPTY_USAGE = Usage()

DEFAULT_MODEL = Model(
    id="unknown",
    name="unknown",
    api="unknown",
    provider="unknown",
    base_url="",
    reasoning=False,
    input=[],
    cost=ModelCost(),
    context_window=0,
    max_tokens=0,
)


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Default converter: keep only LLM-compatible messages."""
    return [
        m
        for m in messages
        if getattr(m, "role", None) in ("user", "assistant", "toolResult")
    ]


class AgentOptions:
    """Options for constructing an Agent. Mirrors AgentOptions interface."""

    def __init__(
        self,
        initial_state: dict[str, Any] | None = None,
        convert_to_llm: Callable[[list[AgentMessage]], list[Message] | Any] | None = None,
        transform_context: Callable | None = None,
        stream_fn: StreamFn | None = None,
        get_api_key: Callable | None = None,
        on_payload: Callable | None = None,
        on_response: Callable | None = None,
        before_tool_call: Callable | None = None,
        after_tool_call: Callable | None = None,
        should_stop_after_turn: Callable | None = None,
        prepare_next_turn: Callable | None = None,
        prepare_next_turn_with_context: Callable | None = None,
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
        session_id: str | None = None,
        thinking_budgets: ThinkingBudgets | None = None,
        transport: Transport = "auto",
        max_retry_delay_ms: int | None = None,
        tool_execution: ToolExecutionMode = "parallel",
    ):
        self.initial_state = initial_state
        self.convert_to_llm = convert_to_llm
        self.transform_context = transform_context
        self.stream_fn = stream_fn
        self.get_api_key = get_api_key
        self.on_payload = on_payload
        self.on_response = on_response
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.should_stop_after_turn = should_stop_after_turn
        self.prepare_next_turn = prepare_next_turn
        self.prepare_next_turn_with_context = prepare_next_turn_with_context
        self.steering_mode = steering_mode
        self.follow_up_mode = follow_up_mode
        self.session_id = session_id
        self.thinking_budgets = thinking_budgets
        self.transport = transport
        self.max_retry_delay_ms = max_retry_delay_ms
        self.tool_execution = tool_execution

    @classmethod
    def from_dict(cls, opts_dict: dict[str, Any]) -> "AgentOptions":
        """Construct AgentOptions from a dict (mirrors TS object literal usage)."""
        return cls(**opts_dict)


class PendingMessageQueue:
    """Steering / follow-up queue with all vs one-at-a-time drain modes."""

    def __init__(self, mode: QueueMode) -> None:
        self.mode = mode
        self._messages: list[AgentMessage] = []

    def enqueue(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return bool(self._messages)

    def peek(self) -> list[AgentMessage]:
        """Return queued messages without draining them."""
        return list(self._messages)

    def drain(self) -> list[AgentMessage]:
        if self.mode == "all":
            drained = list(self._messages)
            self._messages = []
            return drained
        if not self._messages:
            return []
        first = self._messages[0]
        self._messages = self._messages[1:]
        return [first]

    def clear(self) -> None:
        self._messages = []

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)


class _ActiveRun:
    def __init__(self) -> None:
        self.future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.cancel_event = asyncio.Event()


class Agent:
    """
    Stateful agent wrapper around the agent loop.
    Mirrors the Agent class in TypeScript.
    """

    def __init__(self, opts: AgentOptions | dict[str, Any] | None = None) -> None:
        if isinstance(opts, dict):
            opts = AgentOptions.from_dict(opts)
        opts = opts or AgentOptions()

        initial = dict(opts.initial_state or {})
        # Map older field names if callers still pass them
        if "stream_message" in initial and "streaming_message" not in initial:
            initial["streaming_message"] = initial.pop("stream_message")
        if "error" in initial and "error_message" not in initial:
            initial["error_message"] = initial.pop("error")

        self._state = AgentState(
            system_prompt=initial.get("system_prompt", ""),
            model=initial.get("model") or DEFAULT_MODEL,
            thinking_level=initial.get("thinking_level", "off"),
            tools=list(initial.get("tools") or []),
            messages=list(initial.get("messages") or []),
            is_streaming=False,
            streaming_message=None,
            pending_tool_calls=set(),
            error_message=None,
        )

        self.convert_to_llm = opts.convert_to_llm or _default_convert_to_llm
        self.transform_context = opts.transform_context
        self.stream_function: StreamFn = opts.stream_fn or get_default_stream_fn()
        self.get_api_key = opts.get_api_key
        self.on_payload = opts.on_payload
        self.on_response = opts.on_response
        self.before_tool_call = opts.before_tool_call
        self.after_tool_call = opts.after_tool_call
        self.should_stop_after_turn = opts.should_stop_after_turn
        self.prepare_next_turn = opts.prepare_next_turn
        self.prepare_next_turn_with_context = opts.prepare_next_turn_with_context
        self._steering_queue = PendingMessageQueue(opts.steering_mode)
        self._follow_up_queue = PendingMessageQueue(opts.follow_up_mode)
        self.session_id = opts.session_id
        self.thinking_budgets = opts.thinking_budgets
        self.transport: Transport = opts.transport
        self.max_retry_delay_ms = opts.max_retry_delay_ms
        self.tool_execution: ToolExecutionMode = opts.tool_execution

        self._listeners: set[Callable] = set()
        self._active_run: _ActiveRun | None = None

        # TS parity: expose `agent.continue()` despite Python keyword constraints.
        setattr(self, "continue", self.continue_from_context)

    # Backward-compatible aliases
    @property
    def stream_fn(self) -> StreamFn:
        return self.stream_function

    @stream_fn.setter
    def stream_fn(self, value: StreamFn) -> None:
        self.stream_function = value

    @property
    def _convert_to_llm(self):
        return self.convert_to_llm

    @_convert_to_llm.setter
    def _convert_to_llm(self, value) -> None:
        self.convert_to_llm = value

    @property
    def _transform_context(self):
        return self.transform_context

    @_transform_context.setter
    def _transform_context(self, value) -> None:
        self.transform_context = value

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering_queue.mode = mode

    def set_steering_mode(self, mode: str) -> None:
        self.steering_mode = mode  # type: ignore[assignment]

    def get_steering_mode(self) -> str:
        return self.steering_mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_queue.mode = mode

    def set_follow_up_mode(self, mode: str) -> None:
        self.follow_up_mode = mode  # type: ignore[assignment]

    def get_follow_up_mode(self) -> str:
        return self.follow_up_mode

    @property
    def signal(self) -> asyncio.Event | None:
        """Active cancel event for the current run, if any."""
        return self._active_run.cancel_event if self._active_run else None

    def set_transport(self, value: Transport) -> None:
        self.transport = value

    # ── Subscriptions ─────────────────────────────────────────────────────────

    def subscribe(self, fn: Callable) -> Callable[[], None]:
        """
        Subscribe to agent lifecycle events.

        Listeners may be sync or async and may accept ``(event)`` or
        ``(event, cancel_event)``. Listener awaitables are included in run
        settlement; ``agent_end`` is the last event, but the agent is idle
        only after those listeners finish.
        """
        self._listeners.add(fn)
        return lambda: self._listeners.discard(fn)

    # ── State mutators (kept for existing Python callers) ─────────────────────

    def set_system_prompt(self, v: str) -> None:
        self._state.system_prompt = v

    def set_model(self, m: Model) -> None:
        self._state.model = m

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        self._state.thinking_level = level

    def set_tools(self, tools: list[AgentTool]) -> None:
        self._state.tools = list(tools)

    def replace_messages(self, messages: list[AgentMessage]) -> None:
        self._state.messages = list(messages)

    def append_message(self, message: AgentMessage) -> None:
        self._state.messages = [*self._state.messages, message]

    def clear_messages(self) -> None:
        self._state.messages = []

    def steer(self, message: AgentMessage) -> None:
        """Queue a message to inject after the current assistant turn finishes."""
        self._steering_queue.enqueue(message)

    def follow_up(self, message: AgentMessage) -> None:
        """Queue a message to run only after the agent would otherwise stop."""
        self._follow_up_queue.enqueue(message)

    def peek_steering_messages(self) -> list[AgentMessage]:
        return self._steering_queue.peek()

    def peek_follow_up_messages(self) -> list[AgentMessage]:
        return self._follow_up_queue.peek()

    def clear_steering_queue(self) -> None:
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    def abort(self) -> None:
        """Abort the current run, if one is active."""
        if self._active_run:
            self._active_run.cancel_event.set()

    async def wait_for_idle(self) -> None:
        """Resolve when the current run and all awaited listeners have finished."""
        if self._active_run is not None:
            await self._active_run.future

    def reset(self) -> None:
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before resetting.")
        self._state.messages = []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._state.error_message = None
        self.clear_all_queues()

    # ── Prompt / Continue ─────────────────────────────────────────────────────

    async def prompt(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        """Send a prompt to the agent."""
        if self._active_run is not None:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or follow_up() to queue messages, or wait for completion."
            )
        messages = self._normalize_prompt_input(input, images)
        await self._run_prompt_messages(messages)

    async def continue_from_context(self) -> None:
        """Continue from current context (for retries)."""
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")

        messages = self._state.messages
        if not messages:
            raise RuntimeError("No messages to continue from")

        last = messages[-1]
        if getattr(last, "role", None) == "assistant":
            queued_steering = self._steering_queue.drain()
            if queued_steering:
                await self._run_prompt_messages(queued_steering, skip_initial_steering_poll=True)
                return

            queued_follow = self._follow_up_queue.drain()
            if queued_follow:
                await self._run_prompt_messages(queued_follow)
                return

            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_continuation()

    def _normalize_prompt_input(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None,
    ) -> list[AgentMessage]:
        if isinstance(input, list):
            return input
        if not isinstance(input, str):
            return [input]
        content: list[TextContent | ImageContent] = [TextContent(type="text", text=input)]
        if images:
            content.extend(images)
        return [
            UserMessage(
                role="user",
                content=content,
                timestamp=int(time.time() * 1000),
            )
        ]

    async def _run_prompt_messages(
        self,
        messages: list[AgentMessage],
        skip_initial_steering_poll: bool = False,
    ) -> None:
        async def _executor(cancel_event: asyncio.Event) -> None:
            await run_agent_loop(
                messages,
                self._create_context_snapshot(),
                self._create_loop_config(skip_initial_steering_poll=skip_initial_steering_poll),
                self._process_events,
                cancel_event,
                self.stream_function,
            )

        await self._run_with_lifecycle(_executor)

    async def _run_continuation(self) -> None:
        async def _executor(cancel_event: asyncio.Event) -> None:
            await run_agent_loop_continue(
                self._create_context_snapshot(),
                self._create_loop_config(),
                self._process_events,
                cancel_event,
                self.stream_function,
            )

        await self._run_with_lifecycle(_executor)

    def _create_context_snapshot(self) -> AgentContext:
        return AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

    def _create_loop_config(self, skip_initial_steering_poll: bool = False) -> AgentLoopConfig:
        skip = skip_initial_steering_poll

        async def get_steering() -> list[AgentMessage]:
            nonlocal skip
            if skip:
                skip = False
                return []
            return self._steering_queue.drain()

        async def get_follow_up() -> list[AgentMessage]:
            return self._follow_up_queue.drain()

        should_stop = self.should_stop_after_turn

        async def should_stop_after_turn(context: ShouldStopAfterTurnContext) -> bool:
            if not should_stop:
                return False
            return bool(await _maybe_await(should_stop(context, self.signal)))

        prepare = self.prepare_next_turn
        prepare_with = self.prepare_next_turn_with_context

        async def prepare_next_turn(context: PrepareNextTurnContext):
            if prepare_with:
                return await _maybe_await(prepare_with(context, self.signal))
            if prepare:
                return await _maybe_await(prepare(self.signal))
            return None

        return AgentLoopConfig(
            model=self._state.model,
            reasoning=None if self._state.thinking_level == "off" else self._state.thinking_level,
            session_id=self.session_id,
            on_payload=self.on_payload,
            on_response=self.on_response,
            transport=self.transport,
            thinking_budgets=self.thinking_budgets,
            max_retry_delay_ms=self.max_retry_delay_ms,
            tool_execution=self.tool_execution,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            should_stop_after_turn=should_stop_after_turn if should_stop else None,
            prepare_next_turn=prepare_next_turn if (prepare_with or prepare) else None,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=get_steering,
            get_follow_up_messages=get_follow_up,
        )

    async def _run_with_lifecycle(self, executor: Callable[[asyncio.Event], Any]) -> None:
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing.")

        run = _ActiveRun()
        self._active_run = run
        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        try:
            await executor(run.cancel_event)
        except Exception as error:
            await self._handle_run_failure(error, run.cancel_event.is_set())
        finally:
            self._finish_run()

    async def _handle_run_failure(self, error: Any, aborted: bool) -> None:
        failure_message = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="")],
            api=self._state.model.api,
            provider=self._state.model.provider,
            model=self._state.model.id,
            usage=EMPTY_USAGE,
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
            timestamp=int(time.time() * 1000),
        )
        await self._process_events(AgentEventMessageStart(message=failure_message))
        await self._process_events(AgentEventMessageEnd(message=failure_message))
        await self._process_events(AgentEventTurnEnd(message=failure_message, tool_results=[]))
        await self._process_events(AgentEventAgentEnd(messages=[failure_message]))

    def _finish_run(self) -> None:
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        run = self._active_run
        self._active_run = None
        if run is not None and not run.future.done():
            run.future.set_result(None)

    async def _process_events(self, event: AgentEvent) -> None:
        """Reduce internal state for a loop event, then await listeners."""
        if event.type == "message_start":
            self._state.streaming_message = event.message
        elif event.type == "message_update":
            self._state.streaming_message = event.message
        elif event.type == "message_end":
            self._state.streaming_message = None
            self._state.messages = [*self._state.messages, event.message]
        elif event.type == "tool_execution_start":
            pending = set(self._state.pending_tool_calls)
            pending.add(event.tool_call_id)
            self._state.pending_tool_calls = pending
        elif event.type == "tool_execution_end":
            pending = set(self._state.pending_tool_calls)
            pending.discard(event.tool_call_id)
            self._state.pending_tool_calls = pending
        elif event.type == "turn_end":
            err = getattr(event.message, "error_message", None)
            if getattr(event.message, "role", None) == "assistant" and err:
                self._state.error_message = err
        elif event.type == "agent_end":
            self._state.streaming_message = None

        if self._active_run is None:
            raise RuntimeError("Agent listener invoked outside active run")
        cancel_event = self._active_run.cancel_event
        for listener in list(self._listeners):
            await _invoke_listener(listener, event, cancel_event)


# ─── helpers used by Agent ────────────────────────────────────────────────────


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _callable_positional_count(fn: Callable) -> int:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 1
    count = 0
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            return 2
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            count += 1
    return count


async def _invoke_listener(listener: Callable, event: AgentEvent, cancel_event: asyncio.Event) -> None:
    if _callable_positional_count(listener) >= 2:
        result = listener(event, cancel_event)
    else:
        result = listener(event)
    await _maybe_await(result)
