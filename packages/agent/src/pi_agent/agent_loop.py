"""
Agent loop — mirrors packages/agent/src/agent-loop.ts

Core loop logic: agent_loop(), agent_loop_continue(), run_agent_loop().
Transforms to LLM Message[] only at the provider-call boundary.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pi_ai.types import (
    AssistantMessage,
    Context,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from pi_ai.utils.event_stream import EventStream
from pi_ai.utils.validation import validate_tool_arguments

from .stream_fn import get_default_stream_fn
from .types import (
    AfterToolCallContext,
    AgentContext,
    AgentEvent,
    AgentEventAgentEnd,
    AgentEventAgentStart,
    AgentEventMessageEnd,
    AgentEventMessageStart,
    AgentEventMessageUpdate,
    AgentEventToolEnd,
    AgentEventToolStart,
    AgentEventToolUpdate,
    AgentEventTurnEnd,
    AgentEventTurnStart,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    BeforeToolCallContext,
    PrepareNextTurnContext,
    ShouldStopAfterTurnContext,
    StreamFn,
)

AgentEventSink = Callable[[AgentEvent], Awaitable[None] | None]


def _create_agent_stream() -> EventStream[AgentEvent, list[AgentMessage]]:
    return EventStream(
        is_done=lambda e: e.type == "agent_end",
        get_result=lambda e: e.messages if e.type == "agent_end" else [],
    )


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """
    Start an agent loop with new prompt messages.
    Mirrors agentLoop() in TypeScript.
    """
    ev_stream = _create_agent_stream()

    async def _emit(event: AgentEvent) -> None:
        ev_stream.push(event)

    async def _run() -> None:
        try:
            messages = await run_agent_loop(
                prompts, context, config, _emit, cancel_event, stream_fn
            )
            if not ev_stream._result_event.is_set():
                ev_stream.end(messages)
        except Exception as e:
            if not ev_stream._result_event.is_set():
                ev_stream.fail(e)

    asyncio.ensure_future(_run())
    return ev_stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """
    Continue from the current context without adding a new message.
    Mirrors agentLoopContinue() in TypeScript.

    The last message must convert to a user or toolResult message via
    convert_to_llm. If it doesn't, the LLM provider will reject the request.
    """
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    last = context.messages[-1]
    if getattr(last, "role", None) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    ev_stream = _create_agent_stream()

    async def _emit(event: AgentEvent) -> None:
        ev_stream.push(event)

    async def _run() -> None:
        try:
            messages = await run_agent_loop_continue(
                context, config, _emit, cancel_event, stream_fn
            )
            if not ev_stream._result_event.is_set():
                ev_stream.end(messages)
        except Exception as e:
            if not ev_stream._result_event.is_set():
                ev_stream.fail(e)

    asyncio.ensure_future(_run())
    return ev_stream


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancel_event: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """Low-level prompt run. Mirrors runAgentLoop()."""
    new_messages: list[AgentMessage] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages) + list(prompts),
        tools=context.tools,
    )

    await _emit(emit, AgentEventAgentStart())
    await _emit(emit, AgentEventTurnStart())
    for prompt in prompts:
        await _emit(emit, AgentEventMessageStart(message=prompt))
        await _emit(emit, AgentEventMessageEnd(message=prompt))

    await _run_loop(
        current_context,
        new_messages,
        config,
        cancel_event,
        emit,
        stream_fn or get_default_stream_fn(),
    )
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancel_event: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """Low-level continuation. Mirrors runAgentLoopContinue()."""
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    last = context.messages[-1]
    if getattr(last, "role", None) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
    )

    await _emit(emit, AgentEventAgentStart())
    await _emit(emit, AgentEventTurnStart())

    await _run_loop(
        current_context,
        new_messages,
        config,
        cancel_event,
        emit,
        stream_fn or get_default_stream_fn(),
    )
    return new_messages


async def _run_loop(
    initial_context: AgentContext,
    new_messages: list[AgentMessage],
    initial_config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    emit: AgentEventSink,
    stream_function: StreamFn,
) -> None:
    """Main loop logic shared by prompt and continue runs."""
    current_context = initial_context
    config = initial_config
    first_turn = True
    pending_messages: list[AgentMessage] = []
    if config.get_steering_messages:
        pending_messages = list(await _maybe_await(config.get_steering_messages()) or [])

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:
            if not first_turn:
                await _emit(emit, AgentEventTurnStart())
            else:
                first_turn = False

            if pending_messages:
                for message in pending_messages:
                    await _emit(emit, AgentEventMessageStart(message=message))
                    await _emit(emit, AgentEventMessageEnd(message=message))
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            message = await _stream_assistant_response(
                current_context, config, cancel_event, emit, stream_function
            )
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                await _emit(emit, AgentEventTurnEnd(message=message, tool_results=[]))
                await _emit(emit, AgentEventAgentEnd(messages=new_messages))
                return

            tool_calls = [c for c in message.content if isinstance(c, ToolCall)]
            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                # A "length" stop means the output was cut off by the token
                # limit, so every tool call may carry truncated arguments.
                if message.stop_reason == "length":
                    executed = await _fail_tool_calls_from_truncated_message(tool_calls, emit)
                else:
                    executed = await _execute_tool_calls(
                        current_context, message, config, cancel_event, emit
                    )
                tool_results.extend(executed.messages)
                has_more_tool_calls = not executed.terminate

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await _emit(emit, AgentEventTurnEnd(message=message, tool_results=tool_results))

            next_turn_context = PrepareNextTurnContext(
                message=message,
                tool_results=tool_results,
                context=current_context,
                new_messages=new_messages,
            )
            if config.prepare_next_turn:
                snapshot = await _maybe_await(config.prepare_next_turn(next_turn_context))
                if snapshot:
                    current_context = _field(snapshot, "context") or current_context
                    next_model = _field(snapshot, "model")
                    next_thinking = _field(snapshot, "thinking_level")
                    updates: dict[str, Any] = {}
                    if next_model is not None:
                        updates["model"] = next_model
                    if next_thinking is not None:
                        updates["reasoning"] = None if next_thinking == "off" else next_thinking
                    if updates:
                        config = config.model_copy(update=updates)

            if config.should_stop_after_turn:
                stop = await _maybe_await(
                    config.should_stop_after_turn(
                        ShouldStopAfterTurnContext(
                            message=message,
                            tool_results=tool_results,
                            context=current_context,
                            new_messages=new_messages,
                        )
                    )
                )
                if stop:
                    await _emit(emit, AgentEventAgentEnd(messages=new_messages))
                    return

            pending_messages = []
            if config.get_steering_messages:
                pending_messages = list(await _maybe_await(config.get_steering_messages()) or [])

        follow_up_messages: list[AgentMessage] = []
        if config.get_follow_up_messages:
            follow_up_messages = list(await _maybe_await(config.get_follow_up_messages()) or [])
        if follow_up_messages:
            pending_messages = follow_up_messages
            continue
        break

    await _emit(emit, AgentEventAgentEnd(messages=new_messages))


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    emit: AgentEventSink,
    stream_function: StreamFn,
) -> AssistantMessage:
    """Stream an assistant response. This is the AgentMessage → Message boundary."""
    messages = context.messages
    if config.transform_context:
        messages = await _maybe_await(config.transform_context(messages, cancel_event))

    llm_messages = await _maybe_await(config.convert_to_llm(messages))

    llm_context = Context(
        system_prompt=context.system_prompt or None,
        messages=llm_messages,
        tools=list(context.tools or []),
    )

    resolved_api_key = config.api_key
    if config.get_api_key:
        key_result = await _maybe_await(config.get_api_key(config.model.provider))
        resolved_api_key = key_result or resolved_api_key

    stream_opts = SimpleStreamOptions(
        reasoning=config.reasoning,
        thinking_budgets=config.thinking_budgets,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        signal=cancel_event,
        api_key=resolved_api_key,
        transport=config.transport,
        cache_retention=config.cache_retention,
        session_id=config.session_id,
        on_payload=config.on_payload,
        on_response=config.on_response,
        headers=config.headers,
        max_retry_delay_ms=config.max_retry_delay_ms,
        metadata=config.metadata,
        sampling_params=config.sampling_params,
        tool_choice=config.tool_choice,
        deferred=config.deferred,
    )

    response = stream_function(config.model, llm_context, stream_opts)
    # Coroutines (async def returning a stream) must be awaited. Async generators
    # are iterable but not awaitable.
    if inspect.isawaitable(response):
        response = await response

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        if event.type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            await _emit(emit, AgentEventMessageStart(message=_copy_message(partial_message)))

        elif event.type in (
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ):
            if partial_message is not None:
                partial_message = event.partial
                context.messages[-1] = partial_message
                await _emit(
                    emit,
                    AgentEventMessageUpdate(
                        message=_copy_message(partial_message),
                        assistant_message_event=event,
                    ),
                )

        elif event.type in ("done", "error"):
            final_message = await _final_stream_message(response, event)
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                await _emit(emit, AgentEventMessageStart(message=_copy_message(final_message)))
            await _emit(emit, AgentEventMessageEnd(message=final_message))
            return final_message

    final_message = await _final_stream_message(response, None)
    if final_message is None:
        if partial_message is not None:
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Request was aborted")
            return partial_message
        raise RuntimeError("Stream ended without a final message")

    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await _emit(emit, AgentEventMessageStart(message=_copy_message(final_message)))
    await _emit(emit, AgentEventMessageEnd(message=final_message))
    return final_message


async def _fail_tool_calls_from_truncated_message(
    tool_calls: list[AgentToolCall],
    emit: AgentEventSink,
) -> "ExecutedToolCallBatch":
    """Fail every tool call from a length-truncated assistant message."""
    messages: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _emit(
            emit,
            AgentEventToolStart(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            ),
        )
        finalized = FinalizedToolCallOutcome(
            tool_call=tool_call,
            result=_create_error_tool_result(
                f'Tool call "{tool_call.name}" was not executed: the response hit the '
                "output token limit, so its arguments may be truncated. Re-issue the "
                "tool call with complete arguments."
            ),
            is_error=True,
        )
        await _emit_tool_execution_end(finalized, emit)
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        messages.append(tool_result_message)
    return ExecutedToolCallBatch(messages=messages, terminate=False)


async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    emit: AgentEventSink,
) -> "ExecutedToolCallBatch":
    tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCall)]
    has_sequential = any(
        (t.execution_mode == "sequential")
        for tc in tool_calls
        for t in [next((x for x in (current_context.tools or []) if x.name == tc.name), None)]
        if t is not None
    )
    if config.tool_execution == "sequential" or has_sequential:
        return await _execute_tool_calls_sequential(
            current_context, assistant_message, tool_calls, config, cancel_event, emit
        )
    return await _execute_tool_calls_parallel(
        current_context, assistant_message, tool_calls, config, cancel_event, emit
    )


@dataclass
class ExecutedToolCallBatch:
    messages: list[ToolResultMessage]
    terminate: bool


@dataclass
class PreparedToolCall:
    kind: str  # "prepared"
    tool_call: AgentToolCall
    tool: AgentTool
    args: Any


@dataclass
class ImmediateToolCallOutcome:
    kind: str  # "immediate"
    result: AgentToolResult
    is_error: bool


@dataclass
class ExecutedToolCallOutcome:
    result: AgentToolResult
    is_error: bool


@dataclass
class FinalizedToolCallOutcome:
    tool_call: AgentToolCall
    result: AgentToolResult
    is_error: bool


def _should_terminate_tool_batch(finalized_calls: list[FinalizedToolCallOutcome]) -> bool:
    return bool(finalized_calls) and all(
        finalized.result.terminate is True for finalized in finalized_calls
    )


def _prepare_tool_call_arguments(tool: AgentTool, tool_call: AgentToolCall) -> AgentToolCall:
    if not tool.prepare_arguments:
        return tool_call
    prepared_arguments = tool.prepare_arguments(tool_call.arguments)
    if prepared_arguments is tool_call.arguments:
        return tool_call
    return tool_call.model_copy(update={"arguments": prepared_arguments})


async def _prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: AgentToolCall,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
) -> PreparedToolCall | ImmediateToolCallOutcome:
    tool = next((t for t in (current_context.tools or []) if t.name == tool_call.name), None)
    if not tool:
        return ImmediateToolCallOutcome(
            kind="immediate",
            result=_create_error_tool_result(f"Tool {tool_call.name} not found"),
            is_error=True,
        )

    try:
        prepared_tool_call = _prepare_tool_call_arguments(tool, tool_call)
        validated_args = validate_tool_arguments(tool, prepared_tool_call)
        if config.before_tool_call:
            before_result = await _maybe_await(
                config.before_tool_call(
                    BeforeToolCallContext(
                        assistant_message=assistant_message,
                        tool_call=tool_call,
                        args=validated_args,
                        context=current_context,
                    ),
                    cancel_event,
                )
            )
            if _is_aborted(cancel_event):
                return ImmediateToolCallOutcome(
                    kind="immediate",
                    result=_create_error_tool_result("Operation aborted"),
                    is_error=True,
                )
            if before_result and _field(before_result, "block"):
                result = _create_error_tool_result(
                    _field(before_result, "reason") or "Tool execution was blocked"
                )
                if _field(before_result, "terminate") is True:
                    result.terminate = True
                return ImmediateToolCallOutcome(kind="immediate", result=result, is_error=True)
        if _is_aborted(cancel_event):
            return ImmediateToolCallOutcome(
                kind="immediate",
                result=_create_error_tool_result("Operation aborted"),
                is_error=True,
            )
        return PreparedToolCall(kind="prepared", tool_call=tool_call, tool=tool, args=validated_args)
    except Exception as error:
        return ImmediateToolCallOutcome(
            kind="immediate",
            result=_create_error_tool_result(str(error)),
            is_error=True,
        )


async def _execute_prepared_tool_call(
    prepared: PreparedToolCall,
    cancel_event: asyncio.Event | None,
    emit: AgentEventSink,
) -> ExecutedToolCallOutcome:
    update_events: list[asyncio.Task[None] | Awaitable[None]] = []
    accepting_updates = True

    def on_update(partial_result: AgentToolResult) -> None:
        if not accepting_updates:
            return
        update_events.append(
            asyncio.ensure_future(
                _emit(
                    emit,
                    AgentEventToolUpdate(
                        tool_call_id=prepared.tool_call.id,
                        tool_name=prepared.tool_call.name,
                        args=prepared.tool_call.arguments,
                        partial_result=partial_result,
                    ),
                )
            )
        )

    try:
        result = await prepared.tool.execute(
            prepared.tool_call.id,
            prepared.args,
            cancel_event,
            on_update,
        )
        accepting_updates = False
        if update_events:
            await asyncio.gather(*update_events)
        return ExecutedToolCallOutcome(result=result, is_error=False)
    except Exception as error:
        accepting_updates = False
        if update_events:
            await asyncio.gather(*update_events)
        return ExecutedToolCallOutcome(
            result=_create_error_tool_result(str(error)),
            is_error=True,
        )
    finally:
        accepting_updates = False


async def _finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: PreparedToolCall,
    executed: ExecutedToolCallOutcome,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
) -> FinalizedToolCallOutcome:
    result = executed.result
    is_error = executed.is_error

    if config.after_tool_call:
        try:
            after_result = await _maybe_await(
                config.after_tool_call(
                    AfterToolCallContext(
                        assistant_message=assistant_message,
                        tool_call=prepared.tool_call,
                        args=prepared.args,
                        result=result,
                        is_error=is_error,
                        context=current_context,
                    ),
                    cancel_event,
                )
            )
            if after_result:
                result = AgentToolResult(
                    content=_override(after_result, "content", result.content),
                    details=_override(after_result, "details", result.details),
                    usage=_override(after_result, "usage", result.usage),
                    added_tool_names=result.added_tool_names,
                    terminate=_override(after_result, "terminate", result.terminate),
                )
                is_error = _override(after_result, "is_error", is_error)
        except Exception as error:
            result = _create_error_tool_result(str(error))
            is_error = True

    return FinalizedToolCallOutcome(tool_call=prepared.tool_call, result=result, is_error=is_error)


async def _execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[AgentToolCall],
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    emit: AgentEventSink,
) -> ExecutedToolCallBatch:
    finalized_calls: list[FinalizedToolCallOutcome] = []
    messages: list[ToolResultMessage] = []

    for tool_call in tool_calls:
        await _emit(
            emit,
            AgentEventToolStart(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            ),
        )

        preparation = await _prepare_tool_call(
            current_context, assistant_message, tool_call, config, cancel_event
        )
        if isinstance(preparation, ImmediateToolCallOutcome):
            finalized = FinalizedToolCallOutcome(
                tool_call=tool_call,
                result=preparation.result,
                is_error=preparation.is_error,
            )
        else:
            executed = await _execute_prepared_tool_call(preparation, cancel_event, emit)
            finalized = await _finalize_executed_tool_call(
                current_context, assistant_message, preparation, executed, config, cancel_event
            )

        await _emit_tool_execution_end(finalized, emit)
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        finalized_calls.append(finalized)
        messages.append(tool_result_message)

        if _is_aborted(cancel_event):
            break

    return ExecutedToolCallBatch(
        messages=messages,
        terminate=_should_terminate_tool_batch(finalized_calls),
    )


async def _execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[AgentToolCall],
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    emit: AgentEventSink,
) -> ExecutedToolCallBatch:
    entries: list[FinalizedToolCallOutcome | Callable[[], Awaitable[FinalizedToolCallOutcome]]] = []

    for tool_call in tool_calls:
        await _emit(
            emit,
            AgentEventToolStart(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            ),
        )

        preparation = await _prepare_tool_call(
            current_context, assistant_message, tool_call, config, cancel_event
        )
        if isinstance(preparation, ImmediateToolCallOutcome):
            finalized = FinalizedToolCallOutcome(
                tool_call=tool_call,
                result=preparation.result,
                is_error=preparation.is_error,
            )
            await _emit_tool_execution_end(finalized, emit)
            entries.append(finalized)
            if _is_aborted(cancel_event):
                break
            continue

        async def _run(prep: PreparedToolCall = preparation) -> FinalizedToolCallOutcome:
            executed = await _execute_prepared_tool_call(prep, cancel_event, emit)
            finalized_inner = await _finalize_executed_tool_call(
                current_context, assistant_message, prep, executed, config, cancel_event
            )
            await _emit_tool_execution_end(finalized_inner, emit)
            return finalized_inner

        entries.append(_run)
        if _is_aborted(cancel_event):
            break

    async def _resolve(
        entry: FinalizedToolCallOutcome | Callable[[], Awaitable[FinalizedToolCallOutcome]],
    ) -> FinalizedToolCallOutcome:
        if callable(entry):
            return await entry()
        return entry

    ordered = await asyncio.gather(*[_resolve(entry) for entry in entries])
    messages: list[ToolResultMessage] = []
    for finalized in ordered:
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        messages.append(tool_result_message)

    return ExecutedToolCallBatch(
        messages=messages,
        terminate=_should_terminate_tool_batch(list(ordered)),
    )


def _create_error_tool_result(message: str) -> AgentToolResult:
    return AgentToolResult(
        content=[TextContent(type="text", text=message)],
        details={},
    )


async def _emit_tool_execution_end(
    finalized: FinalizedToolCallOutcome,
    emit: AgentEventSink,
) -> None:
    await _emit(
        emit,
        AgentEventToolEnd(
            tool_call_id=finalized.tool_call.id,
            tool_name=finalized.tool_call.name,
            result=finalized.result,
            is_error=finalized.is_error,
        ),
    )


def _create_tool_result_message(finalized: FinalizedToolCallOutcome) -> ToolResultMessage:
    added = finalized.result.added_tool_names
    return ToolResultMessage(
        role="toolResult",
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        content=list(finalized.result.content or []),
        details=finalized.result.details,
        usage=finalized.result.usage,
        added_tool_names=list(added) if added else None,
        is_error=finalized.is_error,
        timestamp=int(time.time() * 1000),
    )


async def _emit_tool_result_message(
    tool_result_message: ToolResultMessage,
    emit: AgentEventSink,
) -> None:
    await _emit(emit, AgentEventMessageStart(message=tool_result_message))
    await _emit(emit, AgentEventMessageEnd(message=tool_result_message))


# ─── helpers ──────────────────────────────────────────────────────────────────


async def _emit(emit: AgentEventSink, event: AgentEvent) -> None:
    result = emit(event)
    if inspect.isawaitable(result):
        await result


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _is_aborted(cancel_event: asyncio.Event | None) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _override(after: Any, name: str, current: Any) -> Any:
    """Field-by-field merge: missing/None keeps current (nullish coalescing)."""
    if isinstance(after, dict):
        if name not in after:
            return current
        value = after[name]
        return current if value is None else value
    if not hasattr(after, name):
        return current
    value = getattr(after, name)
    return current if value is None else value


def _copy_message(message: Any) -> Any:
    if hasattr(message, "model_copy"):
        return message.model_copy()
    return message


async def _final_stream_message(response: Any, event: Any) -> AssistantMessage | None:
    result_fn = getattr(response, "result", None)
    if callable(result_fn):
        try:
            maybe = result_fn()
            if inspect.isawaitable(maybe):
                return await maybe
            if maybe is not None:
                return maybe
        except Exception:
            pass
    if event is None:
        return None
    if event.type == "done":
        return event.message
    return getattr(event, "error", None)
