"""
OpenAI Chat Completions API provider — mirrors packages/ai/src/providers/openai-completions.ts
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator

import openai as _openai

from ..types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    EventDone,
    EventError,
    EventStart,
    EventTextDelta,
    EventTextEnd,
    EventTextStart,
    EventThinkingDelta,
    EventThinkingEnd,
    EventThinkingStart,
    EventToolCallDelta,
    EventToolCallEnd,
    EventToolCallStart,
    ImageContent,
    Model,
    OpenAICompletionsCompat,
    SimpleStreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from ..utils.deferred_tools import split_deferred_tools
from ..utils.json_parse import parse_partial_json
from ..utils.pi_user_agent import get_pi_user_agent
from .transform_messages import transform_messages as _transform_messages


_COMPAT_CAMEL_TO_SNAKE = {
    "supportsStore": "supports_store",
    "supportsDeveloperRole": "supports_developer_role",
    "supportsReasoningEffort": "supports_reasoning_effort",
    "supportsUsageInStreaming": "supports_usage_in_streaming",
    "supportsFinishReason": "supports_finish_reason",
    "maxTokensField": "max_tokens_field",
    "requiresToolResultName": "requires_tool_result_name",
    "requiresAssistantAfterToolResult": "requires_assistant_after_tool_result",
    "requiresThinkingAsText": "requires_thinking_as_text",
    "requiresReasoningContentOnAssistantMessages": "requires_reasoning_content_on_assistant_messages",
    "thinkingFormat": "thinking_format",
    "chatTemplateKwargs": "chat_template_kwargs",
    "chatTemplateArgs": "chat_template_args",
    "openRouterRouting": "open_router_routing",
    "vercelGatewayRouting": "vercel_gateway_routing",
    "supportsOpenAIGrammarTools": "supports_openai_grammar_tools",
    "supportsStrictMode": "supports_strict_mode",
    "cacheControlFormat": "cache_control_format",
    "sendSessionAffinityHeaders": "send_session_affinity_headers",
    "deferredToolsMode": "deferred_tools_mode",
    "sessionAffinityFormat": "session_affinity_format",
    "supportsLongCacheRetention": "supports_long_cache_retention",
}


def _compat_from_mapping(raw: dict[str, Any]) -> OpenAICompletionsCompat:
    fields = OpenAICompletionsCompat.__dataclass_fields__
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        snake = _COMPAT_CAMEL_TO_SNAKE.get(key, key)
        if snake in fields:
            kwargs[snake] = value
    return OpenAICompletionsCompat(**{k: v for k, v in kwargs.items() if k in fields})


def _get_compat(model: Model) -> OpenAICompletionsCompat:
    compat = getattr(model, "compat", None)
    if isinstance(compat, OpenAICompletionsCompat):
        return compat
    if isinstance(compat, dict):
        detected = _detect_openai_compat(model)
        override = _compat_from_mapping(compat)
        specified = {_COMPAT_CAMEL_TO_SNAKE.get(k, k) for k in compat}
        updates = {k: getattr(override, k) for k in specified if hasattr(override, k)}
        return OpenAICompletionsCompat(**{**detected.__dict__, **updates})
    return _detect_openai_compat(model)


def _detect_openai_compat(model: Model) -> OpenAICompletionsCompat:
    """Auto-detect compat settings from model baseUrl / properties."""
    url = (model.base_url or "").lower()
    is_openai = "api.openai.com" in url or not url
    is_openrouter = "openrouter.ai" in url
    return OpenAICompletionsCompat(
        supports_store=is_openai,
        supports_developer_role=is_openai or bool(model.reasoning),
        supports_reasoning_effort=is_openai or bool(model.reasoning),
        supports_usage_in_streaming=True,
        max_tokens_field="max_completion_tokens" if (is_openai and model.reasoning) else None,
        requires_tool_result_name=is_openrouter,
        requires_assistant_after_tool_result=False,
        requires_thinking_as_text=False,
        thinking_format="openai",
        supports_strict_mode=True,
    )


def _uses_max_completion_tokens(model: Model) -> bool:
    compat = _get_compat(model)
    if compat.max_tokens_field:
        return compat.max_tokens_field == "max_completion_tokens"
    return bool(getattr(model, "reasoning", False))


def _build_messages(context: Context, model: Model) -> list[dict[str, Any]]:
    """Convert Context messages to OpenAI Chat Completions format."""
    compat = _get_compat(model)
    result: list[dict[str, Any]] = []

    if context.system_prompt:
        role = "developer" if compat.supports_developer_role else "system"
        result.append({"role": role, "content": context.system_prompt})

    for msg in context.messages:
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                result.append({"role": "user", "content": msg.content})
            else:
                content_blocks: list[dict[str, Any]] = []
                for block in msg.content:
                    if isinstance(block, TextContent):
                        content_blocks.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageContent):
                        content_blocks.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{block.mime_type};base64,{block.data}",
                            },
                        })
                result.append({"role": "user", "content": content_blocks})

        elif isinstance(msg, AssistantMessage):
            tool_calls = [c for c in msg.content if isinstance(c, ToolCall)]
            text_parts = [c for c in msg.content if isinstance(c, TextContent)]
            text = " ".join(t.text for t in text_parts) if text_parts else None

            if tool_calls:
                tc_list = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]
                entry: dict[str, Any] = {"role": "assistant", "tool_calls": tc_list}
                if text:
                    entry["content"] = text
                result.append(entry)
            else:
                result.append({"role": "assistant", "content": text or ""})

        elif isinstance(msg, ToolResultMessage):
            content_text = " ".join(
                b.text for b in msg.content if isinstance(b, TextContent)
            )
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": content_text,
            }
            if compat.requires_tool_result_name and hasattr(msg, "tool_name") and msg.tool_name:
                tool_msg["name"] = msg.tool_name

            if compat.requires_assistant_after_tool_result:
                next_idx = context.messages.index(msg) + 1 if msg in context.messages else -1
                needs_bridge = next_idx >= 0 and next_idx < len(context.messages) and isinstance(context.messages[next_idx], UserMessage)
                if needs_bridge:
                    result.append(tool_msg)
                    result.append({"role": "assistant", "content": ""})
                    continue

            result.append(tool_msg)

    return result


def _build_tools(context: Context, model: Model) -> list[dict[str, Any]] | None:
    if not context.tools:
        return None
    compat = _get_compat(model)
    result = []
    for tool in context.tools:
        fn: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if compat.supports_strict_mode:
            fn["strict"] = False
        result.append({"type": "function", "function": fn})
    return result


def _make_empty_assistant(model: Model) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


async def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """Stream a response from the OpenAI Chat Completions API."""
    opts = options or SimpleStreamOptions()

    base_url = model.base_url if model.base_url != "https://api.openai.com/v1" else None
    extra_headers: dict[str, str] = {"User-Agent": get_pi_user_agent()}
    if opts.headers:
        extra_headers.update({k: v for k, v in opts.headers.items() if v is not None})
    extra_headers.update(model.headers or {})

    client = _openai.AsyncOpenAI(
        api_key=opts.api_key or None,
        base_url=base_url,
        default_headers=extra_headers or None,
    )

    # Transform messages for cross-provider compatibility
    transformed_msgs = _transform_messages(context.messages, model)
    compat = _get_compat(model)
    immediate_tools, _deferred = split_deferred_tools(
        Context(system_prompt=context.system_prompt, messages=transformed_msgs, tools=context.tools),
        bool(compat.deferred_tools_mode),
    )
    transformed_context = Context(
        system_prompt=context.system_prompt,
        messages=transformed_msgs,
        tools=immediate_tools,
    )

    messages = _build_messages(transformed_context, model)
    tools = _build_tools(transformed_context, model)

    params: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": True,
    }

    if compat.supports_usage_in_streaming:
        params["stream_options"] = {"include_usage": True}

    if compat.supports_store:
        params["store"] = True

    if opts.max_tokens:
        if _uses_max_completion_tokens(model):
            params["max_completion_tokens"] = opts.max_tokens
        else:
            params["max_tokens"] = opts.max_tokens

    if opts.temperature is not None:
        params["temperature"] = opts.temperature

    if tools:
        params["tools"] = tools

    if model.reasoning:
        effort = opts.reasoning
        mapped_effort = None
        if effort and model.thinking_level_map and effort in model.thinking_level_map:
            mapped_effort = model.thinking_level_map[effort]
        elif effort and compat.reasoning_effort_map:
            mapped_effort = compat.reasoning_effort_map.get(effort)
        elif effort:
            default_map = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}
            mapped_effort = default_map.get(effort, effort)

        fmt = compat.thinking_format or "openai"
        if fmt == "zai":
            params["thinking"] = {"type": "enabled", "clear_thinking": False} if effort else {"type": "disabled"}
            if effort and compat.supports_reasoning_effort and isinstance(mapped_effort, str):
                params["reasoning_effort"] = mapped_effort
        elif fmt == "qwen":
            params["enable_thinking"] = bool(effort)
            if effort and compat.supports_reasoning_effort and isinstance(mapped_effort, str):
                params["reasoning_effort"] = mapped_effort
        elif fmt == "qwen-chat-template":
            params["chat_template_kwargs"] = {"enable_thinking": bool(effort), "preserve_thinking": True}
        elif fmt == "deepseek":
            if effort:
                params["thinking"] = {"type": "enabled"}
            else:
                params["thinking"] = {"type": "disabled"}
            if effort and compat.supports_reasoning_effort and isinstance(mapped_effort, str):
                params["reasoning_effort"] = mapped_effort
        elif fmt == "openrouter":
            if effort and isinstance(mapped_effort, str):
                params["reasoning"] = {"effort": mapped_effort}
            else:
                params["reasoning"] = {"effort": "none"}
        elif fmt == "together":
            params["reasoning"] = {"enabled": bool(effort)}
            if effort and compat.supports_reasoning_effort and isinstance(mapped_effort, str):
                params["reasoning_effort"] = mapped_effort
        elif fmt == "string-thinking":
            params["thinking"] = mapped_effort if effort and isinstance(mapped_effort, str) else "none"
        elif fmt == "ant-ling":
            if effort and isinstance(mapped_effort, str):
                params["reasoning"] = {"effort": mapped_effort}
        elif effort and compat.supports_reasoning_effort and isinstance(mapped_effort, str):
            params["reasoning_effort"] = mapped_effort

    if compat.open_router_routing:
        route: dict[str, Any] = {}
        if compat.open_router_routing.only:
            route["only"] = compat.open_router_routing.only
        if compat.open_router_routing.order:
            route["order"] = compat.open_router_routing.order
        if route:
            params["route"] = route

    partial = _make_empty_assistant(model)
    content_blocks: list[Any] = []
    text_index = -1
    thinking_index = -1
    tool_indices: dict[str, int] = {}
    tool_arg_buffers: dict[str, str] = {}
    usage = Usage()
    finish_reason: str | None = None

    yield EventStart(type="start", partial=partial)

    try:
        async with await client.chat.completions.create(**params) as stream:
            async for chunk in stream:
                # Process usage from chunks
                if chunk.usage:
                    u = chunk.usage
                    usage = Usage(
                        input=getattr(u, "prompt_tokens", 0) or 0,
                        output=getattr(u, "completion_tokens", 0) or 0,
                        total_tokens=getattr(u, "total_tokens", 0) or 0,
                    )
                    # Check for reasoning tokens
                    details = getattr(u, "completion_tokens_details", None)
                    if details:
                        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
                        if reasoning_tokens:
                            usage.output = (getattr(u, "completion_tokens", 0) or 0) - reasoning_tokens

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                # Fallback: some providers (e.g. Moonshot/Kimi) return usage
                # in choice.usage instead of the standard chunk.usage
                if not chunk.usage:
                    choice_usage = getattr(choice, "usage", None)
                    if choice_usage:
                        usage = Usage(
                            input=getattr(choice_usage, "prompt_tokens", 0) or 0,
                            output=getattr(choice_usage, "completion_tokens", 0) or 0,
                            total_tokens=getattr(choice_usage, "total_tokens", 0) or 0,
                        )

                delta = choice.delta
                finish_reason = choice.finish_reason

                # Reasoning / thinking content (for o1/o3 models)
                reasoning_content = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning_content:
                    if thinking_index == -1:
                        thinking_index = len(content_blocks)
                        content_blocks.append(ThinkingContent(type="thinking", thinking=""))
                        partial = partial.model_copy(update={"content": list(content_blocks)})
                        yield EventThinkingStart(type="thinking_start", content_index=thinking_index, partial=partial)

                    content_blocks[thinking_index] = ThinkingContent(
                        type="thinking",
                        thinking=content_blocks[thinking_index].thinking + reasoning_content,
                    )
                    partial = partial.model_copy(update={"content": list(content_blocks)})
                    yield EventThinkingDelta(
                        type="thinking_delta",
                        content_index=thinking_index,
                        delta=reasoning_content,
                        partial=partial,
                    )

                # Text delta
                if delta.content:
                    # Close thinking block if transitioning to text
                    if thinking_index >= 0 and text_index == -1:
                        yield EventThinkingEnd(
                            type="thinking_end",
                            content_index=thinking_index,
                            content=content_blocks[thinking_index].thinking,
                            partial=partial,
                        )

                    if text_index == -1:
                        text_index = len(content_blocks)
                        content_blocks.append(TextContent(type="text", text=""))
                        partial = partial.model_copy(update={"content": list(content_blocks)})
                        yield EventTextStart(type="text_start", content_index=text_index, partial=partial)

                    content_blocks[text_index] = TextContent(
                        type="text",
                        text=content_blocks[text_index].text + delta.content,
                    )
                    partial = partial.model_copy(update={"content": list(content_blocks)})
                    yield EventTextDelta(
                        type="text_delta",
                        content_index=text_index,
                        delta=delta.content,
                        partial=partial,
                    )

                # Tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        tc_id = tc_delta.id or ""
                        idx_key = str(tc_delta.index)

                        if idx_key not in tool_indices:
                            idx = len(content_blocks)
                            tool_indices[idx_key] = idx
                            tool_arg_buffers[idx_key] = ""
                            content_blocks.append(ToolCall(
                                type="toolCall",
                                id=tc_id or f"call_{idx}",
                                name=tc_delta.function.name or "",
                                arguments={},
                            ))
                            partial = partial.model_copy(update={"content": list(content_blocks)})
                            yield EventToolCallStart(type="toolcall_start", content_index=idx, partial=partial)

                        if tc_delta.function and tc_delta.function.arguments:
                            tool_arg_buffers[idx_key] += tc_delta.function.arguments
                            partial = partial.model_copy(update={"content": list(content_blocks)})
                            yield EventToolCallDelta(
                                type="toolcall_delta",
                                content_index=tool_indices[idx_key],
                                delta=tc_delta.function.arguments,
                                partial=partial,
                            )

                if finish_reason:
                    # Finalize thinking
                    if thinking_index >= 0 and text_index == -1:
                        yield EventThinkingEnd(
                            type="thinking_end",
                            content_index=thinking_index,
                            content=content_blocks[thinking_index].thinking,
                            partial=partial,
                        )

                    # Finalize text block
                    if text_index >= 0:
                        yield EventTextEnd(
                            type="text_end",
                            content_index=text_index,
                            content=content_blocks[text_index].text,
                            partial=partial,
                        )

                    # Finalize tool calls
                    for idx_key, idx in tool_indices.items():
                        raw = tool_arg_buffers.get(idx_key, "{}")
                        parsed = parse_partial_json(raw) or {}
                        tc = content_blocks[idx]
                        content_blocks[idx] = ToolCall(
                            type="toolCall",
                            id=tc.id,
                            name=tc.name,
                            arguments=parsed,
                        )
                        partial = partial.model_copy(update={"content": list(content_blocks)})
                        yield EventToolCallEnd(
                            type="toolcall_end",
                            content_index=idx,
                            tool_call=content_blocks[idx],
                            partial=partial,
                        )

        # Build final message
        compat = _get_compat(model)
        stop_reason_map = {"stop": "stop", "length": "length", "tool_calls": "toolUse"}
        provider_error: str | None = None
        has_finish = bool(finish_reason)
        if not has_finish and not compat.supports_finish_reason:
            stop_reason = "toolUse" if tool_indices else "stop"
        elif compat.supports_finish_reason and not has_finish:
            raise RuntimeError("Stream ended without finish_reason")
        elif finish_reason in stop_reason_map:
            stop_reason = stop_reason_map[finish_reason]
        elif finish_reason:
            stop_reason = "error"
            provider_error = f"Provider finish_reason: {finish_reason}"
        else:
            stop_reason = "stop"
        if tool_indices and stop_reason == "stop":
            stop_reason = "toolUse"

        signal = getattr(opts, "signal", None)
        if signal and callable(getattr(signal, "is_set", None)) and signal.is_set():
            stop_reason = "aborted"

        final = AssistantMessage(
            role="assistant",
            content=content_blocks,
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=usage,
            stop_reason=stop_reason,
            error_message=provider_error if stop_reason == "error" else None,
            timestamp=int(time.time() * 1000),
        )
        
        # EventDone only accepts "stop", "length", "toolUse"
        # For "error" or "aborted", emit EventError instead
        if stop_reason in ("error", "aborted"):
            yield EventError(type="error", reason=stop_reason, error=final)
        else:
            yield EventDone(type="done", reason=stop_reason, message=final)

    except _openai.APIError as e:
        error_msg = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="error",
            error_message=str(e),
            timestamp=int(time.time() * 1000),
        )
        yield EventError(type="error", reason="error", error=error_msg)
    except Exception as e:
        error_msg = AssistantMessage(
            role="assistant",
            content=content_blocks or [TextContent(type="text", text="")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=usage,
            stop_reason="error",
            error_message=str(e),
            timestamp=int(time.time() * 1000),
        )
        yield EventError(type="error", reason="error", error=error_msg)
