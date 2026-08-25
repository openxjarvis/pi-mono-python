"""
Mistral Conversations API provider — mirrors packages/ai/src/providers/mistral.ts

Native Mistral SDK provider with:
- KV-cache reuse via x-affinity header (prefix caching)
- Tool call ID normalization (9-char alphanumeric)
- Native thinking/reasoning content chunks
- promptMode: "reasoning" support
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator

from ..env_api_keys import get_env_api_key
from ..models import calculate_cost
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
    Message,
    Model,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
    UsageCost,
)
from ..utils.hash import short_hash
from ..utils.json_parse import parse_streaming_json
from ..utils.sanitize_unicode import sanitize_surrogates
from .simple_options import build_base_options, clamp_reasoning

MISTRAL_TOOL_CALL_ID_LENGTH = 9
MAX_MISTRAL_ERROR_BODY_CHARS = 4000


def _create_output(model: Model) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(cost=UsageCost()),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


def _derive_mistral_tool_call_id(id_: str, attempt: int) -> str:
    import re
    normalized = re.sub(r"[^a-zA-Z0-9]", "", id_)
    if attempt == 0 and len(normalized) == MISTRAL_TOOL_CALL_ID_LENGTH:
        return normalized
    seed_base = normalized or id_
    seed = seed_base if attempt == 0 else f"{seed_base}:{attempt}"
    return short_hash(seed).replace("/", "").replace("+", "")[:MISTRAL_TOOL_CALL_ID_LENGTH]


def _create_mistral_tool_call_id_normalizer():
    id_map: dict[str, str] = {}
    reverse_map: dict[str, str] = {}

    def normalize(id_: str) -> str:
        existing = id_map.get(id_)
        if existing:
            return existing
        attempt = 0
        while True:
            candidate = _derive_mistral_tool_call_id(id_, attempt)
            owner = reverse_map.get(candidate)
            if owner is None or owner == id_:
                id_map[id_] = candidate
                reverse_map[candidate] = id_
                return candidate
            attempt += 1

    return normalize


def _format_mistral_error(error: Exception) -> str:
    status_code = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    body_text = body.strip() if isinstance(body, str) else None

    if status_code is not None and body_text:
        if len(body_text) > MAX_MISTRAL_ERROR_BODY_CHARS:
            body_text = f"{body_text[:MAX_MISTRAL_ERROR_BODY_CHARS]}... [truncated {len(body_text) - MAX_MISTRAL_ERROR_BODY_CHARS} chars]"
        return f"Mistral API error ({status_code}): {body_text}"
    if status_code is not None:
        return f"Mistral API error ({status_code}): {error}"
    return str(error)


def _map_chat_stop_reason(reason: str | None) -> StopReason:
    if reason is None:
        return "stop"
    mapping: dict[str, StopReason] = {
        "stop": "stop",
        "length": "length",
        "model_length": "length",
        "tool_calls": "toolUse",
        "error": "error",
    }
    return mapping.get(reason, "stop")


def _to_chat_messages(messages: list[Message], supports_images: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                result.append({"role": "user", "content": sanitize_surrogates(msg.content)})
                continue
            had_images = any(getattr(c, "type", None) == "image" for c in msg.content)
            content_parts: list[dict[str, Any]] = []
            for item in msg.content:
                if item.type == "text":
                    content_parts.append({"type": "text", "text": sanitize_surrogates(item.text)})
                elif item.type == "image" and supports_images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": f"data:{item.mime_type};base64,{item.data}",
                    })
            if content_parts:
                result.append({"role": "user", "content": content_parts})
            elif had_images and not supports_images:
                result.append({"role": "user", "content": "(image omitted: model does not support images)"})
            continue

        if msg.role == "assistant":
            content_parts_a: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []
            for block in msg.content:
                if block.type == "text":
                    if block.text.strip():
                        content_parts_a.append({"type": "text", "text": sanitize_surrogates(block.text)})
                elif block.type == "thinking":
                    if block.thinking.strip():
                        content_parts_a.append({
                            "type": "thinking",
                            "thinking": [{"type": "text", "text": sanitize_surrogates(block.thinking)}],
                        })
                elif block.type == "toolCall":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.arguments or {}),
                        },
                    })
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if content_parts_a:
                assistant_msg["content"] = content_parts_a
            if tool_calls:
                assistant_msg["toolCalls"] = tool_calls
            if content_parts_a or tool_calls:
                result.append(assistant_msg)
            continue

        # tool_result
        text_parts = []
        has_images = False
        for part in msg.content:
            if part.type == "text":
                text_parts.append(sanitize_surrogates(part.text))
            elif part.type == "image":
                has_images = True
        text_result = "\n".join(text_parts)
        tool_text = _build_tool_result_text(text_result, has_images, supports_images, getattr(msg, "is_error", False))
        tool_content: list[dict[str, Any]] = [{"type": "text", "text": tool_text}]
        if supports_images:
            for part in msg.content:
                if part.type == "image":
                    tool_content.append({
                        "type": "image_url",
                        "image_url": f"data:{part.mime_type};base64,{part.data}",
                    })
        result.append({
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "name": msg.tool_name,
            "content": tool_content,
        })

    return result


def _build_tool_result_text(text: str, has_images: bool, supports_images: bool, is_error: bool) -> str:
    trimmed = text.strip()
    error_prefix = "[tool error] " if is_error else ""
    if trimmed:
        image_suffix = "\n[tool image omitted: model does not support images]" if (has_images and not supports_images) else ""
        return f"{error_prefix}{trimmed}{image_suffix}"
    if has_images:
        if supports_images:
            return "[tool error] (see attached image)" if is_error else "(see attached image)"
        return "[tool error] (image omitted: model does not support images)" if is_error else "(image omitted: model does not support images)"
    return "[tool error] (no tool output)" if is_error else "(no tool output)"


def _to_function_tools(tools: list) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": False,
            },
        }
        for tool in tools
    ]


async def stream_mistral(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """Stream responses from Mistral using the native Mistral SDK."""
    from mistralai import Mistral

    output = _create_output(model)

    try:
        api_key = (options.api_key if options else None) or get_env_api_key(model.provider)
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")

        client = Mistral(api_key=api_key, server_url=model.base_url or "https://api.mistral.ai")

        normalizer = _create_mistral_tool_call_id_normalizer()
        from .transform_messages import transform_messages
        transformed = transform_messages(
            context.messages,
            model,
            normalize_tool_call_id=lambda id_, _model, _msg: normalizer(id_),
        )

        chat_messages = _to_chat_messages(transformed, "image" in (model.input or []))

        if context.system_prompt:
            chat_messages.insert(0, {
                "role": "system",
                "content": sanitize_surrogates(context.system_prompt),
            })

        payload: dict[str, Any] = {
            "model": model.id,
            "stream": True,
            "messages": chat_messages,
        }
        if context.tools:
            payload["tools"] = _to_function_tools(context.tools)
        if options and options.temperature is not None:
            payload["temperature"] = options.temperature
        if options and options.max_tokens is not None:
            payload["max_tokens"] = options.max_tokens

        prompt_mode = getattr(options, "prompt_mode", None)
        if prompt_mode:
            payload["prompt_mode"] = prompt_mode

        headers: dict[str, str] = {}
        if model.headers:
            headers.update(model.headers)
        if options and options.headers:
            headers.update(options.headers)
        if options and options.session_id and "x-affinity" not in headers:
            headers["x-affinity"] = options.session_id

        yield EventStart(type="start", partial=output)

        current_block: TextContent | ThinkingContent | None = None
        tool_blocks_by_key: dict[str, int] = {}
        partial_args: dict[int, str] = {}

        def _block_index() -> int:
            return len(output.content) - 1

        async for event in await client.chat.stream_async(**payload):
            chunk = event.data

            if chunk.usage:
                output.usage.input = chunk.usage.prompt_tokens or 0
                output.usage.output = chunk.usage.completion_tokens or 0
                output.usage.cache_read = 0
                output.usage.cache_write = 0
                output.usage.total_tokens = chunk.usage.total_tokens or (output.usage.input + output.usage.output)
                calculate_cost(model, output.usage)

            if not chunk.choices:
                continue
            choice = chunk.choices[0]

            if choice.finish_reason:
                output.stop_reason = _map_chat_stop_reason(choice.finish_reason)

            delta = choice.delta
            if delta.content is not None:
                content_items = [delta.content] if isinstance(delta.content, str) else delta.content
                for item in content_items:
                    if isinstance(item, str):
                        text_delta = sanitize_surrogates(item)
                        if current_block is None or current_block.type != "text":
                            if current_block is not None:
                                if current_block.type == "text":
                                    yield EventTextEnd(type="text_end", content_index=_block_index(), content=current_block.text, partial=output)
                                elif current_block.type == "thinking":
                                    yield EventThinkingEnd(type="thinking_end", content_index=_block_index(), content=current_block.thinking, partial=output)
                            current_block = TextContent(type="text", text="")
                            output.content.append(current_block)
                            yield EventTextStart(type="text_start", content_index=_block_index(), partial=output)
                        current_block.text += text_delta
                        yield EventTextDelta(type="text_delta", content_index=_block_index(), delta=text_delta, partial=output)
                        continue

                    item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)

                    if item_type == "thinking":
                        thinking_parts = getattr(item, "thinking", None) or (item.get("thinking") if isinstance(item, dict) else [])
                        delta_text = "".join(
                            (getattr(p, "text", "") if not isinstance(p, dict) else p.get("text", ""))
                            for p in (thinking_parts or [])
                        )
                        thinking_delta = sanitize_surrogates(delta_text)
                        if not thinking_delta:
                            continue
                        if current_block is None or current_block.type != "thinking":
                            if current_block is not None:
                                if current_block.type == "text":
                                    yield EventTextEnd(type="text_end", content_index=_block_index(), content=current_block.text, partial=output)
                                elif current_block.type == "thinking":
                                    yield EventThinkingEnd(type="thinking_end", content_index=_block_index(), content=current_block.thinking, partial=output)
                            current_block = ThinkingContent(type="thinking", thinking="")
                            output.content.append(current_block)
                            yield EventThinkingStart(type="thinking_start", content_index=_block_index(), partial=output)
                        current_block.thinking += thinking_delta
                        yield EventThinkingDelta(type="thinking_delta", content_index=_block_index(), delta=thinking_delta, partial=output)
                        continue

                    if item_type == "text":
                        text_val = getattr(item, "text", "") if not isinstance(item, dict) else item.get("text", "")
                        text_delta = sanitize_surrogates(text_val)
                        if current_block is None or current_block.type != "text":
                            if current_block is not None:
                                if current_block.type == "text":
                                    yield EventTextEnd(type="text_end", content_index=_block_index(), content=current_block.text, partial=output)
                                elif current_block.type == "thinking":
                                    yield EventThinkingEnd(type="thinking_end", content_index=_block_index(), content=current_block.thinking, partial=output)
                            current_block = TextContent(type="text", text="")
                            output.content.append(current_block)
                            yield EventTextStart(type="text_start", content_index=_block_index(), partial=output)
                        current_block.text += text_delta
                        yield EventTextDelta(type="text_delta", content_index=_block_index(), delta=text_delta, partial=output)

            tool_calls = getattr(delta, "tool_calls", None) or []
            for tc in tool_calls:
                if current_block is not None:
                    if current_block.type == "text":
                        yield EventTextEnd(type="text_end", content_index=_block_index(), content=current_block.text, partial=output)
                    elif current_block.type == "thinking":
                        yield EventThinkingEnd(type="thinking_end", content_index=_block_index(), content=current_block.thinking, partial=output)
                    current_block = None

                call_id = tc.id if tc.id and tc.id != "null" else _derive_mistral_tool_call_id(f"toolcall:{getattr(tc, 'index', 0)}", 0)
                key = f"{call_id}:{getattr(tc, 'index', 0)}"
                existing_idx = tool_blocks_by_key.get(key)

                if existing_idx is not None and existing_idx < len(output.content):
                    block = output.content[existing_idx]
                else:
                    block = ToolCall(
                        type="toolCall",
                        id=call_id,
                        name=tc.function.name,
                        arguments={},
                    )
                    output.content.append(block)
                    idx = len(output.content) - 1
                    tool_blocks_by_key[key] = idx
                    partial_args[idx] = ""
                    yield EventToolCallStart(type="toolcall_start", content_index=idx, partial=output)

                tc_idx = tool_blocks_by_key[key]
                args_delta = tc.function.arguments if isinstance(tc.function.arguments, str) else json.dumps(tc.function.arguments or {})
                partial_args[tc_idx] = partial_args.get(tc_idx, "") + args_delta
                if isinstance(block, ToolCall):
                    block.arguments = parse_streaming_json(partial_args[tc_idx]) or {}
                yield EventToolCallDelta(type="toolcall_delta", content_index=tc_idx, delta=args_delta, partial=output)

        # Finish current block
        if current_block is not None:
            if current_block.type == "text":
                yield EventTextEnd(type="text_end", content_index=_block_index(), content=current_block.text, partial=output)
            elif current_block.type == "thinking":
                yield EventThinkingEnd(type="thinking_end", content_index=_block_index(), content=current_block.thinking, partial=output)

        # Finalize tool call blocks
        for key, idx in tool_blocks_by_key.items():
            block = output.content[idx]
            if isinstance(block, ToolCall):
                block.arguments = parse_streaming_json(partial_args.get(idx, "")) or {}
                yield EventToolCallEnd(type="toolcall_end", content_index=idx, tool_call=block, partial=output)

        yield EventDone(type="done", reason=output.stop_reason, message=output)

    except Exception as e:
        output.stop_reason = "error"
        output.error_message = _format_mistral_error(e)
        yield EventError(type="error", reason="error", error=output)


async def stream_simple_mistral(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """Maps SimpleStreamOptions to Mistral-native options."""
    api_key = (options.api_key if options else None) or get_env_api_key(model.provider)
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, options, api_key)
    reasoning = clamp_reasoning(getattr(options, "reasoning", None))

    if model.reasoning and reasoning:
        base.prompt_mode = "reasoning"

    async for event in stream_mistral(model, context, base):
        yield event
