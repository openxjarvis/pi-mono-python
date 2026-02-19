"""
Cross-provider message transformation — mirrors packages/ai/src/providers/transform-messages.ts

Converts messages between provider formats, handles thinking blocks,
orphaned tool calls, etc.
"""
from __future__ import annotations

import time
from typing import Any

from ..types import (
    AssistantMessage,
    Context,
    Message,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)


def transform_messages(context: Context, target_api: str) -> Context:
    """
    Transform messages in a context for a different provider.

    Handles:
    - Converting thinking blocks to text <thinking> delimiters
    - Handling orphaned tool calls (assistant msg with toolCall but no following toolResult)
    - Tool call ID normalization
    """
    messages = context.messages
    transformed: list[Message] = []

    for i, msg in enumerate(messages):
        if isinstance(msg, AssistantMessage):
            new_content: list[Any] = []
            for block in msg.content:
                if isinstance(block, ThinkingContent):
                    # Convert thinking to text for providers that don't support it
                    if _requires_thinking_as_text(target_api):
                        new_content.append(TextContent(
                            type="text",
                            text=f"<thinking>\n{block.thinking}\n</thinking>",
                        ))
                    else:
                        new_content.append(block)
                else:
                    new_content.append(block)

            transformed.append(AssistantMessage(
                role="assistant",
                content=new_content,
                api=msg.api,
                provider=msg.provider,
                model=msg.model,
                usage=msg.usage,
                stop_reason=msg.stop_reason,
                error_message=msg.error_message,
                timestamp=msg.timestamp,
            ))
        else:
            transformed.append(msg)

    # Handle orphaned tool calls: if last assistant message has tool calls
    # but no following tool results, insert synthetic error results
    transformed = _handle_orphaned_tool_calls(transformed)

    return Context(
        system_prompt=context.system_prompt,
        messages=transformed,
        tools=context.tools,
    )


def _requires_thinking_as_text(api: str) -> bool:
    """Check if the target API requires thinking as text blocks."""
    return api in ("openai-completions", "openai-responses", "google-generative-ai")


def _handle_orphaned_tool_calls(messages: list[Message]) -> list[Message]:
    """
    Insert synthetic error tool results for orphaned tool calls.

    An orphaned tool call is a toolCall in an assistant message
    that has no corresponding toolResult in the following messages.
    """
    result: list[Message] = list(messages)
    i = 0
    while i < len(result):
        msg = result[i]
        if isinstance(msg, AssistantMessage):
            tool_calls = [c for c in msg.content if isinstance(c, ToolCall)]
            if tool_calls:
                # Find existing tool results that follow
                existing_ids: set[str] = set()
                j = i + 1
                while j < len(result):
                    next_msg = result[j]
                    if isinstance(next_msg, ToolResultMessage):
                        existing_ids.add(next_msg.tool_call_id)
                        j += 1
                    else:
                        break

                # Insert synthetic results for missing ones
                insert_pos = i + 1
                for tc in tool_calls:
                    if tc.id not in existing_ids:
                        synthetic = ToolResultMessage(
                            role="toolResult",
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            content=[TextContent(text="[Tool result missing — orphaned tool call]")],
                            is_error=True,
                            timestamp=int(time.time() * 1000),
                        )
                        result.insert(insert_pos, synthetic)
                        insert_pos += 1
        i += 1

    return result
