"""Harness custom messages — mirrors harness/messages.ts."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from pi_ai.types import ImageContent, Message, TextContent

from pi_agent.types import AgentMessage

COMPACTION_SUMMARY_PREFIX = """The conversation history before this point was compacted into the following summary:

<summary>
"""

COMPACTION_SUMMARY_SUFFIX = """
</summary>"""

BRANCH_SUMMARY_PREFIX = """The following is a summary of a branch that this conversation came back from:

<summary>
"""

BRANCH_SUMMARY_SUFFIX = """</summary>"""


class BashExecutionMessage(BaseModel):
    role: Literal["bashExecution"] = "bashExecution"
    command: str
    output: str
    exit_code: int | None = None
    cancelled: bool = False
    truncated: bool = False
    full_output_path: str | None = None
    timestamp: int
    exclude_from_context: bool | None = None


class CustomMessage(BaseModel):
    role: Literal["custom"] = "custom"
    custom_type: str
    content: str | list[TextContent | ImageContent]
    display: bool
    details: Any = None
    timestamp: int


class BranchSummaryMessage(BaseModel):
    role: Literal["branchSummary"] = "branchSummary"
    summary: str
    from_id: str
    timestamp: int


class CompactionSummaryMessage(BaseModel):
    role: Literal["compactionSummary"] = "compactionSummary"
    summary: str
    tokens_before: int
    timestamp: int


def _as_timestamp(timestamp: str | int) -> int:
    if isinstance(timestamp, int):
        return timestamp
    from datetime import datetime

    return int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)


def bash_execution_to_text(msg: BashExecutionMessage) -> str:
    text = f"Ran `{msg.command}`\n"
    if msg.output:
        text += f"```\n{msg.output}\n```"
    else:
        text += "(no output)"
    if msg.cancelled:
        text += "\n\n(command cancelled)"
    elif msg.exit_code is not None and msg.exit_code != 0:
        text += f"\n\nCommand exited with code {msg.exit_code}"
    if msg.truncated and msg.full_output_path:
        text += f"\n\n[Output truncated. Full output: {msg.full_output_path}]"
    return text


def create_branch_summary_message(summary: str, from_id: str, timestamp: str | int) -> BranchSummaryMessage:
    return BranchSummaryMessage(summary=summary, from_id=from_id, timestamp=_as_timestamp(timestamp))


def create_compaction_summary_message(
    summary: str,
    tokens_before: int,
    timestamp: str | int,
) -> CompactionSummaryMessage:
    return CompactionSummaryMessage(summary=summary, tokens_before=tokens_before, timestamp=_as_timestamp(timestamp))


def create_custom_message(
    custom_type: str,
    content: str | list[TextContent | ImageContent],
    display: bool,
    details: Any,
    timestamp: str | int,
) -> CustomMessage:
    return CustomMessage(
        custom_type=custom_type,
        content=content,
        display=display,
        details=details,
        timestamp=_as_timestamp(timestamp),
    )


def _role(message: Any) -> str | None:
    return getattr(message, "role", None) if not isinstance(message, dict) else message.get("role")


def convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    from pi_ai.types import UserMessage

    converted: list[Message] = []
    for message in messages:
        role = _role(message)
        if role == "bashExecution":
            msg = message if isinstance(message, BashExecutionMessage) else BashExecutionMessage.model_validate(message)
            if msg.exclude_from_context:
                continue
            converted.append(
                UserMessage(role="user", content=[{"type": "text", "text": bash_execution_to_text(msg)}], timestamp=msg.timestamp)
            )
        elif role == "custom":
            msg = message if isinstance(message, CustomMessage) else CustomMessage.model_validate(message)
            content = [{"type": "text", "text": msg.content}] if isinstance(msg.content, str) else msg.content
            converted.append(UserMessage(role="user", content=content, timestamp=msg.timestamp))
        elif role == "branchSummary":
            msg = message if isinstance(message, BranchSummaryMessage) else BranchSummaryMessage.model_validate(message)
            converted.append(
                UserMessage(
                    role="user",
                    content=[{"type": "text", "text": BRANCH_SUMMARY_PREFIX + msg.summary + BRANCH_SUMMARY_SUFFIX}],
                    timestamp=msg.timestamp,
                )
            )
        elif role == "compactionSummary":
            msg = (
                message
                if isinstance(message, CompactionSummaryMessage)
                else CompactionSummaryMessage.model_validate(message)
            )
            converted.append(
                UserMessage(
                    role="user",
                    content=[{"type": "text", "text": COMPACTION_SUMMARY_PREFIX + msg.summary + COMPACTION_SUMMARY_SUFFIX}],
                    timestamp=msg.timestamp,
                )
            )
        elif role in ("user", "assistant", "toolResult"):
            converted.append(message)  # type: ignore[arg-type]
    return converted
