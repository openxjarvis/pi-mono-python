"""Compaction helpers — mirrors harness/compaction/compaction.ts."""
from __future__ import annotations

import json
import math
from typing import Any, TypedDict

from pi_ai.models_runtime import Models
from pi_ai.stream import complete_simple
from pi_ai.types import AssistantMessage, Context, Model, SimpleStreamOptions, Usage, UserMessage
from pi_ai.utils.retry import RetryCallbacks, RetryPolicy, retry_assistant_call
from pi_ai.utils.text import content_text
from pi_ai.utils.uuid import uuidv7

from pi_agent.harness.compaction.utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)
from pi_agent.harness.messages import convert_to_llm, create_branch_summary_message, create_compaction_summary_message
from pi_agent.harness.session.context import build_session_context
from pi_agent.harness.session.types import CompactionEntry, Entry
from pi_agent.harness.types import CompactionError, Result, err, ok
from pi_agent.types import AgentMessage, ThinkingLevel


class CompactionDetails(TypedDict, total=False):
    read_files: list[str]
    modified_files: list[str]


class CompactResult(TypedDict, total=False):
    summary: str
    tokens_before: int
    usage: Usage
    retained_tail: list[AgentMessage]
    details: Any


class CompactionSettings(TypedDict):
    enabled: bool
    reserve_tokens: int
    keep_recent_tokens: int


DEFAULT_COMPACTION_SETTINGS: CompactionSettings = {
    "enabled": True,
    "reserve_tokens": 16384,
    "keep_recent_tokens": 20000,
}


class ContextUsageEstimate(TypedDict):
    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


class CutPointResult(TypedDict):
    first_kept_entry_index: int
    turn_start_index: int
    is_split_turn: bool


class CompactionPreparation(TypedDict, total=False):
    messages_to_summarize: list[AgentMessage]
    turn_prefix_messages: list[AgentMessage]
    retained_tail: list[AgentMessage]
    is_split_turn: bool
    tokens_before: int
    previous_summary: str
    file_ops: FileOperations
    settings: CompactionSettings


def _safe_json_stringify(value: Any) -> str:
    try:
        encoded = json.dumps(value)
        return "undefined" if encoded is None else encoded
    except Exception:
        return "[unserializable]"


def _role(message: Any) -> str | None:
    return getattr(message, "role", None) if not isinstance(message, dict) else message.get("role")


def extract_file_operations(messages: list[AgentMessage], entries: list[Entry], prev_compaction_index: int) -> FileOperations:
    file_ops = create_file_ops()
    if prev_compaction_index >= 0:
        prev = entries[prev_compaction_index]
        details = prev.get("details")
        if details:
            read_files = details.get("read_files") or details.get("readFiles")
            modified_files = details.get("modified_files") or details.get("modifiedFiles")
            if isinstance(read_files, list):
                for path in read_files:
                    file_ops["read"].add(path)
            if isinstance(modified_files, list):
                for path in modified_files:
                    file_ops["edited"].add(path)
    for msg in messages:
        extract_file_ops_from_message(msg, file_ops)
    return file_ops


def _get_message_from_entry(entry: Entry) -> AgentMessage | None:
    if entry.get("type") == "message":
        return entry["message"]
    if entry.get("type") == "branch_summary":
        return create_branch_summary_message(entry["summary"], entry.get("from_id") or entry.get("fromId"), entry["timestamp"])
    if entry.get("type") == "compaction":
        return create_compaction_summary_message(
            entry["summary"],
            entry.get("tokens_before", entry.get("tokensBefore", 0)),
            entry["timestamp"],
        )
    return None


def _get_message_from_entry_for_compaction(entry: Entry) -> AgentMessage | None:
    if entry.get("type") == "compaction":
        return None
    return _get_message_from_entry(entry)


async def complete_simple_with_retries(
    models: Models,
    model: Model,
    context: Context,
    options: SimpleStreamOptions,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> AssistantMessage:
    del models
    request_options = options.model_copy(update={"cache_retention": "none", "session_id": uuidv7()})

    async def produce() -> AssistantMessage:
        return await complete_simple(model, context, request_options)

    return await retry_assistant_call(produce, retry, getattr(request_options, "signal", None), callbacks)


def combine_usage(first: Usage, second: Usage) -> Usage:
    return Usage(
        input=first.input + second.input,
        output=first.output + second.output,
        cache_read=first.cache_read + second.cache_read,
        cache_write=first.cache_write + second.cache_write,
        cache_write_1h=(
            (first.cache_write_1h or 0) + (second.cache_write_1h or 0)
            if first.cache_write_1h is not None or second.cache_write_1h is not None
            else None
        ),
        reasoning=(
            (first.reasoning or 0) + (second.reasoning or 0)
            if first.reasoning is not None or second.reasoning is not None
            else None
        ),
        total_tokens=first.total_tokens + second.total_tokens,
        cost=type(first.cost)(
            input=first.cost.input + second.cost.input,
            output=first.cost.output + second.cost.output,
            cache_read=first.cost.cache_read + second.cost.cache_read,
            cache_write=first.cost.cache_write + second.cost.cache_write,
            total=first.cost.total + second.cost.total,
        ),
    )


def calculate_context_tokens(usage: Usage) -> int:
    return usage.total_tokens or (usage.input + usage.output + usage.cache_read + usage.cache_write)


def _get_assistant_usage(msg: AgentMessage) -> Usage | None:
    if _role(msg) != "assistant":
        return None
    stop = getattr(msg, "stop_reason", None) if not isinstance(msg, dict) else msg.get("stop_reason", msg.get("stopReason"))
    usage = getattr(msg, "usage", None) if not isinstance(msg, dict) else msg.get("usage")
    if stop in ("aborted", "error") or not usage:
        return None
    if isinstance(usage, dict):
        usage = Usage.model_validate(usage)
    if calculate_context_tokens(usage) > 0:
        return usage
    return None


def get_last_assistant_usage(entries: list[Entry]) -> Usage | None:
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if entry.get("type") == "message":
            usage = _get_assistant_usage(entry["message"])
            if usage:
                return usage
    return None


def _get_last_assistant_usage_info(messages: list[AgentMessage]) -> dict[str, Any] | None:
    for i in range(len(messages) - 1, -1, -1):
        usage = _get_assistant_usage(messages[i])
        if usage:
            return {"usage": usage, "index": i}
    return None


def estimate_context_tokens(messages: list[AgentMessage]) -> ContextUsageEstimate:
    usage_info = _get_last_assistant_usage_info(messages)
    if not usage_info:
        estimated = sum(estimate_tokens(message) for message in messages)
        return ContextUsageEstimate(tokens=estimated, usage_tokens=0, trailing_tokens=estimated, last_usage_index=None)
    usage_tokens = calculate_context_tokens(usage_info["usage"])
    trailing = sum(estimate_tokens(messages[i]) for i in range(usage_info["index"] + 1, len(messages)))
    return ContextUsageEstimate(
        tokens=usage_tokens + trailing,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing,
        last_usage_index=usage_info["index"],
    )


def should_compact(context_tokens: int, context_window: int, settings: CompactionSettings) -> bool:
    if not settings["enabled"]:
        return False
    return context_tokens > context_window - settings["reserve_tokens"]


ESTIMATED_IMAGE_CHARS = 4800


def _estimate_text_and_image_content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    chars = 0
    for block in content or []:
        block_type = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
        if block_type == "text":
            text = getattr(block, "text", "") if not isinstance(block, dict) else block.get("text") or ""
            chars += len(text)
        elif block_type == "image":
            chars += ESTIMATED_IMAGE_CHARS
    return chars


def estimate_tokens(message: AgentMessage) -> int:
    role = _role(message)
    if role == "user":
        content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
        return math.ceil(_estimate_text_and_image_content_chars(content) / 4)
    if role == "assistant":
        chars = 0
        content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
        for block in content or []:
            block_type = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
            if block_type == "text":
                chars += len(getattr(block, "text", "") if not isinstance(block, dict) else block.get("text") or "")
            elif block_type == "thinking":
                chars += len(getattr(block, "thinking", "") if not isinstance(block, dict) else block.get("thinking") or "")
            elif block_type == "toolCall":
                name = getattr(block, "name", "") if not isinstance(block, dict) else block.get("name") or ""
                args = getattr(block, "arguments", {}) if not isinstance(block, dict) else block.get("arguments")
                chars += len(name) + len(_safe_json_stringify(args))
        return math.ceil(chars / 4)
    if role in ("custom", "toolResult"):
        content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
        return math.ceil(_estimate_text_and_image_content_chars(content) / 4)
    if role == "bashExecution":
        command = getattr(message, "command", "") if not isinstance(message, dict) else message.get("command") or ""
        output = getattr(message, "output", "") if not isinstance(message, dict) else message.get("output") or ""
        return math.ceil((len(command) + len(output)) / 4)
    if role in ("branchSummary", "compactionSummary"):
        summary = getattr(message, "summary", "") if not isinstance(message, dict) else message.get("summary") or ""
        return math.ceil(len(summary) / 4)
    return 0


def _find_valid_cut_points(entries: list[Entry], start_index: int, end_index: int) -> list[int]:
    cut_points: list[int] = []
    for i in range(start_index, end_index):
        entry = entries[i]
        if entry.get("type") == "message":
            if _role(entry.get("message")) in (
                "bashExecution",
                "custom",
                "branchSummary",
                "compactionSummary",
                "user",
                "assistant",
            ):
                cut_points.append(i)
        if entry.get("type") == "branch_summary":
            cut_points.append(i)
    return cut_points


def find_turn_start_index(entries: list[Entry], entry_index: int, start_index: int) -> int:
    for i in range(entry_index, start_index - 1, -1):
        entry = entries[i]
        if entry.get("type") == "branch_summary":
            return i
        if entry.get("type") == "message" and _role(entry.get("message")) in ("user", "bashExecution"):
            return i
    return -1


def find_cut_point(entries: list[Entry], start_index: int, end_index: int, keep_recent_tokens: int) -> CutPointResult:
    cut_points = _find_valid_cut_points(entries, start_index, end_index)
    if not cut_points:
        return CutPointResult(first_kept_entry_index=start_index, turn_start_index=-1, is_split_turn=False)
    accumulated = 0
    cut_index = cut_points[0]
    for i in range(end_index - 1, start_index - 1, -1):
        entry = entries[i]
        if entry.get("type") != "message":
            continue
        accumulated += estimate_tokens(entry["message"])
        if accumulated >= keep_recent_tokens:
            for point in cut_points:
                if point >= i:
                    cut_index = point
                    break
            break
    while cut_index > start_index:
        prev = entries[cut_index - 1]
        if prev.get("type") in ("compaction", "message"):
            break
        cut_index -= 1
    cut_entry = entries[cut_index]
    is_user = cut_entry.get("type") == "message" and _role(cut_entry.get("message")) == "user"
    turn_start = -1 if is_user else find_turn_start_index(entries, cut_index, start_index)
    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start,
        is_split_turn=not is_user and turn_start != -1,
    )


SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, "
    "then produce a structured summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary."
)

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""


async def generate_summary(
    current_messages: list[AgentMessage],
    models: Models,
    model: Model,
    reserve_tokens: int,
    abort: Any = None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    thinking_level: ThinkingLevel | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> Result:
    result = await generate_summary_with_usage(
        current_messages,
        models,
        model,
        reserve_tokens,
        abort,
        custom_instructions,
        previous_summary,
        thinking_level,
        retry,
        callbacks,
    )
    return ok(result["value"]["text"]) if result["ok"] else err(result["error"])


async def generate_summary_with_usage(
    current_messages: list[AgentMessage],
    models: Models,
    model: Model,
    reserve_tokens: int,
    abort: Any = None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    thinking_level: ThinkingLevel | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> Result:
    max_tokens = min(math.floor(0.8 * reserve_tokens), model.max_tokens if model.max_tokens > 0 else math.inf)
    base_prompt = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"
    conversation_text = serialize_conversation(convert_to_llm(current_messages))
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt
    import time

    summarization_messages = [
        UserMessage(role="user", content=[{"type": "text", "text": prompt_text}], timestamp=int(time.time() * 1000))
    ]
    completion_options = SimpleStreamOptions(max_tokens=int(max_tokens), signal=abort)
    if model.reasoning and thinking_level and thinking_level != "off":
        completion_options = completion_options.model_copy(update={"reasoning": thinking_level})
    response = await complete_simple_with_retries(
        models,
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        completion_options,
        retry,
        callbacks,
    )
    if response.stop_reason == "aborted":
        return err(CompactionError("aborted", response.error_message or "Summarization aborted"))
    if response.stop_reason == "error":
        return err(CompactionError("summarization_failed", f"Summarization failed: {response.error_message or 'Unknown error'}"))
    return ok({"text": content_text(response.content), "usage": response.usage})


def prepare_compaction(path_entries: list[Entry], settings: CompactionSettings) -> Result:
    if not path_entries or path_entries[-1].get("type") == "compaction":
        return ok(None)
    prev_compaction_index = -1
    for i in range(len(path_entries) - 1, -1, -1):
        if path_entries[i].get("type") == "compaction":
            prev_compaction_index = i
            break
    previous_summary = None
    compactable = path_entries
    if prev_compaction_index >= 0:
        prev = path_entries[prev_compaction_index]
        previous_summary = prev.get("summary")
        retained = prev.get("retained_tail") or prev.get("retainedTail") or []
        virtual = []
        for index, message in enumerate(retained):
            timestamp = getattr(message, "timestamp", None) if not isinstance(message, dict) else message.get("timestamp")
            virtual.append(
                {
                    "type": "message",
                    "id": f"{prev['id']}:retained:{index}",
                    "parent_id": prev["id"] if index == 0 else f"{prev['id']}:retained:{index - 1}",
                    "seq": prev["seq"],
                    "timestamp": timestamp,
                    "message": message,
                }
            )
        compactable = [*virtual, *path_entries[prev_compaction_index + 1 :]]
    tokens_before = estimate_context_tokens(build_session_context(path_entries)["messages"])["tokens"]
    cut_point = find_cut_point(compactable, 0, len(compactable), settings["keep_recent_tokens"])
    history_end = cut_point["turn_start_index"] if cut_point["is_split_turn"] else cut_point["first_kept_entry_index"]
    messages_to_summarize = [
        msg for i in range(history_end) if (msg := _get_message_from_entry_for_compaction(compactable[i]))
    ]
    turn_prefix: list[AgentMessage] = []
    if cut_point["is_split_turn"]:
        for i in range(cut_point["turn_start_index"], cut_point["first_kept_entry_index"]):
            msg = _get_message_from_entry_for_compaction(compactable[i])
            if msg:
                turn_prefix.append(msg)
    retained_tail = [
        msg
        for i in range(cut_point["first_kept_entry_index"], len(compactable))
        if (msg := _get_message_from_entry_for_compaction(compactable[i]))
    ]
    file_ops = extract_file_operations(messages_to_summarize, path_entries, prev_compaction_index)
    if cut_point["is_split_turn"]:
        for msg in turn_prefix:
            extract_file_ops_from_message(msg, file_ops)
    prep: CompactionPreparation = {
        "messages_to_summarize": messages_to_summarize,
        "turn_prefix_messages": turn_prefix,
        "retained_tail": retained_tail,
        "is_split_turn": cut_point["is_split_turn"],
        "tokens_before": tokens_before,
        "file_ops": file_ops,
        "settings": settings,
    }
    if previous_summary is not None:
        prep["previous_summary"] = previous_summary
    return ok(prep)


async def compact(
    preparation: CompactionPreparation,
    models: Models,
    model: Model,
    custom_instructions: str | None = None,
    abort: Any = None,
    thinking_level: ThinkingLevel | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> Result:
    settings = preparation["settings"]
    if preparation["is_split_turn"] and preparation["turn_prefix_messages"]:
        history_text = "No prior history."
        history_usage = None
        if preparation["messages_to_summarize"]:
            history_result = await generate_summary_with_usage(
                preparation["messages_to_summarize"],
                models,
                model,
                settings["reserve_tokens"],
                abort,
                custom_instructions,
                preparation.get("previous_summary"),
                thinking_level,
                retry,
                callbacks,
            )
            if not history_result["ok"]:
                return err(history_result["error"])
            history_text = history_result["value"]["text"]
            history_usage = history_result["value"]["usage"]
        turn_prefix_result = await _generate_turn_prefix_summary(
            preparation["turn_prefix_messages"],
            models,
            model,
            settings["reserve_tokens"],
            abort,
            thinking_level,
            retry,
            callbacks,
        )
        if not turn_prefix_result["ok"]:
            return err(turn_prefix_result["error"])
        summary = f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_prefix_result['value']['text']}"
        summary_usage = (
            combine_usage(history_usage, turn_prefix_result["value"]["usage"])
            if history_usage
            else turn_prefix_result["value"]["usage"]
        )
    else:
        summary_result = await generate_summary_with_usage(
            preparation["messages_to_summarize"],
            models,
            model,
            settings["reserve_tokens"],
            abort,
            custom_instructions,
            preparation.get("previous_summary"),
            thinking_level,
            retry,
            callbacks,
        )
        if not summary_result["ok"]:
            return err(summary_result["error"])
        summary = summary_result["value"]["text"]
        summary_usage = summary_result["value"]["usage"]
    lists = compute_file_lists(preparation["file_ops"])
    summary += format_file_operations(lists["read_files"], lists["modified_files"])
    return ok(
        CompactResult(
            summary=summary,
            tokens_before=preparation["tokens_before"],
            usage=summary_usage,
            retained_tail=preparation["retained_tail"],
            details={"read_files": lists["read_files"], "modified_files": lists["modified_files"]},
        )
    )


async def _generate_turn_prefix_summary(
    messages: list[AgentMessage],
    models: Models,
    model: Model,
    reserve_tokens: int,
    abort: Any,
    thinking_level: ThinkingLevel | None,
    retry: RetryPolicy | None,
    callbacks: RetryCallbacks | None,
) -> Result:
    import time

    max_tokens = min(math.floor(0.5 * reserve_tokens), model.max_tokens if model.max_tokens > 0 else math.inf)
    conversation_text = serialize_conversation(convert_to_llm(messages))
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"
    summarization_messages = [
        UserMessage(role="user", content=[{"type": "text", "text": prompt_text}], timestamp=int(time.time() * 1000))
    ]
    completion_options = SimpleStreamOptions(max_tokens=int(max_tokens), signal=abort)
    if model.reasoning and thinking_level and thinking_level != "off":
        completion_options = completion_options.model_copy(update={"reasoning": thinking_level})
    response = await complete_simple_with_retries(
        models,
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        completion_options,
        retry,
        callbacks,
    )
    if response.stop_reason == "aborted":
        return err(CompactionError("aborted", response.error_message or "Turn prefix summarization aborted"))
    if response.stop_reason == "error":
        return err(
            CompactionError(
                "summarization_failed",
                f"Turn prefix summarization failed: {response.error_message or 'Unknown error'}",
            )
        )
    return ok({"text": content_text(response.content), "usage": response.usage})
