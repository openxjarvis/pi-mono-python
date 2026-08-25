"""
Context compaction — mirrors packages/coding-agent/src/core/compaction/compaction.ts

Handles automatic context compression when the context window fills up.
Pure functions for compaction logic; session manager handles I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pi_ai.types import AssistantMessage, Message, TextContent, UserMessage

# ─── Compaction settings ──────────────────────────────────────────────────────

DEFAULT_COMPACTION_SETTINGS = {
    "enabled": True,
    "reserveTokens": 16384,
    "keepRecentTokens": 20000,
}

# ─── Summarization prompts ────────────────────────────────────────────────────

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a precise technical assistant that creates structured summaries of AI coding sessions. "
    "Focus on technical facts: files changed, decisions made, current state, next steps. "
    "Be concise and accurate."
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


# ─── Token estimation ─────────────────────────────────────────────────────────

def estimate_tokens(message: dict[str, Any] | Any) -> int:
    """
    Estimate token count for a message using chars/4 heuristic.
    Mirrors estimateTokens() in TypeScript.
    """
    if isinstance(message, dict):
        role = message.get("role", "")
        content = message.get("content", "")
    else:
        role = getattr(message, "role", "")
        content = getattr(message, "content", "")

    chars = 0

    if role in ("user", "bashExecution", "branchSummary", "compactionSummary", "custom"):
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        chars += len(block.get("text", ""))
                    elif block.get("type") == "image":
                        chars += 4800
                elif hasattr(block, "type"):
                    if block.type == "text":
                        chars += len(getattr(block, "text", ""))
                    elif block.type == "image":
                        chars += 4800
        # summary field for compaction/branch messages
        if hasattr(message, "summary"):
            chars = len(message.summary)
    elif role == "assistant":
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        chars += len(block.get("text", ""))
                    elif btype == "thinking":
                        chars += len(block.get("thinking", ""))
                    elif btype == "tool_call" or btype == "toolCall":
                        name = block.get("name", "")
                        args = block.get("arguments") or block.get("input", {})
                        import json
                        chars += len(name) + len(json.dumps(args))
                elif hasattr(block, "type"):
                    btype = block.type
                    if btype == "text":
                        chars += len(getattr(block, "text", ""))
                    elif btype == "thinking":
                        chars += len(getattr(block, "thinking", ""))
    elif role == "toolResult":
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        chars += len(block.get("text", ""))
                    elif block.get("type") == "image":
                        chars += 4800

    return max(1, (chars + 3) // 4)


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    """Normalize Usage objects / dicts to the camelCase fields used by TS."""
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {
            "totalTokens": usage.get("totalTokens") or usage.get("total_tokens") or 0,
            "input": usage.get("input", 0) or 0,
            "output": usage.get("output", 0) or 0,
            "cacheRead": usage.get("cacheRead") or usage.get("cache_read") or 0,
            "cacheWrite": usage.get("cacheWrite") or usage.get("cache_write") or 0,
        }
    if hasattr(usage, "model_dump"):
        return _usage_to_dict(usage.model_dump())
    return {
        "totalTokens": getattr(usage, "total_tokens", 0) or getattr(usage, "totalTokens", 0) or 0,
        "input": getattr(usage, "input", 0) or 0,
        "output": getattr(usage, "output", 0) or 0,
        "cacheRead": getattr(usage, "cache_read", 0) or getattr(usage, "cacheRead", 0) or 0,
        "cacheWrite": getattr(usage, "cache_write", 0) or getattr(usage, "cacheWrite", 0) or 0,
    }


def calculate_context_tokens(usage: dict[str, Any] | Any) -> int:
    """Calculate total context tokens from usage.

    Uses the native totalTokens field when available, falls back to components.
    Mirrors calculateContextTokens() in TypeScript.
    """
    d = _usage_to_dict(usage)
    if d.get("totalTokens"):
        return int(d["totalTokens"])
    return int(d.get("input", 0) + d.get("output", 0) + d.get("cacheRead", 0) + d.get("cacheWrite", 0))


def _message_role(msg: Any) -> str:
    return msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")


def _message_stop_reason(msg: Any) -> str:
    if isinstance(msg, dict):
        return msg.get("stop_reason") or msg.get("stopReason") or ""
    return getattr(msg, "stop_reason", "") or getattr(msg, "stopReason", "") or ""


def get_assistant_usage(msg: Any) -> dict[str, Any] | None:
    """Return usage from an assistant message if it is valid.

    Skips aborted, error, and all-zero usage messages.
    Mirrors getAssistantUsage() in TypeScript.
    """
    if _message_role(msg) != "assistant":
        return None
    if _message_stop_reason(msg) in ("aborted", "error"):
        return None
    usage = msg.get("usage") if isinstance(msg, dict) else getattr(msg, "usage", None)
    if not usage:
        return None
    usage_dict = _usage_to_dict(usage)
    if calculate_context_tokens(usage_dict) <= 0:
        return None
    return usage_dict


def get_last_assistant_usage(entries: list[Any]) -> dict[str, Any] | None:
    """Find the last valid assistant message usage from session entries."""
    for entry in reversed(entries):
        raw = _as_entry_dict(entry)
        if raw.get("type") == "message":
            usage = get_assistant_usage(raw.get("message"))
            if usage:
                return usage
    return None


def estimate_context_tokens(messages: list[Any]) -> dict[str, Any]:
    """
    Estimate context tokens from messages, using the last assistant usage when available.
    Mirrors estimateContextTokens() in TypeScript.

    Without provider usage (error / all-zero), tokens is the pure message-size estimate.
    Returns {"tokens", "usageTokens", "trailingTokens", "lastUsageIndex"}.
    """
    last_usage: dict[str, Any] | None = None
    last_usage_idx: int | None = None

    for i in reversed(range(len(messages))):
        usage = get_assistant_usage(messages[i])
        if usage:
            last_usage = usage
            last_usage_idx = i
            break

    if last_usage is None:
        total = sum(estimate_tokens(m) for m in messages)
        return {
            "tokens": total,
            "usageTokens": 0,
            "trailingTokens": total,
            "lastUsageIndex": None,
        }

    usage_tokens = calculate_context_tokens(last_usage)
    trailing = sum(estimate_tokens(messages[i]) for i in range(last_usage_idx + 1, len(messages)))

    return {
        "tokens": usage_tokens + trailing,
        "usageTokens": usage_tokens,
        "trailingTokens": trailing,
        "lastUsageIndex": last_usage_idx,
    }


# ─── Cut point detection ───────────────────────────────────────────────────────

def _is_valid_cut_entry(entry: dict[str, Any]) -> bool:
    """Return True if this entry is a valid cut point (not a tool result)."""
    etype = entry.get("type", "")
    if etype in ("branch_summary", "custom_message"):
        return True
    if etype == "message":
        role = entry.get("message", {}).get("role", "")
        return role in ("user", "assistant", "bashExecution", "custom", "branchSummary", "compactionSummary")
    return False


def find_valid_cut_points(entries: list[dict[str, Any]], start: int, end: int) -> list[int]:
    """
    Find valid cut points: indices of user/assistant/custom messages.
    Never cut at tool results. Mirrors findValidCutPoints() in TypeScript.
    """
    return [i for i in range(start, end) if _is_valid_cut_entry(entries[i])]


def find_turn_start_index(entries: list[dict[str, Any]], entry_idx: int, start: int) -> int:
    """
    Find the user message (or bashExecution) that starts the turn containing entry_idx.
    Returns -1 if none found. Mirrors findTurnStartIndex() in TypeScript.
    """
    for i in range(entry_idx, start - 1, -1):
        entry = entries[i]
        etype = entry.get("type", "")
        if etype in ("branch_summary", "custom_message"):
            return i
        if etype == "message":
            role = entry.get("message", {}).get("role", "")
            if role in ("user", "bashExecution"):
                return i
    return -1


def find_cut_point(
    entries: list[dict[str, Any]],
    start_index: int,
    end_index: int,
    keep_recent_tokens: int,
) -> dict[str, Any]:
    """
    Find the optimal cut point in session entries.
    Mirrors findCutPoint() in TypeScript.

    Returns {firstKeptEntryIndex, turnStartIndex, isSplitTurn}.
    """
    cut_points = find_valid_cut_points(entries, start_index, end_index)

    if not cut_points:
        return {"firstKeptEntryIndex": start_index, "turnStartIndex": -1, "isSplitTurn": False}

    accumulated = 0
    cut_index = cut_points[0]

    for i in range(end_index - 1, start_index - 1, -1):
        entry = entries[i]
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        msg_tokens = estimate_tokens(msg)
        accumulated += msg_tokens

        if accumulated >= keep_recent_tokens:
            # Find closest cut point at or after this entry
            for cp in cut_points:
                if cp >= i:
                    cut_index = cp
                    break
            break

    # Scan backwards to include non-message entries before the cut
    while cut_index > start_index:
        prev = entries[cut_index - 1]
        if prev.get("type") in ("compaction", "message"):
            break
        cut_index -= 1

    # Determine if this is a split turn
    cut_entry = entries[cut_index]
    is_user = (
        cut_entry.get("type") == "message"
        and cut_entry.get("message", {}).get("role") == "user"
    )
    turn_start = -1 if is_user else find_turn_start_index(entries, cut_index, start_index)

    return {
        "firstKeptEntryIndex": cut_index,
        "turnStartIndex": turn_start,
        "isSplitTurn": not is_user and turn_start != -1,
    }


def _as_entry_dict(entry: Any) -> dict[str, Any]:
    """Normalize SessionEntry objects or raw dicts to the on-disk entry shape."""
    if isinstance(entry, dict):
        return entry
    data = getattr(entry, "data", None)
    if isinstance(data, dict) and data:
        return data
    return {
        "id": getattr(entry, "id", ""),
        "type": getattr(entry, "type", ""),
        "timestamp": getattr(entry, "timestamp", 0),
        "parentId": getattr(entry, "parent_id", None),
    }


def _message_from_entry(entry: dict[str, Any]) -> Any | None:
    """Extract an AgentMessage from a session entry, or None if it is not in LLM context."""
    etype = entry.get("type", "")
    if etype == "compaction":
        return None
    if etype == "message":
        return entry.get("message")
    if etype == "custom_message":
        return {
            "role": "custom",
            "content": entry.get("content", ""),
            "customType": entry.get("customType", ""),
            "timestamp": entry.get("timestamp", 0),
        }
    if etype == "branch_summary":
        return {
            "role": "branchSummary",
            "summary": entry.get("summary", ""),
            "timestamp": entry.get("timestamp", 0),
        }
    return None


def prepare_compaction(
    path_entries: list[Any],
    settings: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Pre-calculate compaction cut points and messages to summarize.

    Mirrors prepareCompaction() in TypeScript.
    """
    s = {**DEFAULT_COMPACTION_SETTINGS, **(settings or {})}
    entries = [_as_entry_dict(e) for e in path_entries]
    if not entries:
        return None
    if entries[-1].get("type") == "compaction":
        return None

    prev_compaction_index = -1
    for i in range(len(entries) - 1, -1, -1):
        if entries[i].get("type") == "compaction":
            prev_compaction_index = i
            break

    previous_summary: str | None = None
    boundary_start = 0
    if prev_compaction_index >= 0:
        prev = entries[prev_compaction_index]
        previous_summary = prev.get("summary")
        first_kept_id = prev.get("firstKeptEntryId")
        first_kept_idx = next((i for i, e in enumerate(entries) if e.get("id") == first_kept_id), -1)
        boundary_start = first_kept_idx if first_kept_idx >= 0 else prev_compaction_index + 1

    cut = find_cut_point(entries, boundary_start, len(entries), s.get("keepRecentTokens", 20000))
    first_kept = entries[cut["firstKeptEntryIndex"]] if cut["firstKeptEntryIndex"] < len(entries) else None
    if not first_kept or not first_kept.get("id"):
        return None

    history_end = cut["turnStartIndex"] if cut["isSplitTurn"] else cut["firstKeptEntryIndex"]
    messages_to_summarize = [
        msg
        for i in range(boundary_start, max(history_end, 0))
        if (msg := _message_from_entry(entries[i])) is not None
    ]
    turn_prefix_messages: list[Any] = []
    if cut["isSplitTurn"]:
        turn_prefix_messages = [
            msg
            for i in range(cut["turnStartIndex"], cut["firstKeptEntryIndex"])
            if (msg := _message_from_entry(entries[i])) is not None
        ]

    if not messages_to_summarize and not turn_prefix_messages:
        return None

    tokens_before = estimate_context_tokens(
        [m for e in entries if (m := _message_from_entry(e)) is not None]
    )["tokens"]

    from .utils import create_file_ops, extract_file_ops_from_message

    file_ops = create_file_ops()
    if prev_compaction_index >= 0:
        details = entries[prev_compaction_index].get("details") or {}
        if isinstance(details, dict):
            for f in details.get("readFiles") or []:
                file_ops.read.add(f)
            for f in details.get("modifiedFiles") or []:
                file_ops.edited.add(f)
    for msg in messages_to_summarize + turn_prefix_messages:
        extract_file_ops_from_message(msg, file_ops)

    return {
        "firstKeptEntryId": first_kept["id"],
        "messagesToSummarize": messages_to_summarize,
        "turnPrefixMessages": turn_prefix_messages,
        "isSplitTurn": cut["isSplitTurn"],
        "tokensBefore": tokens_before,
        "previousSummary": previous_summary,
        "fileOps": file_ops,
        "settings": s,
    }


async def compact(
    preparation: dict[str, Any],
    model: Any,
    api_key: str | None = None,
    custom_instructions: str | None = None,
) -> dict[str, Any]:
    """Generate compaction summaries from a prepare_compaction() result.

    Mirrors compact() in TypeScript (without retry/streamFn plumbing).
    """
    from .utils import compute_file_lists, format_file_operations

    messages_to_summarize = preparation["messagesToSummarize"]
    turn_prefix_messages = preparation["turnPrefixMessages"]
    is_split = preparation["isSplitTurn"]
    previous_summary = preparation.get("previousSummary")
    settings = preparation.get("settings") or DEFAULT_COMPACTION_SETTINGS
    reserve = settings.get("reserveTokens", 16384)
    file_ops = preparation.get("fileOps")

    if is_split and turn_prefix_messages:
        history_text = "No prior history."
        if messages_to_summarize:
            history_text = await generate_summary(
                current_messages=messages_to_summarize,
                model=model,
                reserve_tokens=reserve,
                api_key=api_key,
                custom_instructions=custom_instructions,
                previous_summary=previous_summary,
            )
        prefix_text = await generate_summary(
            current_messages=turn_prefix_messages,
            model=model,
            reserve_tokens=max(1, reserve // 2),
            api_key=api_key,
        )
        summary = f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix_text}"
    else:
        summary = await generate_summary(
            current_messages=messages_to_summarize,
            model=model,
            reserve_tokens=reserve,
            api_key=api_key,
            custom_instructions=custom_instructions,
            previous_summary=previous_summary,
        )

    if file_ops is not None:
        read_files, modified_files = compute_file_lists(file_ops)
        summary += format_file_operations(read_files, modified_files)
    else:
        read_files, modified_files = [], []

    return {
        "summary": summary,
        "firstKeptEntryId": preparation["firstKeptEntryId"],
        "tokensBefore": preparation["tokensBefore"],
        "details": {"readFiles": read_files, "modifiedFiles": modified_files},
    }


# ─── Legacy API ───────────────────────────────────────────────────────────────

def should_compact(
    messages: list[Any],
    context_window: int,
    threshold: float = 0.8,
    settings: dict[str, Any] | None = None,
) -> bool:
    """
    Determine if context should be compacted.
    Mirrors shouldCompact() in TypeScript.
    """
    s = settings or DEFAULT_COMPACTION_SETTINGS
    if not s.get("enabled", True):
        return False
    reserve = s.get("reserveTokens", 16384)
    estimate = estimate_context_tokens(messages)
    return estimate["tokens"] > (context_window - reserve)


def _estimate_tokens_legacy(messages: list[Message]) -> int:
    """Legacy token estimator for old API compatibility."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg)
    return total


# ─── Summarization ────────────────────────────────────────────────────────────

def get_summarization_failure(response: Any, label: str) -> str | None:
    """Return an error message when a summarization response cannot safely be persisted.

    A length stop contains partial text and must not become a session checkpoint.
    Mirrors getSummarizationFailure() in TypeScript.
    """
    stop_reason = getattr(response, "stop_reason", None) or (
        response.get("stop_reason") or response.get("stopReason") if isinstance(response, dict) else None
    )
    if stop_reason == "error":
        error_message = getattr(response, "error_message", None) or (
            response.get("error_message") or response.get("errorMessage") if isinstance(response, dict) else None
        )
        return f"{label} failed: {error_message or 'Unknown error'}"
    if stop_reason == "length":
        return f"{label} failed: generation hit the token cap and the summary is incomplete"
    return None




async def generate_summary(
    current_messages: list[Any],
    model: Any,
    reserve_tokens: int,
    api_key: str | None = None,
    signal: Any = None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
) -> str:
    """
    Generate a summary of the conversation using the LLM.
    Mirrors generateSummary() in TypeScript.
    """
    from pi_ai import complete_simple
    from pi_ai.types import Context, SimpleStreamOptions

    max_tokens = int(0.8 * reserve_tokens)

    base_prompt = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    conv_text = _serialize_conversation(current_messages)
    prompt_text = f"<conversation>\n{conv_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    # Only pass reasoning="high" for reasoning models; non-reasoning models reject it
    reasoning_kwarg: dict[str, Any] = {"reasoning": "high"} if getattr(model, "reasoning", False) else {}
    opts = SimpleStreamOptions(max_tokens=max_tokens, **reasoning_kwarg)
    if signal is not None:
        opts = SimpleStreamOptions(max_tokens=max_tokens, signal=signal, **reasoning_kwarg)
    if api_key:
        opts = SimpleStreamOptions(max_tokens=max_tokens, api_key=api_key, **reasoning_kwarg)

    try:
        ctx = Context(
            system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
            messages=[UserMessage(
                role="user",
                content=[{"type": "text", "text": prompt_text}],
                timestamp=0,
            )],
        )
        response = await complete_simple(model, ctx, opts)
        failure = get_summarization_failure(response, "Summarization")
        if failure:
            raise RuntimeError(failure)

        return " ".join(
            b.text for b in response.content if isinstance(b, TextContent)
        )
    except Exception:
        raise


def _serialize_conversation(messages: list[Any]) -> str:
    """Serialize conversation messages to text."""
    parts = []
    for msg in messages:
        role = msg.get("role", "?") if isinstance(msg, dict) else getattr(msg, "role", "?")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    texts.append(block.text)
            text = " ".join(texts)
        else:
            text = str(content)

        parts.append(f"{role.upper()}: {text}")
    return "\n\n".join(parts)


# ─── Main compaction (compact_context) — legacy + new API ────────────────────

async def compact_context(
    messages: list[Message],
    system_prompt: str,
    stream_fn: Any,
    model: Any,
    settings: dict[str, Any] | None = None,
    previous_summary: str | None = None,
) -> tuple[list[Message], str]:
    """
    Compact the context by summarizing old messages.
    Mirrors compactContext() — returns (new_messages, summary).

    Now uses keepRecentTokens from settings instead of fixed last-4-messages.
    """
    s = settings or DEFAULT_COMPACTION_SETTINGS
    keep_recent_tokens = s.get("keepRecentTokens", 20000)
    reserve_tokens = s.get("reserveTokens", 16384)

    if len(messages) < 4:
        return messages, ""

    # Build simple entry list for cut point detection
    entries = [{"type": "message", "message": m, "id": str(i)} for i, m in enumerate(messages)]
    cut = find_cut_point(entries, 0, len(entries), keep_recent_tokens)
    cut_idx = cut["firstKeptEntryIndex"]

    to_summarize = messages[:cut_idx]
    to_keep = messages[cut_idx:]

    if not to_summarize:
        return messages, ""

    summary = await generate_summary(
        current_messages=to_summarize,
        model=model,
        reserve_tokens=reserve_tokens,
        previous_summary=previous_summary,
    )

    compact_msg = UserMessage(
        role="user",
        content=[{"type": "text", "text": f"[Previous conversation summary]\n{summary}"}],
        timestamp=0,
    )
    return [compact_msg] + list(to_keep), summary


def _build_summary_text(messages: list[Message]) -> str:
    """Build text representation of messages for summarization."""
    return _serialize_conversation(messages)
