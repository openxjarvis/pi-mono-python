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


def calculate_context_tokens(usage: dict[str, Any]) -> int:
    """Calculate total context tokens from usage dict."""
    if usage.get("totalTokens"):
        return usage["totalTokens"]
    return (
        usage.get("input", 0)
        + usage.get("output", 0)
        + usage.get("cacheRead", 0)
        + usage.get("cacheWrite", 0)
    )


def estimate_context_tokens(messages: list[Any]) -> dict[str, Any]:
    """
    Estimate context tokens from messages, using the last assistant usage when available.
    Mirrors estimateContextTokens() in TypeScript.

    Returns {"tokens", "usageTokens", "trailingTokens", "lastUsageIndex"}.
    """
    last_usage: dict[str, Any] | None = None
    last_usage_idx: int | None = None

    for i in reversed(range(len(messages))):
        msg = messages[i]
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        stop_reason = (
            msg.get("stop_reason") or msg.get("stopReason", "")
            if isinstance(msg, dict)
            else getattr(msg, "stop_reason", "") or getattr(msg, "stopReason", "")
        )
        usage = msg.get("usage") if isinstance(msg, dict) else getattr(msg, "usage", None)

        if role == "assistant" and usage and stop_reason not in ("aborted", "error"):
            last_usage = usage if isinstance(usage, dict) else getattr(usage, "__dict__", {})
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

    opts = SimpleStreamOptions(max_tokens=max_tokens)
    if signal is not None:
        opts = SimpleStreamOptions(max_tokens=max_tokens, signal=signal)
    if api_key:
        opts = SimpleStreamOptions(max_tokens=max_tokens, api_key=api_key)

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
        if getattr(response, "stop_reason", None) == "error":
            raise RuntimeError(f"Summarization failed: {getattr(response, 'error_message', 'Unknown error')}")

        return " ".join(
            b.text for b in response.content if isinstance(b, TextContent)
        )
    except Exception:
        return "[Summary generation failed]"


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
