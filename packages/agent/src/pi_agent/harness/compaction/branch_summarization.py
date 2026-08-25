"""Branch summarization — mirrors harness/compaction/branch-summarization.ts."""
from __future__ import annotations

import time
from typing import Any, TypedDict

from pi_ai.models_runtime import Models
from pi_ai.types import Context, Model, SimpleStreamOptions, Usage, UserMessage
from pi_ai.utils.retry import RetryCallbacks, RetryPolicy
from pi_ai.utils.text import content_text

from pi_agent.harness.compaction.compaction import SUMMARIZATION_SYSTEM_PROMPT, complete_simple_with_retries, estimate_tokens
from pi_agent.harness.compaction.utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)
from pi_agent.harness.messages import convert_to_llm, create_branch_summary_message, create_compaction_summary_message
from pi_agent.harness.session.session import Session
from pi_agent.harness.session.types import Entry, SessionError
from pi_agent.harness.types import BranchSummaryError, Result, err, ok
from pi_agent.types import AgentMessage


class BranchSummaryResult(TypedDict, total=False):
    summary: str
    usage: Usage
    read_files: list[str]
    modified_files: list[str]


class BranchSummaryDetails(TypedDict):
    read_files: list[str]
    modified_files: list[str]


class BranchPreparation(TypedDict):
    messages: list[AgentMessage]
    file_ops: FileOperations
    total_tokens: int


class CollectEntriesResult(TypedDict):
    entries: list[Entry]
    common_ancestor_id: str | None


class GenerateBranchSummaryOptions(TypedDict, total=False):
    models: Models
    model: Model
    abort: Any
    custom_instructions: str
    replace_instructions: bool
    reserve_tokens: int
    retry: RetryPolicy
    callbacks: RetryCallbacks


async def collect_entries_for_branch_summary(
    session: Session,
    old_leaf_id: str | None,
    target_id: str,
) -> CollectEntriesResult:
    if not old_leaf_id:
        return {"entries": [], "common_ancestor_id": None}
    old_path = {entry["id"] for entry in await session.find_entries_on_branch({"start": old_leaf_id})}
    target_path = await session.find_entries_on_branch({"start": target_id})
    common_ancestor_id = None
    for entry in target_path:
        if entry["id"] in old_path:
            common_ancestor_id = entry["id"]
            break
    entries: list[Entry] = []
    current: str | None = old_leaf_id
    while current and current != common_ancestor_id:
        entry = await session.get_entry(current)
        if not entry:
            raise SessionError("invalid_entry", f"Entry {current} not found")
        entries.append(entry)
        current = entry.get("parent_id")
    entries.reverse()
    return {"entries": entries, "common_ancestor_id": common_ancestor_id}


def _get_message_from_entry(entry: Entry) -> AgentMessage | None:
    entry_type = entry.get("type")
    if entry_type == "message":
        message = entry["message"]
        role = getattr(message, "role", None) if not isinstance(message, dict) else message.get("role")
        if role == "toolResult":
            return None
        return message
    if entry_type == "branch_summary":
        return create_branch_summary_message(entry["summary"], entry.get("from_id") or entry.get("fromId"), entry["timestamp"])
    if entry_type == "compaction":
        return create_compaction_summary_message(
            entry["summary"],
            entry.get("tokens_before", entry.get("tokensBefore", 0)),
            entry["timestamp"],
        )
    return None


def prepare_branch_entries(entries: list[Entry], token_budget: int = 0) -> BranchPreparation:
    messages: list[AgentMessage] = []
    file_ops = create_file_ops()
    total_tokens = 0
    for entry in entries:
        if entry.get("type") == "branch_summary" and entry.get("details"):
            details = entry["details"]
            read_files = details.get("read_files") or details.get("readFiles")
            modified_files = details.get("modified_files") or details.get("modifiedFiles")
            if isinstance(read_files, list):
                for path in read_files:
                    file_ops["read"].add(path)
            if isinstance(modified_files, list):
                for path in modified_files:
                    file_ops["edited"].add(path)
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        message = _get_message_from_entry(entry)
        if not message:
            continue
        extract_file_ops_from_message(message, file_ops)
        tokens = estimate_tokens(message)
        if token_budget > 0 and total_tokens + tokens > token_budget:
            if entry.get("type") in ("compaction", "branch_summary") and total_tokens < token_budget * 0.9:
                messages.insert(0, message)
                total_tokens += tokens
            break
        messages.insert(0, message)
        total_tokens += tokens
    return {"messages": messages, "file_ops": file_ops, "total_tokens": total_tokens}


BRANCH_SUMMARY_PREAMBLE = """The user explored a different conversation branch before returning here.
Summary of that exploration:

"""

BRANCH_SUMMARY_PROMPT = """Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


async def generate_branch_summary(entries: list[Entry], options: GenerateBranchSummaryOptions) -> Result:
    models = options["models"]
    model = options["model"]
    abort = options.get("abort")
    custom_instructions = options.get("custom_instructions")
    replace_instructions = options.get("replace_instructions")
    reserve_tokens = options.get("reserve_tokens", 16384)
    retry = options.get("retry")
    callbacks = options.get("callbacks")
    context_window = model.context_window or 128000
    token_budget = context_window - reserve_tokens
    prepared = prepare_branch_entries(entries, token_budget)
    if not prepared["messages"]:
        return ok({"summary": "No content to summarize", "read_files": [], "modified_files": []})
    conversation_text = serialize_conversation(convert_to_llm(prepared["messages"]))
    if replace_instructions and custom_instructions:
        instructions = custom_instructions
    elif custom_instructions:
        instructions = f"{BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {custom_instructions}"
    else:
        instructions = BRANCH_SUMMARY_PROMPT
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}"
    summarization_messages = [
        UserMessage(role="user", content=[{"type": "text", "text": prompt_text}], timestamp=int(time.time() * 1000))
    ]
    response = await complete_simple_with_retries(
        models,
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        SimpleStreamOptions(signal=abort, max_tokens=2048),
        retry,
        callbacks,
    )
    if response.stop_reason == "aborted":
        return err(BranchSummaryError("aborted", response.error_message or "Branch summary aborted"))
    if response.stop_reason == "error":
        return err(
            BranchSummaryError(
                "summarization_failed",
                f"Branch summary failed: {response.error_message or 'Unknown error'}",
            )
        )
    summary = BRANCH_SUMMARY_PREAMBLE + content_text(response.content)
    lists = compute_file_lists(prepared["file_ops"])
    summary += format_file_operations(lists["read_files"], lists["modified_files"])
    return ok(
        {
            "summary": summary or "No summary generated",
            "usage": response.usage,
            "read_files": lists["read_files"],
            "modified_files": lists["modified_files"],
        }
    )
