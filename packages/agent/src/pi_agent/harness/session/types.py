"""Session protocol types — mirrors harness/session/types.ts."""
from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict

from pi_agent.types import AgentMessage

JsonValue = Any
SessionStopReason = Literal["stop", "length", "toolUse", "error", "aborted", "deferred"]


class IdGenerator(Protocol):
    def next(self) -> str: ...


class EntryBase(TypedDict):
    type: str
    id: str
    seq: int
    parent_id: str | None
    timestamp: int


class MessageEntry(TypedDict, total=False):
    type: Literal["message"]
    id: str
    seq: int
    parent_id: str | None
    timestamp: int
    message: AgentMessage
    terminate: Literal[True]


class ModelChangeEntry(TypedDict):
    type: Literal["model_change"]
    id: str
    seq: int
    parent_id: str | None
    timestamp: int
    provider: str
    model_id: str


class ThinkingLevelEntry(TypedDict):
    type: Literal["thinking_level_change"]
    id: str
    seq: int
    parent_id: str | None
    timestamp: int
    thinking_level: str


class ActiveToolsEntry(TypedDict):
    type: Literal["active_tools_change"]
    id: str
    seq: int
    parent_id: str | None
    timestamp: int
    active_tool_names: list[str]


class CompactionEntry(TypedDict, total=False):
    type: Literal["compaction"]
    id: str
    seq: int
    parent_id: str | None
    timestamp: int
    summary: str
    retained_tail: list[AgentMessage]
    tokens_before: int
    details: Any
    usage: Any


class BranchSummaryEntry(TypedDict, total=False):
    type: Literal["branch_summary"]
    id: str
    seq: int
    parent_id: str | None
    timestamp: int
    from_id: str
    summary: str
    details: Any
    usage: Any


class CustomEntry(TypedDict, total=False):
    type: Literal["custom"]
    id: str
    seq: int
    parent_id: str | None
    timestamp: int
    custom_type: str
    data: Any


Entry = dict[str, Any]
ProvisionedEntry = dict[str, Any]


class RecordBase(TypedDict):
    id: str
    seq: int
    lane: str
    timestamp: int


class OperationStartedRecord(TypedDict, total=False):
    type: Literal["operation_started"]
    id: str
    seq: int
    lane: str
    timestamp: int
    source_leaf_id: str | None
    intent: dict[str, Any]


class AbortRequestedRecord(TypedDict):
    type: Literal["abort_requested"]
    id: str
    seq: int
    lane: str
    timestamp: int
    run_id: str


class OperationFinishedRecord(TypedDict, total=False):
    type: Literal["operation_finished"]
    id: str
    seq: int
    lane: str
    timestamp: int
    run_id: str
    outcome: Literal["completed", "aborted", "failed", "declined"]
    error: dict[str, str]


class StepAttemptRecord(TypedDict, total=False):
    type: Literal["step_attempt"]
    id: str
    seq: int
    lane: str
    timestamp: int
    run_id: str
    step: Literal["assistant", "branch_summary", "compaction"]
    attempt: int
    result_entry_id: str
    compaction_reason: Literal["manual", "threshold", "overflow"]


class ToolStartedRecord(TypedDict):
    type: Literal["tool_started"]
    id: str
    seq: int
    lane: str
    timestamp: int
    run_id: str
    assistant_entry_id: str
    tool_index: int
    tool_call_id: str
    tool_name: str
    effective_args: dict[str, Any]
    result_entry_id: str
    replay: Literal["never", "safe"]


class QueueEnqueuedRecord(TypedDict, total=False):
    type: Literal["queue_enqueued"]
    id: str
    seq: int
    lane: str
    timestamp: int
    queue: Literal["steer", "followUp", "nextRun"]
    run_id: str
    target: ProvisionedEntry


class QueueCancelledRecord(TypedDict, total=False):
    type: Literal["queue_cancelled"]
    id: str
    seq: int
    lane: str
    timestamp: int
    run_id: str
    entry_id: str


class WriteDeferredRecord(TypedDict):
    type: Literal["write_deferred"]
    id: str
    seq: int
    lane: str
    timestamp: int
    run_id: str
    target: ProvisionedEntry


class UsageRecord(TypedDict, total=False):
    type: Literal["usage"]
    id: str
    seq: int
    lane: str
    timestamp: int
    usage: Any
    cause: str
    run_id: str
    entry_id: str
    attempt: int
    stop_reason: SessionStopReason
    tool_call_id: str
    details: JsonValue


LaneRecord = dict[str, Any]
NewRecord = dict[str, Any]

EntryOrder = Literal["newestFirst", "oldestFirst"]
CompactionReason = Literal["manual", "threshold", "overflow"]


class EntryCursor(TypedDict):
    after_seq: int


class EntryQuery(TypedDict, total=False):
    type: str
    custom_type: str
    order: EntryOrder
    limit: int
    cursor: EntryCursor


class BranchBounds(TypedDict, total=False):
    start: str
    stop_at_type: str
    stop_at_id: str


class RecordQuery(TypedDict, total=False):
    lane: str
    type: str
    run_id: str
    operation_kind: str
    after_seq: int
    order: EntryOrder
    limit: int


class SessionMetadata(TypedDict, total=False):
    id: str
    created_at: int
    parent_session_id: str


class SessionStats(TypedDict):
    message_count: int
    cached_tokens: int
    uncached_tokens: int
    total_tokens: int
    cost_total: float


class LanePointer(TypedDict):
    lane: str
    leaf_id: str | None


LogItem = dict[str, Any]


class LogOptions(TypedDict, total=False):
    after_seq: int
    limit: int


class SessionCreateOptions(TypedDict, total=False):
    id: str
    parent_session_id: str


ForkOptions = dict[str, Any]

SessionErrorCode = Literal[
    "not_found",
    "already_exists",
    "invalid_entry",
    "invalid_payload",
    "invalid_lane",
    "invalid_query",
    "invalid_fork_target",
    "storage",
]


class SessionError(Exception):
    def __init__(self, code: SessionErrorCode, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.name = "SessionError"
        self.code = code
        if cause is not None:
            self.__cause__ = cause
