"""Pure lane recovery reducer — mirrors harness/reducer.ts."""
from __future__ import annotations

import copy
from typing import Any, Callable, Literal, TypedDict

from pi_agent.harness.session.types import Entry, LaneRecord, OperationStartedRecord, ProvisionedEntry
from pi_agent.types import AgentMessage, ThinkingLevel

RecordLogCorruptionReason = Literal[
    "multiple_open_operations",
    "unknown_operation",
    "record_after_finish",
    "non_consecutive_attempt",
    "invalid_compaction_reason",
    "queue_after_abort",
    "invalid_queue_cancellation",
    "inconsistent_step",
    "tool_call_mismatch",
    "duplicate_tool_invocation",
    "provisioned_entry_mismatch",
    "invalid_deferred_handle",
]


class RecordLogCorruption(Exception):
    def __init__(self, reason: RecordLogCorruptionReason, message: str) -> None:
        super().__init__(message)
        self.name = "RecordLogCorruption"
        self.reason = reason


class RecordLogSlice(TypedDict):
    lane: str
    open_operations: list[OperationStartedRecord]
    records: list[LaneRecord]
    entries: list[Entry]


class EffectiveLaneConfiguration(TypedDict):
    model: dict[str, str]
    thinking_level: ThinkingLevel
    active_tool_names: list[str]


class TerminalFailureState(TypedDict):
    entry_id: str
    source: Literal["step", "deferred_fetch"]
    message: Any


class ToolBatchState(TypedDict):
    assistant_entry_id: str
    calls: list[dict[str, Any]]
    truncated: bool
    unresolved: bool


class LaneState(TypedDict):
    lane: str
    leaf_id: str | None
    operation: dict[str, Any] | None
    pending_next_run: list[ProvisionedEntry]


class LaneReductionInput(RecordLogSlice):
    leaf_id: str | None
    own_entries: list[Entry]
    configuration_entries: list[Entry]
    defaults: EffectiveLaneConfiguration


class LaneReductionResult(TypedDict):
    lane_state: LaneState
    effective_configuration: EffectiveLaneConfiguration
    terminal_failure: TerminalFailureState | None


def _corrupt(reason: RecordLogCorruptionReason, message: str) -> None:
    raise RecordLogCorruption(reason, message)


def _has_run_id(record: LaneRecord) -> bool:
    return "run_id" in record and isinstance(record.get("run_id"), str)


def _deep_equal(left: Any, right: Any) -> bool:
    return left == right


def _matches_provisioned_entry(entry: Entry, target: ProvisionedEntry) -> bool:
    payload = {key: value for key, value in entry.items() if key not in ("parent_id", "seq", "timestamp")}
    return _deep_equal(payload, target)


def _validate_exact_provisioned_entry(entries_by_id: dict[str, Entry], target: ProvisionedEntry) -> None:
    entry = entries_by_id.get(target["id"])
    if entry and not _matches_provisioned_entry(entry, target):
        _corrupt("provisioned_entry_mismatch", f"Provisioned entry {target['id']} exists with content different from its intent")


def _validate_result_entry(
    entries_by_id: dict[str, Entry],
    result_entry_id: str,
    matches: Callable[[Entry], bool],
    description: str,
) -> None:
    entry = entries_by_id.get(result_entry_id)
    if entry and not matches(entry):
        _corrupt(
            "provisioned_entry_mismatch",
            f"Provisioned {description} entry {result_entry_id} exists with different content",
        )


def _validate_attempt_reason(record: LaneRecord) -> None:
    reason = record.get("compaction_reason")
    if record.get("step") == "compaction":
        if reason not in ("manual", "threshold", "overflow"):
            _corrupt("invalid_compaction_reason", f"Compaction attempt {record['id']} has no valid compaction reason")
    elif reason is not None:
        _corrupt("invalid_compaction_reason", f"{record.get('step')} attempt {record['id']} has a compaction reason")


def _validate_attempt_sequence(record: LaneRecord, previous: dict[str, Any] | None, entries_by_id: dict[str, Entry]) -> None:
    previous_record = previous["record"] if previous else None
    previous_result = entries_by_id.get(previous_record["result_entry_id"]) if previous_record else None
    continues_series = (
        previous_record is not None
        and previous_record.get("step") == record.get("step")
        and (previous_result is None or previous_result["seq"] >= record["seq"])
    )
    expected_attempt = previous_record["attempt"] + 1 if continues_series else 1
    if record.get("attempt") != expected_attempt:
        _corrupt(
            "non_consecutive_attempt",
            f"{record.get('step')} attempt {record['id']} is {record.get('attempt')}; expected {expected_attempt}",
        )
    if not continues_series or record.get("step") == "assistant" or previous_record is None:
        return
    if record.get("result_entry_id") != previous_record.get("result_entry_id"):
        _corrupt("inconsistent_step", f"{record.get('step')} attempts disagree on their result entry id")
    if record.get("compaction_reason") != previous_record.get("compaction_reason"):
        _corrupt("inconsistent_step", f"{record.get('step')} attempts disagree on their compaction reason")


def _message_role(message: Any) -> str | None:
    return getattr(message, "role", None) if not isinstance(message, dict) else message.get("role")


def _message_field(message: Any, snake: str, camel: str | None = None) -> Any:
    if isinstance(message, dict):
        return message.get(snake, message.get(camel or snake))
    return getattr(message, snake, None)


def _content_blocks(message: Any) -> list[Any]:
    content = _message_field(message, "content")
    return list(content) if content else []


def _validate_attempt_result(entries_by_id: dict[str, Entry], record: LaneRecord) -> None:
    step = record.get("step")
    result_id = record["result_entry_id"]
    if step == "assistant":
        _validate_result_entry(
            entries_by_id,
            result_id,
            lambda entry: entry.get("type") == "message" and _message_role(entry.get("message")) == "assistant",
            "assistant result",
        )
    elif step == "compaction":
        _validate_result_entry(entries_by_id, result_id, lambda entry: entry.get("type") == "compaction", "compaction result")
    elif step == "branch_summary":
        _validate_result_entry(
            entries_by_id,
            result_id,
            lambda entry: entry.get("type") == "branch_summary",
            "branch-summary result",
        )


def _validate_tool_start(record: LaneRecord, entries_by_id: dict[str, Entry], invocations: set[str]) -> None:
    invocation = f"{record['assistant_entry_id']}\0{record['tool_index']}"
    if invocation in invocations:
        _corrupt(
            "duplicate_tool_invocation",
            f"Tool invocation {record['assistant_entry_id']}:{record['tool_index']} is duplicated",
        )
    invocations.add(invocation)
    assistant_entry = entries_by_id.get(record["assistant_entry_id"])
    if (
        not assistant_entry
        or assistant_entry.get("type") != "message"
        or _message_role(assistant_entry.get("message")) != "assistant"
    ):
        _corrupt("tool_call_mismatch", f"Tool start {record['id']} does not reference an assistant entry")
    tool_calls = [block for block in _content_blocks(assistant_entry["message"]) if _block_type(block) == "toolCall"]
    tool_call = tool_calls[record["tool_index"]] if record["tool_index"] < len(tool_calls) else None
    if (
        not tool_call
        or _block_field(tool_call, "id") != record["tool_call_id"]
        or _block_field(tool_call, "name") != record["tool_name"]
    ):
        _corrupt("tool_call_mismatch", f"Tool start {record['id']} does not match its assistant tool-call ordinal")
    _validate_result_entry(
        entries_by_id,
        record["result_entry_id"],
        lambda entry: (
            entry.get("type") == "message"
            and _message_role(entry.get("message")) == "toolResult"
            and _message_field(entry.get("message"), "tool_call_id", "toolCallId") == record["tool_call_id"]
            and _message_field(entry.get("message"), "tool_name", "toolName") == record["tool_name"]
        ),
        "tool result",
    )


def _block_type(block: Any) -> str | None:
    return getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")


def _block_field(block: Any, name: str) -> Any:
    return getattr(block, name, None) if not isinstance(block, dict) else block.get(name)


def _validate_deferred_handles(entries: Any) -> None:
    for entry in entries:
        if (
            entry.get("type") == "message"
            and _message_role(entry.get("message")) == "assistant"
            and _message_field(entry.get("message"), "stop_reason", "stopReason") == "deferred"
            and not _message_field(entry.get("message"), "deferred")
        ):
            _corrupt("invalid_deferred_handle", f"Deferred assistant entry {entry['id']} does not carry a handle")


def _validate_operation_result(entries_by_id: dict[str, Entry], record: OperationStartedRecord) -> None:
    intent = record.get("intent") or {}
    kind = intent.get("kind")
    if kind == "run":
        for target in intent.get("initial_messages") or intent.get("initialMessages") or []:
            _validate_exact_provisioned_entry(entries_by_id, target)
    elif kind == "compaction":
        result_id = intent.get("result_entry_id") or intent.get("resultEntryId")
        _validate_result_entry(entries_by_id, result_id, lambda entry: entry.get("type") == "compaction", "manual compaction")
    elif kind == "navigation":
        summary_id = intent.get("summary_entry_id") or intent.get("summaryEntryId")
        if summary_id:
            _validate_result_entry(
                entries_by_id,
                summary_id,
                lambda entry: entry.get("type") == "branch_summary",
                "navigation summary",
            )


def validate_record_log(input_data: RecordLogSlice) -> None:
    if len(input_data["open_operations"]) > 1:
        _corrupt("multiple_open_operations", f"Lane {input_data['lane']} has at least two open operations")
    entries_by_id = {entry["id"]: entry for entry in input_data["entries"]}
    _validate_deferred_handles(entries_by_id.values())
    starts: dict[str, OperationStartedRecord] = {}
    finished_at: dict[str, int] = {}
    aborted_at: dict[str, int] = {}
    queue_enqueues: dict[str, LaneRecord] = {}
    latest_attempt: dict[str, dict[str, Any]] = {}
    tool_invocations: set[str] = set()
    records = sorted(input_data["records"], key=lambda item: item["seq"])
    for record in records:
        if record.get("type") == "operation_started":
            starts[record["id"]] = record
            _validate_operation_result(entries_by_id, record)
            continue
        if _has_run_id(record):
            if record["run_id"] not in starts:
                _corrupt("unknown_operation", f"Record {record['id']} references unknown operation {record['run_id']}")
            finish_seq = finished_at.get(record["run_id"])
            if finish_seq is not None and record["seq"] > finish_seq:
                _corrupt("record_after_finish", f"Record {record['id']} follows the finish of operation {record['run_id']}")
        record_type = record.get("type")
        if record_type == "operation_finished":
            finished_at[record["run_id"]] = record["seq"]
        elif record_type == "abort_requested":
            aborted_at[record["run_id"]] = record["seq"]
        elif record_type == "step_attempt":
            _validate_attempt_reason(record)
            _validate_attempt_sequence(record, latest_attempt.get(record["run_id"]), entries_by_id)
            _validate_attempt_result(entries_by_id, record)
            latest_attempt[record["run_id"]] = {"record": record}
        elif record_type == "tool_started":
            _validate_tool_start(record, entries_by_id, tool_invocations)
        elif record_type == "queue_enqueued":
            abort_seq = aborted_at.get(record.get("run_id"))
            if record.get("queue") != "nextRun" and abort_seq is not None and record["seq"] > abort_seq:
                _corrupt("queue_after_abort", f"{record.get('queue')} item {record['target']['id']} was enqueued after abort")
            queue_enqueues[record["target"]["id"]] = record
            _validate_exact_provisioned_entry(entries_by_id, record["target"])
        elif record_type == "queue_cancelled":
            enqueue = queue_enqueues.get(record["entry_id"])
            if (
                not enqueue
                or enqueue["seq"] >= record["seq"]
                or enqueue.get("run_id") != record.get("run_id")
                or record["entry_id"] in entries_by_id
            ):
                _corrupt("invalid_queue_cancellation", f"Queue cancellation {record['id']} has no pending matching enqueue")
        elif record_type == "write_deferred":
            _validate_exact_provisioned_entry(entries_by_id, record["target"])


def _by_sequence(values: list[Any]) -> list[Any]:
    return sorted(values, key=lambda item: item["seq"])


def _derive_effective_configuration(input_data: LaneReductionInput) -> EffectiveLaneConfiguration:
    configuration = copy.deepcopy(input_data["defaults"])
    entries_by_id: dict[str, Entry] = {}
    for entry in [*input_data["configuration_entries"], *input_data["own_entries"]]:
        entries_by_id[entry["id"]] = entry
    for entry in _by_sequence(list(entries_by_id.values())):
        entry_type = entry.get("type")
        if entry_type == "model_change":
            configuration = {
                **configuration,
                "model": {"provider": entry["provider"], "model_id": entry.get("model_id") or entry.get("modelId")},
            }
        elif entry_type == "thinking_level_change":
            configuration = {**configuration, "thinking_level": entry.get("thinking_level") or entry.get("thinkingLevel")}
        elif entry_type == "active_tools_change":
            configuration = {
                **configuration,
                "active_tool_names": list(entry.get("active_tool_names") or entry.get("activeToolNames") or []),
            }
        elif entry_type == "message" and _message_role(entry.get("message")) == "assistant":
            message = entry["message"]
            configuration = {
                **configuration,
                "model": {
                    "provider": _message_field(message, "provider"),
                    "model_id": _message_field(message, "model"),
                },
            }
    return configuration


def _derive_newest_own(entry: Entry | None) -> dict[str, Any] | None:
    if not entry:
        return None
    if entry.get("type") != "message":
        return {"entry_id": entry["id"], "type": entry.get("type")}
    role = _message_role(entry.get("message"))
    if role != "assistant":
        return {"entry_id": entry["id"], "type": entry.get("type"), "role": role}
    return {
        "entry_id": entry["id"],
        "type": entry.get("type"),
        "role": role,
        "stop_reason": _message_field(entry.get("message"), "stop_reason", "stopReason"),
    }


def _derive_tool_batch(
    operation_id: str,
    records: list[LaneRecord],
    own_entries: list[Entry],
    entries_by_id: dict[str, Entry],
    deferred_write_ids: set[str],
) -> ToolBatchState | None:
    assistant_entry = None
    for entry in reversed(own_entries):
        if (
            entry.get("type") == "message"
            and _message_role(entry.get("message")) == "assistant"
            and any(_block_type(block) == "toolCall" for block in _content_blocks(entry.get("message")))
        ):
            assistant_entry = entry
            break
    if assistant_entry is None:
        return None
    tool_calls = [block for block in _content_blocks(assistant_entry["message"]) if _block_type(block) == "toolCall"]
    starts: dict[int, LaneRecord] = {}
    for record in records:
        if (
            record.get("type") == "tool_started"
            and record.get("run_id") == operation_id
            and record.get("assistant_entry_id") == assistant_entry["id"]
        ):
            starts[record["tool_index"]] = record
    calls = []
    for tool_index, tool_call in enumerate(tool_calls):
        started = starts.get(tool_index)
        started_result = entries_by_id.get(started["result_entry_id"]) if started else None
        blocked_result = next(
            (
                entry
                for entry in own_entries
                if entry["seq"] > assistant_entry["seq"]
                and entry["id"] not in deferred_write_ids
                and entry.get("type") == "message"
                and _message_role(entry.get("message")) == "toolResult"
                and _message_field(entry.get("message"), "tool_call_id", "toolCallId") == _block_field(tool_call, "id")
            ),
            None,
        )
        result = started_result or blocked_result
        call: dict[str, Any] = {
            "tool_index": tool_index,
            "tool_call": copy.deepcopy(tool_call),
            "result_exists": result is not None,
        }
        if started:
            call["started"] = copy.deepcopy(started)
        if result and result.get("type") == "message" and result.get("terminate") is True:
            call["terminate"] = True
        calls.append(call)
    return ToolBatchState(
        assistant_entry_id=assistant_entry["id"],
        calls=calls,
        truncated=_message_field(assistant_entry["message"], "stop_reason", "stopReason") == "length",
        unresolved=any(not call["result_exists"] for call in calls),
    )


def reduce_lane_records(input_data: LaneReductionInput) -> LaneReductionResult:
    return reduce_lane_state(input_data)


def reduce_lane_state(input_data: LaneReductionInput) -> LaneReductionResult:
    validate_record_log(input_data)
    records = _by_sequence(input_data["records"])
    own_entries = _by_sequence(input_data["own_entries"])
    entries_by_id: dict[str, Entry] = {}
    for entry in [*input_data["entries"], *own_entries]:
        entries_by_id[entry["id"]] = entry
    cancelled_queue_ids = {record["entry_id"] for record in records if record.get("type") == "queue_cancelled"}
    pending_queue_records = [
        record
        for record in records
        if record.get("type") == "queue_enqueued"
        and record["target"]["id"] not in entries_by_id
        and record["target"]["id"] not in cancelled_queue_ids
    ]
    started = input_data["open_operations"][0] if input_data["open_operations"] else None
    captured_ids = set()
    if started and (started.get("intent") or {}).get("kind") == "run":
        for target in (started["intent"].get("initial_messages") or started["intent"].get("initialMessages") or []):
            captured_ids.add(target["id"])
    pending_next_run = [
        copy.deepcopy(record["target"])
        for record in pending_queue_records
        if record.get("queue") == "nextRun" and record["target"]["id"] not in captured_ids
    ]
    effective_configuration = _derive_effective_configuration(input_data)
    if not started:
        return LaneReductionResult(
            lane_state=LaneState(
                lane=input_data["lane"],
                leaf_id=input_data["leaf_id"],
                operation=None,
                pending_next_run=pending_next_run,
            ),
            effective_configuration=effective_configuration,
            terminal_failure=None,
        )

    operation_records = [
        record
        for record in records
        if (record.get("type") == "operation_started" and record["id"] == started["id"])
        or record.get("run_id") == started["id"]
    ]
    aborting = any(record.get("type") == "abort_requested" for record in operation_records)
    pending_steer = (
        []
        if aborting
        else [
            copy.deepcopy(record["target"])
            for record in pending_queue_records
            if record.get("queue") == "steer" and record.get("run_id") == started["id"]
        ]
    )
    pending_follow_up = (
        []
        if aborting
        else [
            copy.deepcopy(record["target"])
            for record in pending_queue_records
            if record.get("queue") == "followUp" and record.get("run_id") == started["id"]
        ]
    )
    pending_writes = [
        copy.deepcopy(record["target"])
        for record in operation_records
        if record.get("type") == "write_deferred" and record["target"]["id"] not in entries_by_id
    ]
    missing_initial = (
        [
            copy.deepcopy(target)
            for target in (started["intent"].get("initial_messages") or started["intent"].get("initialMessages") or [])
            if target["id"] not in entries_by_id
        ]
        if (started.get("intent") or {}).get("kind") == "run"
        else []
    )
    newest_attempt = next((record for record in reversed(operation_records) if record.get("type") == "step_attempt"), None)
    step = None
    if newest_attempt and newest_attempt["result_entry_id"] not in entries_by_id:
        step = {
            "kind": newest_attempt["step"],
            "attempts": newest_attempt["attempt"],
            "result_entry_id": newest_attempt["result_entry_id"],
        }
        if newest_attempt.get("step") == "compaction":
            step["compaction_reason"] = newest_attempt.get("compaction_reason")

    consumed_input_ids: set[str] = set()
    if (started.get("intent") or {}).get("kind") == "run":
        for target in started["intent"].get("initial_messages") or started["intent"].get("initialMessages") or []:
            consumed_input_ids.add(target["id"])
    for record in operation_records:
        if record.get("type") == "queue_enqueued" and record.get("queue") != "nextRun":
            consumed_input_ids.add(record["target"]["id"])
    newest_consumed = float("-inf")
    for entry_id in consumed_input_ids:
        entry = entries_by_id.get(entry_id)
        if entry and entry.get("type") == "message":
            newest_consumed = max(newest_consumed, entry["seq"])
    overflow_recovery_used = any(
        record.get("type") == "step_attempt"
        and record.get("step") == "compaction"
        and record.get("compaction_reason") == "overflow"
        and record["seq"] > newest_consumed
        for record in operation_records
    )
    newest_own_entry = own_entries[-1] if own_entries else None
    newest_own = _derive_newest_own(newest_own_entry)
    deferred = None
    if (
        newest_own_entry
        and newest_own_entry.get("type") == "message"
        and _message_role(newest_own_entry.get("message")) == "assistant"
        and _message_field(newest_own_entry.get("message"), "stop_reason", "stopReason") == "deferred"
        and _message_field(newest_own_entry.get("message"), "deferred")
    ):
        deferred = copy.deepcopy(_message_field(newest_own_entry.get("message"), "deferred"))
    targets: dict[str, bool] = {}
    intent = started.get("intent") or {}
    if intent.get("kind") == "compaction":
        targets["result"] = (intent.get("result_entry_id") or intent.get("resultEntryId")) in entries_by_id
    elif intent.get("kind") == "navigation" and (intent.get("summary_entry_id") or intent.get("summaryEntryId")):
        targets["summary"] = (intent.get("summary_entry_id") or intent.get("summaryEntryId")) in entries_by_id
    deferred_write_ids = {record["target"]["id"] for record in operation_records if record.get("type") == "write_deferred"}
    terminal_failure = None
    if (
        newest_own_entry
        and newest_own_entry.get("type") == "message"
        and _message_role(newest_own_entry.get("message")) == "assistant"
        and _message_field(newest_own_entry.get("message"), "stop_reason", "stopReason") == "error"
        and newest_own_entry["id"] not in deferred_write_ids
    ):
        produced_by_step = any(
            record.get("type") == "step_attempt" and record.get("result_entry_id") == newest_own_entry["id"]
            for record in operation_records
        )
        previous_own = own_entries[-2] if len(own_entries) >= 2 else None
        produced_by_deferred = any(
            record.get("type") == "usage"
            and record.get("cause") == "deferred_fetch"
            and record.get("entry_id") == newest_own_entry["id"]
            for record in operation_records
        ) or (
            previous_own
            and previous_own.get("type") == "message"
            and _message_role(previous_own.get("message")) == "assistant"
            and _message_field(previous_own.get("message"), "stop_reason", "stopReason") == "deferred"
        )
        if produced_by_step or produced_by_deferred:
            terminal_failure = TerminalFailureState(
                entry_id=newest_own_entry["id"],
                source="step" if produced_by_step else "deferred_fetch",
                message=copy.deepcopy(newest_own_entry["message"]),
            )
    return LaneReductionResult(
        lane_state=LaneState(
            lane=input_data["lane"],
            leaf_id=input_data["leaf_id"],
            operation={
                "id": started["id"],
                "kind": intent.get("kind"),
                "intent": copy.deepcopy(intent),
                "aborting": aborting,
                "step": step,
                "tool_batch": _derive_tool_batch(started["id"], operation_records, own_entries, entries_by_id, deferred_write_ids),
                "missing_initial_messages": missing_initial,
                "pending_steer": pending_steer,
                "pending_follow_up": pending_follow_up,
                "pending_writes": pending_writes,
                "deferred": deferred,
                "overflow_recovery_used": overflow_recovery_used,
                "newest_own": newest_own,
                "targets": targets,
            },
            pending_next_run=pending_next_run,
        ),
        effective_configuration=effective_configuration,
        terminal_failure=terminal_failure,
    )
