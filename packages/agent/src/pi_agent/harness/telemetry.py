"""Harness telemetry — noop + same schema type names as harness/telemetry.ts."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

TelemetryContext = Any
TelemetrySpan = Any
ExactTelemetryAttributes = dict[str, Any]
SchemaTelemetrySpan = Any
TelemetrySchemaSpanEndAttributes = dict[str, Any]
TelemetrySchemaSpanEventAttributes = dict[str, Any]
TelemetrySchemaSpanEventName = str
TelemetrySchemaSpanName = str
TelemetrySchemaSpanStartAttributes = dict[str, Any]
TelemetrySchemaSpanUnion = Any

AI_TELEMETRY_SCHEMA = {
    "version": 1,
    "spans": {
        "pi.ai.request": {
            "description": "One logical request to an AI provider",
            "parents": {"kind": "any"},
            "startAttributes": {
                "pi.ai.operation": {
                    "type": "string",
                    "required": True,
                    "values": ["stream", "fetch_deferred", "cancel_deferred", "generate_images"],
                    "description": "Logical provider operation",
                },
                "pi.ai.provider": {"type": "string", "required": True, "description": "Selected provider id"},
                "pi.ai.model": {"type": "string", "required": True, "description": "Requested model id"},
                "pi.ai.api": {"type": "string", "required": True, "description": "Provider API id"},
                "pi.ai.streaming": {
                    "type": "boolean",
                    "required": True,
                    "description": "Whether this operation returns a stream",
                },
                "pi.ai.deferred": {
                    "type": "boolean",
                    "required": False,
                    "description": "Whether the operation requests or participates in deferred execution",
                },
            },
            "endAttributes": {
                "pi.ai.response.model": {"type": "string", "description": "Concrete response model"},
                "pi.ai.response.id": {"type": "string", "cardinality": "high", "description": "Provider response id"},
                "pi.ai.response.stop_reason": {
                    "type": "string",
                    "values": ["stop", "length", "tool_use", "error", "aborted", "deferred"],
                    "description": "Normalized terminal response reason",
                },
                "pi.ai.http.status_code": {"type": "number", "description": "Final HTTP status"},
                "pi.ai.usage.input_tokens": {"type": "number", "description": "Reported input tokens"},
                "pi.ai.usage.output_tokens": {"type": "number", "description": "Reported output tokens"},
                "pi.ai.usage.cache_read_tokens": {"type": "number", "description": "Reported cache-read tokens"},
                "pi.ai.usage.cache_write_tokens": {"type": "number", "description": "Reported cache-write tokens"},
                "pi.ai.usage.reasoning_tokens": {"type": "number", "description": "Reported reasoning tokens"},
                "pi.ai.usage.total_tokens": {"type": "number", "description": "Reported total tokens"},
                "pi.ai.usage.cost": {"type": "number", "description": "Reported total cost"},
                "pi.ai.stream.chunk_count": {"type": "number", "description": "Streamed update chunk count"},
                "pi.ai.stream.time_to_first_chunk_ms": {
                    "type": "number",
                    "description": "Elapsed milliseconds to first update chunk",
                },
                "pi.ai.error.type": {
                    "type": "string",
                    "cardinality": "low",
                    "description": "Provider or transport error class",
                },
            },
            "status": {"default": "ok", "errorWhen": "The operation throws or returns an error result"},
        }
    },
}

HOOK_NAMES = (
    "before_run",
    "before_resume",
    "before_run_end",
    "transform_context",
    "before_request",
    "before_payload",
    "after_response",
    "before_tool",
    "after_tool",
    "before_compaction",
    "before_navigation",
)

EVENT_TYPES = (
    "run_start",
    "run_resume",
    "run_suspend",
    "run_abort",
    "run_end",
    "fault",
    "handler_error",
    "turn_start",
    "turn_end",
    "retry_scheduled",
    "retry_start",
    "retry_end",
    "message_start",
    "message_update",
    "message_end",
    "tool_start",
    "tool_update",
    "tool_end",
    "entry_added",
    "write_pending",
    "queue_update",
    "fact_update",
    "config_update",
    "compaction_start",
    "compaction_end",
    "navigation_start",
    "navigation_end",
    "lane_created",
    "usage",
)

_OPERATION_START_ATTRIBUTES = {
    "pi.session.id": {
        "type": "string",
        "required": True,
        "cardinality": "high",
        "description": "Session id",
    },
    "pi.lane.name": {
        "type": "string",
        "required": True,
        "cardinality": "high",
        "description": "Lane name",
    },
    "pi.operation.id": {
        "type": "string",
        "required": True,
        "cardinality": "high",
        "description": "Durable operation id",
    },
    "pi.operation.recovery": {
        "type": "boolean",
        "required": True,
        "description": "Whether this invocation resumes durable work",
    },
}

_OPERATION_ERROR_ATTRIBUTES = {
    "pi.error.code": {
        "type": "string",
        "cardinality": "low",
        "description": "Stable operation error code",
    },
    "pi.error.type": {
        "type": "string",
        "cardinality": "low",
        "description": "Low-cardinality operation error class",
    },
}

HARNESS_TELEMETRY_SCHEMA = {
    "version": 1,
    "spans": {
        "pi.harness.run": {
            "description": "One admitted in-process run invocation",
            "parents": {"kind": "root_or_external"},
            "startAttributes": {
                **_OPERATION_START_ATTRIBUTES,
                "pi.operation.kind": {
                    "type": "string",
                    "required": True,
                    "values": ["run"],
                    "description": "Run operation kind",
                },
            },
            "endAttributes": {
                "pi.operation.outcome": {
                    "type": "string",
                    "values": ["completed", "aborted", "failed", "suspended"],
                    "description": "Run invocation outcome",
                },
                **_OPERATION_ERROR_ATTRIBUTES,
            },
            "status": {"default": "ok", "errorWhen": "The run fails or throws"},
        },
        "pi.harness.compaction": {
            "description": "One admitted in-process manual compaction invocation",
            "parents": {"kind": "root_or_external"},
            "startAttributes": {
                **_OPERATION_START_ATTRIBUTES,
                "pi.operation.kind": {
                    "type": "string",
                    "required": True,
                    "values": ["compaction"],
                    "description": "Compaction operation kind",
                },
            },
            "endAttributes": {
                "pi.operation.outcome": {
                    "type": "string",
                    "values": ["completed", "declined", "aborted", "failed"],
                    "description": "Compaction invocation outcome",
                },
                **_OPERATION_ERROR_ATTRIBUTES,
            },
            "status": {"default": "ok", "errorWhen": "The compaction fails or throws"},
        },
        "pi.harness.navigation": {
            "description": "One admitted in-process navigation invocation",
            "parents": {"kind": "root_or_external"},
            "startAttributes": {
                **_OPERATION_START_ATTRIBUTES,
                "pi.operation.kind": {
                    "type": "string",
                    "required": True,
                    "values": ["navigation"],
                    "description": "Navigation operation kind",
                },
            },
            "endAttributes": {
                "pi.operation.outcome": {
                    "type": "string",
                    "values": ["completed", "declined", "aborted", "failed"],
                    "description": "Navigation invocation outcome",
                },
                **_OPERATION_ERROR_ATTRIBUTES,
            },
            "status": {"default": "ok", "errorWhen": "The navigation fails or throws"},
        },
        "pi.harness.checkpoint": {
            "description": "One run checkpoint",
            "parents": {"kind": "spans", "spans": ["pi.harness.run"]},
            "startAttributes": {
                "pi.lane.name": {
                    "type": "string",
                    "required": True,
                    "cardinality": "high",
                    "description": "Lane name",
                },
                "pi.operation.id": {
                    "type": "string",
                    "required": True,
                    "cardinality": "high",
                    "description": "Durable operation id",
                },
                "pi.checkpoint.kind": {
                    "type": "string",
                    "required": True,
                    "values": ["normal", "failure_drain", "abort_reconcile"],
                    "description": "Checkpoint purpose",
                },
            },
            "endAttributes": {},
            "status": {"default": "ok", "errorWhen": "Checkpoint work throws"},
        },
        "pi.harness.turn": {
            "description": "One assistant response and its tool batch",
            "parents": {"kind": "spans", "spans": ["pi.harness.run"]},
            "startAttributes": {
                "pi.lane.name": {
                    "type": "string",
                    "required": True,
                    "cardinality": "high",
                    "description": "Lane name",
                },
                "pi.operation.id": {
                    "type": "string",
                    "required": True,
                    "cardinality": "high",
                    "description": "Durable operation id",
                },
                "pi.turn.id": {
                    "type": "string",
                    "required": True,
                    "cardinality": "high",
                    "description": "Invocation-local turn id",
                },
            },
            "endAttributes": {},
            "status": {"default": "ok", "errorWhen": "Turn work throws"},
        },
        "pi.harness.step": {
            "description": "One durable retry attempt",
            "parents": {
                "kind": "spans",
                "spans": ["pi.harness.turn", "pi.harness.checkpoint", "pi.harness.compaction", "pi.harness.navigation"],
            },
            "startAttributes": {
                "pi.lane.name": {
                    "type": "string",
                    "required": True,
                    "cardinality": "high",
                    "description": "Lane name",
                },
                "pi.operation.id": {
                    "type": "string",
                    "required": True,
                    "cardinality": "high",
                    "description": "Durable operation id",
                },
                "pi.step.kind": {
                    "type": "string",
                    "required": True,
                    "values": ["assistant", "compaction", "branch_summary"],
                    "description": "Retryable step kind",
                },
                "pi.step.attempt": {
                    "type": "number",
                    "required": True,
                    "description": "One-based durable attempt number",
                },
                "pi.compaction.reason": {
                    "type": "string",
                    "required": False,
                    "values": ["manual", "threshold", "overflow"],
                    "description": "Compaction trigger",
                },
            },
            "endAttributes": {
                "pi.step.outcome": {
                    "type": "string",
                    "values": ["succeeded", "retry", "failed", "aborted", "deferred", "overflow"],
                    "description": "Attempt outcome",
                },
            },
            "status": {"default": "ok", "errorWhen": "The attempt retries, fails, or throws"},
        },
        "pi.harness.tool": {
            "description": "One raw phase-2 tool execution",
            "parents": {"kind": "spans", "spans": ["pi.harness.turn", "pi.harness.run"]},
            "startAttributes": {
                "pi.lane.name": {
                    "type": "string",
                    "required": True,
                    "cardinality": "high",
                    "description": "Lane name",
                },
                "pi.operation.id": {
                    "type": "string",
                    "required": True,
                    "cardinality": "high",
                    "description": "Durable operation id",
                },
                "pi.turn.id": {
                    "type": "string",
                    "required": False,
                    "cardinality": "high",
                    "description": "Invocation-local live turn id",
                },
                "pi.tool.name": {"type": "string", "required": True, "description": "Tool name"},
                "pi.tool.call_id": {
                    "type": "string",
                    "required": True,
                    "cardinality": "high",
                    "description": "Tool call id",
                },
                "pi.tool.replay": {
                    "type": "string",
                    "required": True,
                    "values": ["never", "safe"],
                    "description": "Declared replay policy",
                },
                "pi.tool.recovery": {
                    "type": "boolean",
                    "required": True,
                    "description": "Whether this is recovery execution",
                },
            },
            "endAttributes": {
                "pi.tool.is_error": {
                    "type": "boolean",
                    "description": "Whether raw phase-2 execution returned an error",
                },
            },
            "status": {"default": "ok", "errorWhen": "Raw phase-2 execution returns an error"},
        },
        "pi.harness.hook": {
            "description": "One registered hook handler invocation",
            "parents": {"kind": "any"},
            "startAttributes": {
                "pi.lane.name": {
                    "type": "string",
                    "required": True,
                    "cardinality": "high",
                    "description": "Lane name",
                },
                "pi.operation.id": {
                    "type": "string",
                    "required": False,
                    "cardinality": "high",
                    "description": "Durable operation id when accepted",
                },
                "pi.hook.name": {
                    "type": "string",
                    "required": True,
                    "values": HOOK_NAMES,
                    "description": "Hook name",
                },
                "pi.hook.registration_id": {
                    "type": "string",
                    "required": False,
                    "description": "Stable hook registration id",
                },
            },
            "endAttributes": {
                "pi.hook.outcome": {
                    "type": "string",
                    "values": ["completed", "skipped", "blocked", "failed"],
                    "description": "Handler outcome",
                },
            },
            "status": {"default": "ok", "errorWhen": "The handler throws"},
        },
        "pi.harness.sleep": {
            "description": "One retry delay",
            "parents": {"kind": "spans", "spans": ["pi.harness.step", "pi.harness.run"]},
            "startAttributes": {
                "pi.operation.id": {
                    "type": "string",
                    "required": True,
                    "cardinality": "high",
                    "description": "Durable operation id",
                },
                "pi.sleep.delay_ms": {
                    "type": "number",
                    "required": True,
                    "description": "Requested delay in milliseconds",
                },
            },
            "endAttributes": {
                "pi.sleep.outcome": {
                    "type": "string",
                    "values": ["elapsed", "aborted"],
                    "description": "Delay outcome",
                },
            },
            "status": {"default": "ok", "errorWhen": "Sleep work throws"},
        },
        "pi.harness.event_handler": {
            "description": "One passive event listener invocation",
            "parents": {"kind": "any"},
            "startAttributes": {
                "pi.event.type": {
                    "type": "string",
                    "required": True,
                    "cardinality": "low",
                    "values": EVENT_TYPES,
                    "description": "Delivered harness event type",
                },
                "pi.lane.name": {
                    "type": "string",
                    "required": False,
                    "cardinality": "high",
                    "description": "Lane name for lane-scoped events",
                },
            },
            "endAttributes": {},
            "status": {"default": "ok", "errorWhen": "The listener throws"},
        },
        "pi.session.write": {
            "description": "One committed session mutation",
            "parents": {"kind": "any"},
            "startAttributes": {
                "pi.lane.name": {
                    "type": "string",
                    "required": True,
                    "cardinality": "high",
                    "description": "Lane name",
                },
                "pi.operation.id": {
                    "type": "string",
                    "required": False,
                    "cardinality": "high",
                    "description": "Durable operation id when accepted",
                },
                "pi.session.mutation": {
                    "type": "string",
                    "required": True,
                    "values": ["entry", "record", "lane", "fact"],
                    "description": "Session mutation kind",
                },
                "pi.session.item_type": {
                    "type": "string",
                    "required": False,
                    "description": "Entry, record, lane, or fact subtype",
                },
            },
            "endAttributes": {
                "pi.session.seq": {
                    "type": "number",
                    "description": "Committed session sequence when exposed",
                },
            },
            "status": {"default": "ok", "errorWhen": "Storage rejects the mutation"},
        },
    },
}

AGENT_TELEMETRY_SCHEMAS = (AI_TELEMETRY_SCHEMA, HARNESS_TELEMETRY_SCHEMA)

AiSpanName: TypeAlias = str
AiSpanStartAttributes: TypeAlias = dict[str, Any]
AiSpanEndAttributes: TypeAlias = dict[str, Any]
AiSpanAttributes: TypeAlias = dict[str, Any]
AiSpanEventName: TypeAlias = str
AiSpanEventAttributes: TypeAlias = dict[str, Any]
AiTelemetrySpan: TypeAlias = Any
AiSpan: TypeAlias = Any

HarnessSpanName: TypeAlias = str
HarnessSpanStartAttributes: TypeAlias = dict[str, Any]
HarnessSpanEndAttributes: TypeAlias = dict[str, Any]
HarnessSpanAttributes: TypeAlias = dict[str, Any]
HarnessSpanEventName: TypeAlias = str
HarnessSpanEventAttributes: TypeAlias = dict[str, Any]
HarnessTelemetrySpan: TypeAlias = Any
HarnessSpan: TypeAlias = Any


class NoopTelemetrySpan:
    def __init__(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.name = name
        self.attributes = dict(attributes or {})

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        del name, attributes

    def set_status(self, status: Any) -> None:
        del status

    def end(self, attributes: dict[str, Any] | None = None) -> None:
        del attributes


class NoopTelemetryContext:
    async def start_span(self, options: dict[str, Any], callback: Callable[[NoopTelemetrySpan], Any]) -> Any:
        span = NoopTelemetrySpan(options.get("name", ""), options.get("attributes"))
        result = callback(span)
        if isinstance(result, Awaitable):
            return await result
        return result


NOOP_TELEMETRY_CONTEXT = NoopTelemetryContext()


async def start_ai_span(
    telemetry_context: TelemetryContext,
    name: AiSpanName,
    attributes: dict[str, Any],
    callback: Callable[[Any], Any],
) -> Any:
    return await telemetry_context.start_span({"name": name, "attributes": attributes}, callback)


async def start_harness_span(
    telemetry_context: TelemetryContext,
    name: HarnessSpanName,
    attributes: dict[str, Any],
    callback: Callable[[Any], Any],
) -> Any:
    return await telemetry_context.start_span({"name": name, "attributes": attributes}, callback)
