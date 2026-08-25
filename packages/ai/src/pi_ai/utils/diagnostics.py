"""
Assistant-message diagnostics helpers.
Mirrors packages/ai/src/utils/diagnostics.ts
"""
from __future__ import annotations

import time
from typing import Any, TypedDict


class DiagnosticErrorInfo(TypedDict, total=False):
    name: str
    message: str
    stack: str
    code: str | int


class AssistantMessageDiagnostic(TypedDict, total=False):
    type: str
    timestamp: int
    error: DiagnosticErrorInfo
    details: dict[str, Any]


def format_thrown_value(value: object) -> str:
    if isinstance(value, BaseException):
        return str(value) or type(value).__name__
    if isinstance(value, str):
        return value
    return str(value)


def extract_diagnostic_error(error: object) -> DiagnosticErrorInfo:
    if not isinstance(error, BaseException):
        return {"name": "ThrownValue", "message": format_thrown_value(error)}
    info: DiagnosticErrorInfo = {
        "name": type(error).__name__,
        "message": str(error) or type(error).__name__,
    }
    code = getattr(error, "code", None)
    if isinstance(code, (str, int)):
        info["code"] = code
    return info


def create_assistant_message_diagnostic(
    type: str,
    error: object,
    details: dict[str, Any] | None = None,
) -> AssistantMessageDiagnostic:
    diagnostic: AssistantMessageDiagnostic = {
        "type": type,
        "timestamp": int(time.time() * 1000),
        "error": extract_diagnostic_error(error),
    }
    if details is not None:
        diagnostic["details"] = details
    return diagnostic


def append_assistant_message_diagnostic(message: Any, diagnostic: AssistantMessageDiagnostic) -> None:
    current = list(getattr(message, "diagnostics", None) or [])
    current.append(diagnostic)
    message.diagnostics = current
