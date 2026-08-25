from __future__ import annotations

from pi_agent.harness.session.types import SessionError
from pi_agent.harness.types import Result


class JsonlDecodeError(Exception):
    def __init__(self, kind: str, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.name = "JsonlDecodeError"
        self.kind = kind
        if cause is not None:
            self.__cause__ = cause


def file_result(result: Result, message: str) -> object:
    if not result.get("ok"):
        error = result["error"]
        code = "not_found" if getattr(error, "code", None) == "not_found" else "storage"
        raise SessionError(code, f"{message}: {error}", error)
    return result["value"]


def invalid_file(path: str, line: int, cause: Exception) -> SessionError:
    return SessionError("invalid_entry", f"Invalid JSONL v4 session {path}: line {line} {cause}", cause)
