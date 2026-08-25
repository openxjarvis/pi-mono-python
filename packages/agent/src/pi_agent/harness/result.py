"""Result helpers and tagged errors — mirrors harness/result.ts."""
from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

TValue = TypeVar("TValue")
TError = TypeVar("TError")

Result = dict[str, Any]


def ok(value: TValue) -> Result:
    return {"ok": True, "value": value}


def err(error: TError) -> Result:
    return {"ok": False, "error": error}


Ok = ok
Err = err


def get_or_throw(result: Result) -> Any:
    if not result.get("ok"):
        raise result.get("error") if isinstance(result.get("error"), BaseException) else Exception(result.get("error"))
    return result.get("value")


def get_or_undefined(result: Result) -> Any:
    return result.get("value") if result.get("ok") else None


def to_error(error: object) -> Exception:
    if isinstance(error, Exception):
        return error
    if isinstance(error, str):
        return Exception(error)
    return Exception(str(error))


class _ResultNs:
    @staticmethod
    def ok(value: TValue) -> Result:
        return {"ok": True, "value": value}

    @staticmethod
    def err(error: TError) -> Result:
        return {"ok": False, "error": error}

    @staticmethod
    def is_ok(result: Result) -> bool:
        return bool(result.get("ok"))

    @staticmethod
    def is_err(result: Result) -> bool:
        return not bool(result.get("ok"))


ResultNs = _ResultNs()


class TaggedErrorValue(Exception):
    _tag: str

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key != "_tag":
                payload[key] = value
        return {"_tag": self._tag, "message": str(self), **payload}


def tagged_error(tag: str) -> type[TaggedErrorValue]:
    """Create a tagged error class. Mirrors TaggedError(tag) in TypeScript."""

    class TaggedErrorClass(TaggedErrorValue):
        _tag = tag

        def __init__(self, props: dict[str, Any]):
            super().__init__(props["message"])
            self.name = tag
            for key, value in props.items():
                setattr(self, key, value)

        @classmethod
        def is_(cls, value: object) -> bool:
            return isinstance(value, TaggedErrorClass)

    TaggedErrorClass.__name__ = tag
    TaggedErrorClass.__qualname__ = tag
    return TaggedErrorClass


def match_error(error: TaggedErrorValue, matchers: dict[str, Callable[[Any], TValue]]) -> TValue:
    return matchers[error._tag](error)
