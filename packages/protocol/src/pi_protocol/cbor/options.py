from __future__ import annotations

UINT32_BASE = 0x1_0000_0000
MAX_UINT32 = 0xFFFF_FFFF
MAX_CONFIGURED_DEPTH = 512

DEFAULT_MAX_CBOR_BYTE_LENGTH = 16 * 1024 * 1024
DEFAULT_MAX_CBOR_CONTAINER_LENGTH = 1_000_000
DEFAULT_MAX_CBOR_DEPTH = 64


class CborError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.name = "CborError"


class CborOptions(dict):
    pass


class ResolvedCborOptions:
    def __init__(self, max_byte_length: int, max_container_length: int, max_depth: int) -> None:
        self.max_byte_length = max_byte_length
        self.max_container_length = max_container_length
        self.max_depth = max_depth


def _resolve_limit(name: str, value: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > maximum:
        raise ValueError(f"{name} must be an integer between 0 and {maximum}")
    return value


def resolve_options(options: dict[str, int] | None = None) -> ResolvedCborOptions:
    options = options or {}
    return ResolvedCborOptions(
        max_byte_length=_resolve_limit(
            "maxByteLength",
            options.get("maxByteLength", options.get("max_byte_length", DEFAULT_MAX_CBOR_BYTE_LENGTH)),
            MAX_UINT32,
        ),
        max_container_length=_resolve_limit(
            "maxContainerLength",
            options.get("maxContainerLength", options.get("max_container_length", DEFAULT_MAX_CBOR_CONTAINER_LENGTH)),
            MAX_UINT32,
        ),
        max_depth=_resolve_limit(
            "maxDepth",
            options.get("maxDepth", options.get("max_depth", DEFAULT_MAX_CBOR_DEPTH)),
            MAX_CONFIGURED_DEPTH,
        ),
    )
