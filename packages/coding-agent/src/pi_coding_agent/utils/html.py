"""
HTML entity decoding — mirrors packages/coding-agent/src/utils/html.ts
"""
from __future__ import annotations

from dataclasses import dataclass

_NAMED = {
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "apos": "'",
}


@dataclass
class DecodedHtmlEntity:
    text: str
    length: int


def decode_code_point(code_point: int) -> str | None:
    if not isinstance(code_point, int) or code_point < 0 or code_point > 0x10FFFF:
        return None
    try:
        return chr(code_point)
    except ValueError:
        return None


def decode_html_entity(entity: str) -> str | None:
    if entity in _NAMED:
        return _NAMED[entity]
    if entity.startswith("#x") or entity.startswith("#X"):
        try:
            return decode_code_point(int(entity[2:], 16))
        except ValueError:
            return None
    if entity.startswith("#"):
        try:
            return decode_code_point(int(entity[1:], 10))
        except ValueError:
            return None
    return None


def decode_html_entity_at(html: str, index: int) -> DecodedHtmlEntity | None:
    semicolon = html.find(";", index + 1)
    if semicolon == -1 or semicolon - index > 16:
        return None
    entity = html[index + 1 : semicolon]
    decoded = decode_html_entity(entity)
    if decoded is None:
        return None
    return DecodedHtmlEntity(text=decoded, length=semicolon - index + 1)
