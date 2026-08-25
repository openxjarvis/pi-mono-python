"""
Terminal color-scheme parsing — mirrors packages/tui/src/terminal-colors.ts
"""
from __future__ import annotations

import re
from typing import Literal, TypedDict

TerminalColorScheme = Literal["dark", "light"]


class RgbColor(TypedDict):
    r: int
    g: int
    b: int


def _hex_to_rgb(hex_value: str) -> RgbColor:
    normalized = hex_value[1:] if hex_value.startswith("#") else hex_value
    return {
        "r": int(normalized[0:2], 16),
        "g": int(normalized[2:4], 16),
        "b": int(normalized[4:6], 16),
    }


def _parse_osc_hex_channel(channel: str) -> int | None:
    if not re.fullmatch(r"[0-9a-fA-F]+", channel):
        return None
    maximum = 16 ** len(channel) - 1
    if maximum <= 0:
        return None
    return round((int(channel, 16) / maximum) * 255)


_OSC11_BACKGROUND_COLOR_RESPONSE_PATTERN = re.compile(
    r"^\x1b\]11;([^\x07\x1b]*)(?:\x07|\x1b\\)$",
    re.IGNORECASE,
)
_COLOR_SCHEME_REPORT_PATTERN = re.compile(r"^(?:\x1b\[\?997;(1|2)n)+$")


def is_osc11_background_color_response(data: str) -> bool:
    return bool(_OSC11_BACKGROUND_COLOR_RESPONSE_PATTERN.match(data))


def parse_osc11_background_color(data: str) -> RgbColor | None:
    match = _OSC11_BACKGROUND_COLOR_RESPONSE_PATTERN.match(data)
    if not match:
        return None

    value = match.group(1).strip()
    if value.startswith("#"):
        hex_value = value[1:]
        if re.fullmatch(r"[0-9a-fA-F]{6}", hex_value):
            return _hex_to_rgb(value)
        if re.fullmatch(r"[0-9a-fA-F]{12}", hex_value):
            r = _parse_osc_hex_channel(hex_value[0:4])
            g = _parse_osc_hex_channel(hex_value[4:8])
            b = _parse_osc_hex_channel(hex_value[8:12])
            if r is not None and g is not None and b is not None:
                return {"r": r, "g": g, "b": b}
        return None

    rgb_value = re.sub(r"^rgba?:", "", value, flags=re.IGNORECASE)
    parts = rgb_value.split("/")
    if len(parts) < 3:
        return None
    r = _parse_osc_hex_channel(parts[0])
    g = _parse_osc_hex_channel(parts[1])
    b = _parse_osc_hex_channel(parts[2])
    if r is None or g is None or b is None:
        return None
    return {"r": r, "g": g, "b": b}


def parse_terminal_color_scheme_report(data: str) -> TerminalColorScheme | None:
    """Parse one or more batched CSI ? 997 ; Ps n reports as a single scheme."""
    match = _COLOR_SCHEME_REPORT_PATTERN.match(data)
    if not match:
        return None
    return "light" if match.group(1) == "2" else "dark"
