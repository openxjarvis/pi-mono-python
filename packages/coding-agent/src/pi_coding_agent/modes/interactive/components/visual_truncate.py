"""Visual line truncation. Mirrors visual-truncate.ts"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VisualTruncateResult:
    text: str
    truncated: bool
    omitted_lines: int = 0


def truncate_to_visual_lines(text: str, max_lines: int) -> VisualTruncateResult:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return VisualTruncateResult(text=text, truncated=False)
    kept = lines[:max_lines]
    return VisualTruncateResult(text="\n".join(kept), truncated=True, omitted_lines=len(lines) - max_lines)

from pi_tui.components.text import Text
class VisualTruncateComponent(Text):
    def __init__(self, text: str = '', max_lines: int = 20):
        result = truncate_to_visual_lines(text, max_lines)
        super().__init__(result.text)
