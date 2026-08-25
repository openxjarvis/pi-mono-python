"""Tool execution renderer — mirrors tool-execution.ts"""
from __future__ import annotations

from typing import Any

from .component import Component


class ToolExecutionComponent(Component):
    name = "tool_execution"

    def __init__(
        self,
        tool_name: str = "tool",
        tool_call_id: str = "",
        args: Any | None = None,
        result: Any | None = None,
        is_error: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.args = args
        self.result = result
        self.is_error = is_error
        self.pending = result is None

    def set_result(self, result: Any, is_error: bool = False) -> None:
        self.result = result
        self.is_error = is_error
        self.pending = False
        self.invalidate()

    def _render_body(self, width: int) -> str:
        status = "pending" if self.pending else ("error" if self.is_error else "ok")
        line = f"Tool {self.tool_name} ({status})"
        if self.result is not None:
            content = getattr(self.result, "content", None)
            if content is None and isinstance(self.result, dict):
                content = self.result.get("content")
            snippet = ""
            if isinstance(content, list):
                snippet = " ".join(
                    str(item.get("text") if isinstance(item, dict) else getattr(item, "text", "") or "")
                    for item in content
                ).strip()
            elif isinstance(self.result, str):
                snippet = self.result
            if snippet:
                if len(snippet) > 160:
                    snippet = snippet[:157] + "..."
                line = f"{line} - {snippet}"
        return line
