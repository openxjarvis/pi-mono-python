"""Assistant message renderer — mirrors assistant-message.ts"""
from __future__ import annotations

from typing import Any

from .component import Component


class AssistantMessageComponent(Component):
    name = "assistant_message"

    def __init__(
        self,
        message: Any | None = None,
        hide_thinking_block: bool = False,
        hidden_thinking_label: str = "Thinking...",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.message = message
        self.hide_thinking_block = hide_thinking_block
        self.hidden_thinking_label = hidden_thinking_label
        self.is_streaming = False

    def update_content(self, message: Any) -> None:
        self.message = message
        self.invalidate()

    def set_hide_thinking_block(self, hide: bool) -> None:
        self.hide_thinking_block = hide
        self.invalidate()

    def _extract_text(self) -> str:
        content = getattr(self.message, "content", None)
        if content is None and isinstance(self.message, dict):
            content = self.message.get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if item_type == "thinking" and self.hide_thinking_block:
                continue
            if item_type == "text":
                text = item.get("text") if isinstance(item, dict) else getattr(item, "text", "")
                if text:
                    parts.append(str(text))
            elif item_type == "thinking":
                text = item.get("thinking") if isinstance(item, dict) else getattr(item, "thinking", "")
                if text:
                    parts.append(str(text))
        return "".join(parts)

    def _render_body(self, width: int) -> str:
        text = self._extract_text()
        if not text and self.hide_thinking_block:
            return self.hidden_thinking_label
        return f"Assistant: {text}" if text else "Assistant:"
