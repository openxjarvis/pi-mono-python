"""User message renderer — mirrors user-message.ts"""
from __future__ import annotations

from typing import Any

from .component import Component


class UserMessageComponent(Component):
    name = "user_message"

    def __init__(self, message: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.message = message

    def _extract_text(self) -> str:
        if self.message is None:
            return ""
        if isinstance(self.message, str):
            return self.message
        content = getattr(self.message, "content", None)
        if content is None and isinstance(self.message, dict):
            content = self.message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                (item.get("text") if isinstance(item, dict) else getattr(item, "text", "")) or ""
                for item in content
                if (item.get("type") if isinstance(item, dict) else getattr(item, "type", None)) == "text"
            )
        return str(self.message)

    def _render_body(self, width: int) -> str:
        return f"You: {self._extract_text()}"
