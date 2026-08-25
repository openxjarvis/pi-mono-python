"""Login dialog — mirrors login-dialog.ts"""
from __future__ import annotations

from typing import Any, Callable

from .component import Component


class LoginDialogComponent(Component):
    name = "login_dialog"

    def __init__(
        self,
        provider_id: str,
        on_complete: Callable[[bool, str | None], None] | None = None,
        provider_name: str | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.provider_id = provider_id
        self.provider_name = provider_name or provider_id
        self.title = title or f"Login to {self.provider_name}"
        self.on_complete = on_complete
        self.input_value = ""
        self.cancelled = False
        self.message = "Enter API key or complete OAuth in the browser."

    def submit(self, value: str | None = None) -> None:
        self.input_value = value if value is not None else self.input_value
        if self.on_complete:
            self.on_complete(True, self.input_value or None)

    def cancel(self) -> None:
        self.cancelled = True
        if self.on_complete:
            self.on_complete(False, "cancelled")

    def _render_body(self, width: int) -> str:
        lines = [
            self.title,
            self.message,
            f"> {self.input_value}",
        ]
        return "\n".join(lines)
