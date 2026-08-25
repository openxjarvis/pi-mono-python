"""First-time setup — mirrors first-time-setup.ts"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from pi_coding_agent.config import APP_NAME

from .component import Component

TerminalTheme = Literal["dark", "light"]


@dataclass
class FirstTimeSetupResult:
    theme: TerminalTheme
    share_analytics: bool


THEME_OPTIONS = [("dark", "Dark"), ("light", "Light")]
ANALYTICS_OPTIONS = [(True, "Share anonymous usage data"), (False, "Don't share")]


class FirstTimeSetupComponent(Component):
    name = "first_time_setup"

    def __init__(
        self,
        detected_theme: TerminalTheme = "dark",
        on_theme_preview: Callable[[TerminalTheme], None] | None = None,
        on_submit: Callable[[FirstTimeSetupResult], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.detected_theme = detected_theme
        self.on_theme_preview = on_theme_preview
        self.on_submit = on_submit
        self.on_cancel = on_cancel
        self.step: Literal["theme", "analytics"] = "theme"
        self.theme_index = 0 if detected_theme == "dark" else 1
        self.analytics_index = 0

    def confirm_step(self) -> FirstTimeSetupResult | None:
        if self.step == "theme":
            theme = THEME_OPTIONS[self.theme_index][0]
            if self.on_theme_preview:
                self.on_theme_preview(theme)  # type: ignore[arg-type]
            self.step = "analytics"
            self.invalidate()
            return None
        result = FirstTimeSetupResult(
            theme=THEME_OPTIONS[self.theme_index][0],  # type: ignore[arg-type]
            share_analytics=ANALYTICS_OPTIONS[self.analytics_index][0],
        )
        if self.on_submit:
            self.on_submit(result)
        return result

    def cancel(self) -> None:
        if self.on_cancel:
            self.on_cancel()

    def _render_body(self, width: int) -> str:
        lines = [
            f"Welcome to {APP_NAME}, the minimal coding agent.",
        ]
        if self.step == "theme":
            lines.append("Pick a theme.")
            lines.append(f"Detected system appearance: {self.detected_theme}")
            for index, (_value, label) in enumerate(THEME_OPTIONS):
                marker = ">" if index == self.theme_index else " "
                lines.append(f"  {marker} {label}")
        else:
            lines.append("Opt-in to anonymous usage data sharing?")
            for index, (_value, label) in enumerate(ANALYTICS_OPTIONS):
                marker = ">" if index == self.analytics_index else " "
                lines.append(f"  {marker} {label}")
        return "\n".join(lines)
