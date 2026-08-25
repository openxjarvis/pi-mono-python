"""Status indicators — mirrors components/status-indicator.ts"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pi_tui.components.loader import Loader

from ..theme.theme import get_theme
from .countdown_timer import CountdownTimer
from .keybinding_hints import key_text

if TYPE_CHECKING:
    from pi_tui.tui import TUI

StatusIndicatorKind = Literal["working", "retry", "compaction", "branchSummary"]
CompactionStatusReason = Literal["manual", "threshold", "overflow"]


class StatusIndicator(Loader):
    def __init__(
        self,
        kind: StatusIndicatorKind,
        ui: "TUI",
        spinner_color_fn,
        message_color_fn,
        message: str,
        indicator=None,
    ) -> None:
        super().__init__(ui, spinner_color_fn, message_color_fn, message, indicator)
        self.kind = kind

    def dispose(self) -> None:
        if hasattr(self, "stop"):
            self.stop()


class WorkingStatusIndicator(StatusIndicator):
    def __init__(self, ui: "TUI", message: str, indicator=None) -> None:
        theme = get_theme()
        super().__init__(
            "working",
            ui,
            lambda spinner: theme.fg("accent", spinner),
            lambda text: theme.fg("muted", text),
            message,
            indicator,
        )


class RetryStatusIndicator(StatusIndicator):
    def __init__(self, ui: "TUI", attempt: int, max_attempts: int, delay_ms: int) -> None:
        theme = get_theme()

        def retry_message(seconds: int) -> str:
            return f"Retrying ({attempt}/{max_attempts}) in {seconds}s... ({key_text('app.interrupt')} to cancel)"

        super().__init__(
            "retry",
            ui,
            lambda spinner: theme.fg("warning", spinner),
            lambda text: theme.fg("muted", text),
            retry_message(max(1, (delay_ms + 999) // 1000)),
        )
        self.countdown = CountdownTimer(
            delay_ms,
            on_tick=lambda seconds: self.set_message(retry_message(seconds)) if hasattr(self, "set_message") else None,
            on_expire=lambda: None,
        )

    def dispose(self) -> None:
        if getattr(self, "countdown", None):
            if hasattr(self.countdown, "dispose"):
                self.countdown.dispose()
            self.countdown = None
        super().dispose()


class CompactionStatusIndicator(StatusIndicator):
    def __init__(self, ui: "TUI", reason: CompactionStatusReason) -> None:
        theme = get_theme()
        cancel_hint = f"({key_text('app.interrupt')} to cancel)"
        if reason == "manual":
            label = f"Compacting context... {cancel_hint}"
        elif reason == "overflow":
            label = f"Context overflow detected, Auto-compacting... {cancel_hint}"
        else:
            label = f"Auto-compacting... {cancel_hint}"
        super().__init__(
            "compaction",
            ui,
            lambda spinner: theme.fg("accent", spinner),
            lambda text: theme.fg("muted", text),
            label,
        )


class BranchSummaryStatusIndicator(StatusIndicator):
    def __init__(self, ui: "TUI") -> None:
        theme = get_theme()
        super().__init__(
            "branchSummary",
            ui,
            lambda spinner: theme.fg("accent", spinner),
            lambda text: theme.fg("muted", text),
            f"Summarizing branch... ({key_text('app.interrupt')} to cancel)",
        )


class IdleStatus:
    def invalidate(self) -> None:
        return

    def render(self, width: int) -> list[str]:
        empty = " " * max(0, width)
        return [empty, empty]
