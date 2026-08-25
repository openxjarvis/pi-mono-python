"""Trust selector — mirrors trust-selector.ts"""
from __future__ import annotations

from typing import Any, Callable

from pi_coding_agent.core.trust_manager import (
    ProjectTrustOption,
    ProjectTrustStoreEntry,
    get_project_trust_options,
)

from .component import Component


class TrustSelectorComponent(Component):
    name = "trust_selector"

    def __init__(
        self,
        cwd: str,
        saved_decision: ProjectTrustStoreEntry | None = None,
        project_trusted: bool = False,
        on_select: Callable[[ProjectTrustOption], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cwd = cwd
        self.saved_decision = saved_decision
        self.project_trusted = project_trusted
        self.on_select = on_select
        self.on_cancel = on_cancel
        self.trust_options = get_project_trust_options(cwd)
        self.selected_index = 0
        if saved_decision is not None:
            for index, option in enumerate(self.trust_options):
                if option.saved_path == saved_decision.path and option.trusted == saved_decision.decision:
                    self.selected_index = index
                    break

    def select_current(self) -> ProjectTrustOption:
        option = self.trust_options[self.selected_index]
        if self.on_select:
            self.on_select(option)
        return option

    def cancel(self) -> None:
        if self.on_cancel:
            self.on_cancel()

    def _format_decision(self) -> str:
        if self.saved_decision is None:
            return "none"
        label = "trusted" if self.saved_decision.decision else "untrusted"
        return f"{label} ({self.saved_decision.path})"

    def _render_body(self, width: int) -> str:
        lines = [
            "Project trust",
            self.cwd,
            f"Saved decision: {self._format_decision()}",
            f"Current session: {'trusted' if self.project_trusted else 'untrusted'}",
        ]
        for index, option in enumerate(self.trust_options):
            marker = ">" if index == self.selected_index else " "
            lines.append(f"  {marker} {option.label}")
        return "\n".join(lines)
