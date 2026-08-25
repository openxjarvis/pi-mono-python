"""Settings selector — mirrors settings-selector.ts"""
from __future__ import annotations

from typing import Any, Callable

from .component import Component

SETTINGS_SECTIONS = (
    ("Appearance", ("theme", "tuiMode", "showHardwareCursor", "hideThinkingBlock", "collapseChangelog")),
    ("Session", ("steeringMode", "followUpMode", "thinkingLevel", "autoCompact", "doubleEscapeAction", "treeFilterMode")),
    ("Images", ("showImages", "autoResizeImages", "blockImages")),
    ("Editor", ("editorPaddingX", "outputPad", "autocompleteMaxVisible", "quietStartup")),
    ("Transport", ("transport", "httpIdleTimeoutMs", "showTerminalProgress", "clearOnShrink")),
)


class SettingsSelectorComponent(Component):
    name = "settings_selector"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        on_select: Callable[[str, Any], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        callbacks: dict[str, Callable] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = dict(config or {})
        self.on_select = on_select
        self.on_cancel = on_cancel
        self.callbacks = callbacks or {}
        self.items = self._build_items()
        self.selected_index = 0
        self.section_index = 0

    def _get(self, camel: str, snake: str, default: Any) -> Any:
        if camel in self.config:
            return self.config[camel]
        return self.config.get(snake, default)

    def _build_items(self) -> list[tuple[str, Any]]:
        return [
            ("theme", self._get("currentTheme", "theme", "dark")),
            ("tuiMode", self._get("tuiMode", "tui_mode", "regular")),
            ("showHardwareCursor", self._get("showHardwareCursor", "show_hardware_cursor", False)),
            ("hideThinkingBlock", self._get("hideThinkingBlock", "hide_thinking_block", False)),
            ("collapseChangelog", self._get("collapseChangelog", "collapse_changelog", False)),
            ("steeringMode", self._get("steeringMode", "steering_mode", "one-at-a-time")),
            ("followUpMode", self._get("followUpMode", "follow_up_mode", "one-at-a-time")),
            ("thinkingLevel", self._get("thinkingLevel", "thinking_level", "off")),
            ("autoCompact", self._get("autoCompact", "auto_compact", True)),
            ("doubleEscapeAction", self._get("doubleEscapeAction", "double_escape_action", "tree")),
            ("treeFilterMode", self._get("treeFilterMode", "tree_filter_mode", "default")),
            ("showImages", self._get("showImages", "show_images", True)),
            ("autoResizeImages", self._get("autoResizeImages", "auto_resize_images", True)),
            ("blockImages", self._get("blockImages", "block_images", False)),
            ("editorPaddingX", self._get("editorPaddingX", "editor_padding_x", 1)),
            ("outputPad", self._get("outputPad", "output_pad", 1)),
            ("autocompleteMaxVisible", self._get("autocompleteMaxVisible", "autocomplete_max_visible", 5)),
            ("quietStartup", self._get("quietStartup", "quiet_startup", False)),
            ("transport", self._get("transport", "transport", "auto")),
            ("httpIdleTimeoutMs", self._get("httpIdleTimeoutMs", "http_idle_timeout_ms", 0)),
            ("showTerminalProgress", self._get("showTerminalProgress", "show_terminal_progress", True)),
            ("clearOnShrink", self._get("clearOnShrink", "clear_on_shrink", False)),
        ]

    def refresh(self) -> None:
        current_key = self.items[self.selected_index][0] if self.items else None
        self.items = self._build_items()
        if current_key:
            for index, (key, _) in enumerate(self.items):
                if key == current_key:
                    self.selected_index = index
                    break
        self.invalidate()

    def set_value(self, key: str, value: Any) -> None:
        self.config[key] = value
        callback = self.callbacks.get(f"on_{_snake(key)}_change") or self.callbacks.get(key)
        if callback:
            callback(value)
        if self.on_select:
            self.on_select(key, value)
        self.refresh()

    def select_current(self) -> tuple[str, Any]:
        key, value = self.items[self.selected_index]
        if self.on_select:
            self.on_select(key, value)
        return key, value

    def move(self, delta: int) -> None:
        if not self.items:
            return
        self.selected_index = max(0, min(len(self.items) - 1, self.selected_index + delta))
        self.invalidate()

    def cancel(self) -> None:
        if self.on_cancel:
            self.on_cancel()

    def _render_body(self, width: int) -> str:
        lines = ["Settings"]
        grouped = {key: value for key, value in self.items}
        offset = 0
        for title, keys in SETTINGS_SECTIONS:
            lines.append(f"  {title}")
            for key in keys:
                if key not in grouped:
                    continue
                marker = ">" if offset == self.selected_index else " "
                lines.append(f"    {marker} {key}: {grouped[key]}")
                offset += 1
        return "\n".join(lines)


def _snake(value: str) -> str:
    out: list[str] = []
    for ch in value:
        if ch.isupper():
            out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)
