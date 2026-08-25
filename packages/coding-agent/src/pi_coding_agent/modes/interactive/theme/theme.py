"""
Interactive theme — mirrors packages/coding-agent/src/modes/interactive/theme/theme.ts
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pi_tui.components.editor import EditorTheme
from pi_tui.components.markdown import MarkdownTheme
from pi_tui.components.select_list import SelectListTheme
from pi_coding_agent.utils.text import strip_bom

TerminalTheme = Literal["dark", "light"]
ColorFn = Callable[[str], str]
ColorMode = Literal["truecolor", "256color"]

THEME_COLORS = (
    "accent", "border", "borderAccent", "borderMuted", "success", "error", "warning",
    "muted", "dim", "text", "thinkingText", "searchMatchText", "userMessageText",
    "customMessageText", "customMessageLabel", "toolTitle", "toolOutput",
    "mdHeading", "mdLink", "mdLinkUrl", "mdCode", "mdCodeBlock", "mdCodeBlockBorder",
    "mdQuote", "mdQuoteBorder", "mdHr", "mdListBullet",
    "toolDiffAdded", "toolDiffRemoved", "toolDiffContext",
    "syntaxComment", "syntaxKeyword", "syntaxFunction", "syntaxVariable",
    "syntaxString", "syntaxNumber", "syntaxType", "syntaxOperator", "syntaxPunctuation",
    "thinkingOff", "thinkingMinimal", "thinkingLow", "thinkingMedium",
    "thinkingHigh", "thinkingXhigh", "thinkingMax", "bashMode",
)
THEME_BGS = (
    "selectedBg", "scrollbarThumb", "searchMatchBg", "userMessageBg",
    "customMessageBg", "toolPendingBg", "toolSuccessBg", "toolErrorBg",
)

_ANSI = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "italic": "\x1b[3m",
    "underline": "\x1b[4m",
    "inverse": "\x1b[7m",
    "strikethrough": "\x1b[9m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
}

_NAMED_FG = {
    "accent": "\x1b[36m",
    "border": "\x1b[37m",
    "borderAccent": "\x1b[36m",
    "borderMuted": "\x1b[2m",
    "success": "\x1b[32m",
    "error": "\x1b[31m",
    "warning": "\x1b[33m",
    "muted": "\x1b[2m",
    "dim": "\x1b[2m",
    "text": "\x1b[37m",
    "thinkingText": "\x1b[35m",
    "searchMatchText": "\x1b[37m",
    "userMessageText": "\x1b[37m",
    "customMessageText": "\x1b[37m",
    "customMessageLabel": "\x1b[36m",
    "toolTitle": "\x1b[33m",
    "toolOutput": "\x1b[37m",
    "mdHeading": "\x1b[1m",
    "mdLink": "\x1b[34m",
    "mdLinkUrl": "\x1b[2m",
    "mdCode": "\x1b[33m",
    "mdCodeBlock": "\x1b[33m",
    "mdCodeBlockBorder": "\x1b[2m",
    "mdQuote": "\x1b[2m",
    "mdQuoteBorder": "\x1b[2m",
    "mdHr": "\x1b[2m",
    "mdListBullet": "\x1b[36m",
    "toolDiffAdded": "\x1b[32m",
    "toolDiffRemoved": "\x1b[31m",
    "toolDiffContext": "\x1b[2m",
    "syntaxComment": "\x1b[2m",
    "syntaxKeyword": "\x1b[35m",
    "syntaxFunction": "\x1b[34m",
    "syntaxVariable": "\x1b[37m",
    "syntaxString": "\x1b[32m",
    "syntaxNumber": "\x1b[33m",
    "syntaxType": "\x1b[36m",
    "syntaxOperator": "\x1b[37m",
    "syntaxPunctuation": "\x1b[2m",
    "thinkingOff": "\x1b[2m",
    "thinkingMinimal": "\x1b[34m",
    "thinkingLow": "\x1b[36m",
    "thinkingMedium": "\x1b[33m",
    "thinkingHigh": "\x1b[35m",
    "thinkingXhigh": "\x1b[31m",
    "thinkingMax": "\x1b[31m",
    "bashMode": "\x1b[32m",
}

_NAMED_BG = {
    "selectedBg": "\x1b[44m",
    "scrollbarThumb": "\x1b[47m",
    "searchMatchBg": "\x1b[43m",
    "userMessageBg": "\x1b[44m",
    "customMessageBg": "\x1b[45m",
    "toolPendingBg": "\x1b[43m",
    "toolSuccessBg": "\x1b[42m",
    "toolErrorBg": "\x1b[41m",
}

_COLOR_ALIASES = {
    "accent": "cyan",
    "border": "white",
    "borderAccent": "cyan",
    "borderMuted": "dim",
    "success": "green",
    "error": "red",
    "warning": "yellow",
    "muted": "dim",
    "dim": "dim",
    "text": "white",
    "thinkingText": "magenta",
    "toolTitle": "yellow",
    "toolOutput": "white",
    "toolDiffAdded": "green",
    "toolDiffRemoved": "red",
    "toolDiffContext": "dim",
}

_CUBE = [0, 95, 135, 175, 215, 255]
_GRAY = [8 + i * 10 for i in range(24)]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    cleaned = hex_color.removeprefix("#")
    if len(cleaned) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16)


def _closest(value: int, table: list[int]) -> int:
    return min(range(len(table)), key=lambda i: abs(value - table[i]))


def rgb_to_256(r: int, g: int, b: int) -> int:
    r_i, g_i, b_i = _closest(r, _CUBE), _closest(g, _CUBE), _closest(b, _CUBE)
    cube = 16 + 36 * r_i + 6 * g_i + b_i
    gray = round(0.299 * r + 0.587 * g + 0.114 * b)
    gray_i = _closest(gray, _GRAY)
    if max(r, g, b) - min(r, g, b) < 10:
        return 232 + gray_i
    return cube


def _fg_ansi(value: str | int, mode: ColorMode = "truecolor") -> str:
    if isinstance(value, int):
        return f"\x1b[38;5;{value}m"
    if value.startswith("#"):
        r, g, b = _hex_to_rgb(value)
        if mode == "256color":
            return f"\x1b[38;5;{rgb_to_256(r, g, b)}m"
        return f"\x1b[38;2;{r};{g};{b}m"
    return _NAMED_FG.get(value) or _ANSI.get(value, "")


def _bg_ansi(value: str | int, mode: ColorMode = "truecolor") -> str:
    if isinstance(value, int):
        return f"\x1b[48;5;{value}m"
    if value.startswith("#"):
        r, g, b = _hex_to_rgb(value)
        if mode == "256color":
            return f"\x1b[48;5;{rgb_to_256(r, g, b)}m"
        return f"\x1b[48;2;{r};{g};{b}m"
    return _NAMED_BG.get(value, "")


@dataclass
class Theme:
    name: str = "dark"
    colors: dict[str, str] = field(default_factory=dict)
    backgrounds: dict[str, str] = field(default_factory=dict)
    source_path: str | None = None
    source_info: Any = None
    mode: ColorMode = "truecolor"

    def __post_init__(self) -> None:
        if not self.colors:
            self.colors = dict(_NAMED_FG)
        else:
            merged = dict(_NAMED_FG)
            merged.update(self.colors)
            self.colors = merged
        if not self.backgrounds:
            self.backgrounds = dict(_NAMED_BG)
        self.colors.setdefault("thinkingMax", self.colors.get("thinkingXhigh", _NAMED_FG["thinkingXhigh"]))
        self.colors.setdefault("searchMatchText", self.colors.get("text", _NAMED_FG["text"]))
        self.backgrounds.setdefault("scrollbarThumb", self.backgrounds.get("selectedBg", _NAMED_BG["selectedBg"]))
        self.backgrounds.setdefault("searchMatchBg", self.backgrounds.get("selectedBg", _NAMED_BG["selectedBg"]))

    def fg(self, color: str, text: str) -> str:
        raw = self.colors.get(color) or _COLOR_ALIASES.get(color, color)
        ansi = _fg_ansi(raw, self.mode) if raw.startswith("#") or isinstance(raw, int) else (
            _NAMED_FG.get(color) or _ANSI.get(_COLOR_ALIASES.get(color, raw), "")
        )
        if raw.startswith("#"):
            ansi = _fg_ansi(raw, self.mode)
        if not ansi:
            return text
        return f"{ansi}{text}\x1b[39m"

    def bg(self, color: str, text: str) -> str:
        raw = self.backgrounds.get(color, color)
        ansi = _bg_ansi(raw, self.mode) if (isinstance(raw, int) or str(raw).startswith("#")) else _NAMED_BG.get(color, "")
        if not ansi:
            return text
        return f"{ansi}{text}\x1b[49m"

    def bold(self, text: str) -> str:
        return f"{_ANSI['bold']}{text}{_ANSI['reset']}"

    def italic(self, text: str) -> str:
        return f"{_ANSI['italic']}{text}{_ANSI['reset']}"

    def underline(self, text: str) -> str:
        return f"{_ANSI['underline']}{text}{_ANSI['reset']}"

    def inverse(self, text: str) -> str:
        return f"{_ANSI['inverse']}{text}{_ANSI['reset']}"

    def strikethrough(self, text: str) -> str:
        return f"{_ANSI['strikethrough']}{text}{_ANSI['reset']}"

    def get_fg_ansi(self, color: str) -> str:
        return _NAMED_FG.get(color, "")

    def get_bg_ansi(self, color: str) -> str:
        return _NAMED_BG.get(color, "")


_registered: dict[str, Theme] = {}
_current = Theme(name="dark")
_on_change: Callable[[], None] | None = None


class _ThemeProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(_current, name)


theme = _ThemeProxy()


def get_theme() -> Theme:
    return _current


def get_theme_by_name(name: str) -> Theme:
    if name in _registered:
        return _registered[name]
    if name in ("dark", "light"):
        return Theme(name=name)
    return Theme(name=name)


def get_available_themes() -> list[str]:
    names = ["dark", "light", *sorted(_registered)]
    return list(dict.fromkeys(names))


def get_available_themes_with_paths() -> list[dict[str, str | None]]:
    return [{"name": name, "path": getattr(_registered.get(name), "source_path", None)} for name in get_available_themes()]


def set_theme(next_theme: Theme | str, enable_watcher: bool = False) -> dict[str, Any]:
    global _current
    if isinstance(next_theme, Theme):
        _current = next_theme
        if _on_change:
            _on_change()
        return {"success": True}
    found = _registered.get(next_theme)
    if found is None:
        if next_theme in ("dark", "light"):
            _current = Theme(name=next_theme)
            if _on_change:
                _on_change()
            return {"success": True}
        return {"success": False, "error": f"Unknown theme: {next_theme}"}
    _current = found
    if _on_change:
        _on_change()
    return {"success": True}


def set_theme_instance(theme_instance: Theme) -> None:
    global _current
    _current = theme_instance


def on_theme_change(callback: Callable[[], None]) -> None:
    global _on_change
    _on_change = callback


def init_theme(name: str | None = None, enable_watcher: bool = False) -> Theme:
    set_theme(name or get_default_theme())
    return _current


def set_registered_themes(themes: list[Theme]) -> None:
    _registered.clear()
    for item in themes:
        if item.name:
            _registered[item.name] = item


def load_theme_from_path(path: str, mode: ColorMode = "truecolor") -> Theme:
    with open(path, encoding="utf-8") as f:
        data = json.loads(strip_bom(f.read()) or "{}")
    name = data.get("name") or os.path.splitext(os.path.basename(path))[0]
    colors = data.get("colors") if isinstance(data.get("colors"), dict) else {}
    return Theme(name=str(name), colors=dict(colors), source_path=path, mode=mode)


def parse_auto_theme_setting(value: str | None) -> str | None:
    if not value:
        return None
    if value in ("auto", "dark", "light"):
        return value
    return value


def resolve_theme_setting(setting: str | None, terminal_theme: str = "dark") -> str:
    if not setting or setting == "auto":
        return terminal_theme
    return setting


def detect_terminal_background_from_env() -> dict[str, str]:
    colorfgbg = os.environ.get("COLORFGBG", "")
    if colorfgbg.endswith(";0") or colorfgbg.endswith(";8"):
        return {"theme": "dark"}
    if ";15" in colorfgbg or colorfgbg.endswith(";7"):
        return {"theme": "light"}
    return {"theme": "dark"}


def detect_terminal_theme_for_auto() -> str:
    return detect_terminal_background_from_env()["theme"]


def get_default_theme() -> str:
    return detect_terminal_background_from_env()["theme"]


def get_theme_for_rgb_color(rgb: dict[str, int] | tuple[int, int, int]) -> TerminalTheme:
    if isinstance(rgb, dict):
        r, g, b = int(rgb["r"]), int(rgb["g"]), int(rgb["b"])
    else:
        r, g, b = rgb
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "light" if luminance > 0.5 else "dark"


def is_light_theme(theme_name: str | None = None) -> bool:
    return (theme_name or _current.name) == "light"


def get_resolved_theme_colors(theme_name: str | None = None) -> dict[str, str]:
    selected = get_theme_by_name(theme_name) if theme_name else _current
    return dict(selected.colors)


def get_theme_export_colors(theme_name: str | None = None) -> dict[str, str]:
    colors = get_resolved_theme_colors(theme_name)
    return {
        "pageBg": colors.get("text", ""),
        "cardBg": colors.get("border", ""),
        "infoBg": colors.get("muted", ""),
    }


def highlight_code(code: str, lang: str | None = None) -> list[str]:
    try:
        from pi_coding_agent.utils.syntax_highlight import highlight

        result = highlight(code, lang)
        if isinstance(result, list):
            return result
        return str(result).splitlines() or [""]
    except Exception:
        return code.splitlines() or [""]


def get_language_from_path(file_path: str) -> str | None:
    ext = os.path.splitext(file_path)[1].lower()
    mapping = {
        ".py": "python", ".ts": "typescript", ".js": "javascript", ".json": "json",
        ".md": "markdown", ".rs": "rust", ".go": "go", ".sh": "bash",
    }
    return mapping.get(ext)


def get_markdown_theme() -> MarkdownTheme:
    current = get_theme()
    return MarkdownTheme(
        heading=lambda s: current.fg("mdHeading", s),
        bold=current.bold,
        code=lambda s: current.fg("mdCode", s),
        code_block=lambda s: current.fg("mdCodeBlock", s),
        code_block_border=lambda s: current.fg("mdCodeBlockBorder", s),
        list_bullet=lambda s: current.fg("mdListBullet", s),
    )


def get_select_list_theme() -> SelectListTheme:
    current = get_theme()
    return SelectListTheme(
        selected_text=lambda s: current.bg("selectedBg", current.fg("text", s)),
        description=lambda s: current.fg("muted", s),
        scroll_info=lambda s: current.fg("dim", s),
        no_match=lambda s: current.fg("warning", s),
    )


def get_editor_theme() -> EditorTheme:
    current = get_theme()
    return EditorTheme(
        border_color=lambda s: current.fg("border", s),
        select_list=get_select_list_theme(),
    )


def get_settings_list_theme() -> dict[str, Any]:
    return {"select_list": get_select_list_theme()}


def stop_theme_watcher() -> None:
    return None
