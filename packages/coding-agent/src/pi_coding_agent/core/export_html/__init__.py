"""
HTML export — mirrors packages/coding-agent/src/core/export-html/index.ts

Exports session conversations to standalone HTML files.
"""
from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from typing import Any

# Default theme colors (dark theme, matching pi-tui defaults)
DEFAULT_THEME_COLORS = {
    "userMessageBg": "#343541",
    "assistantMessageBg": "#444654",
    "text": "#ececf1",
    "textMuted": "#8e8ea0",
    "accent": "#19c37d",
    "error": "#ef4146",
    "warning": "#f0c419",
    "border": "#3e3f4b",
    "toolBg": "#2a2b36",
}


@dataclass
class ExportOptions:
    output_path: str | None = None
    theme_name: str | None = None


def _parse_color(color: str) -> tuple[int, int, int] | None:
    """Parse hex or rgb() color to (r, g, b) tuple."""
    color = color.strip()
    if color.startswith("#") and len(color) == 7:
        try:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            return r, g, b
        except ValueError:
            return None
    import re
    m = re.match(r"^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$", color)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _get_luminance(r: int, g: int, b: int) -> float:
    """Calculate relative luminance."""
    def to_linear(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4
    return 0.2126 * to_linear(r) + 0.7152 * to_linear(g) + 0.0722 * to_linear(b)


def _adjust_brightness(color: str, factor: float) -> str:
    """Lighten or darken a color."""
    parsed = _parse_color(color)
    if not parsed:
        return color
    r, g, b = parsed
    r2 = min(255, max(0, round(r * factor)))
    g2 = min(255, max(0, round(g * factor)))
    b2 = min(255, max(0, round(b * factor)))
    return f"rgb({r2}, {g2}, {b2})"


def _derive_export_colors(base_color: str) -> dict[str, str]:
    """Derive page/card/info background colors from a base color."""
    parsed = _parse_color(base_color)
    if not parsed:
        return {
            "pageBg": "rgb(24, 24, 30)",
            "cardBg": "rgb(30, 30, 36)",
            "infoBg": "rgb(60, 55, 40)",
        }
    r, g, b = parsed
    luminance = _get_luminance(r, g, b)
    is_light = luminance > 0.5
    if is_light:
        return {
            "pageBg": _adjust_brightness(base_color, 0.96),
            "cardBg": base_color,
            "infoBg": f"rgb({min(255, r + 10)}, {min(255, g + 5)}, {max(0, b - 20)})",
        }
    return {
        "pageBg": _adjust_brightness(base_color, 0.7),
        "cardBg": _adjust_brightness(base_color, 0.85),
        "infoBg": f"rgb({min(255, r + 20)}, {min(255, g + 15)}, {b})",
    }


def _get_theme_colors(theme_name: str | None = None) -> dict[str, str]:
    """Get theme colors. Uses defaults for now; can be extended for custom themes."""
    return dict(DEFAULT_THEME_COLORS)


def _generate_css_vars(theme_name: str | None = None) -> str:
    """Generate CSS custom property declarations from theme colors."""
    colors = _get_theme_colors(theme_name)
    lines = [f"--{k}: {v};" for k, v in colors.items()]
    user_msg_bg = colors.get("userMessageBg", "#343541")
    derived = _derive_export_colors(user_msg_bg)
    lines.append(f"--exportPageBg: {derived['pageBg']};")
    lines.append(f"--exportCardBg: {derived['cardBg']};")
    lines.append(f"--exportInfoBg: {derived['infoBg']};")
    return "\n        ".join(lines)


def _render_message_html(entry: dict[str, Any]) -> str:
    """Render a single session entry to HTML."""
    entry_type = entry.get("type", "")

    if entry_type == "message":
        msg = entry.get("message", {})
        role = msg.get("role", "unknown")
        content = msg.get("content", [])

        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = html.escape(block.get("text", ""))
                    parts.append(f'<pre class="text-block">{text}</pre>')
                elif block.get("type") == "tool_call":
                    name = html.escape(block.get("name", ""))
                    args_str = html.escape(json.dumps(block.get("input", {}), indent=2))
                    parts.append(
                        f'<div class="tool-call"><code>{name}</code><pre>{args_str}</pre></div>'
                    )
                elif block.get("type") == "tool_result":
                    result_content = block.get("content", [])
                    result_parts = []
                    for rc in result_content:
                        if isinstance(rc, dict) and rc.get("type") == "text":
                            result_parts.append(html.escape(rc.get("text", "")))
                    is_error = block.get("is_error", False)
                    css_class = "tool-result error" if is_error else "tool-result"
                    parts.append(
                        f'<div class="{css_class}"><pre>{"".join(result_parts)}</pre></div>'
                    )
            elif isinstance(block, str):
                parts.append(f'<pre class="text-block">{html.escape(block)}</pre>')

        css_class = f"message message-{role}"
        content_html = "\n".join(parts)
        role_label = html.escape(role.capitalize())
        return f'<div class="{css_class}"><div class="message-role">{role_label}</div><div class="message-content">{content_html}</div></div>'

    elif entry_type == "compaction":
        summary = html.escape(entry.get("summary", ""))
        return f'<div class="compaction"><div class="compaction-label">Context Compacted</div><pre>{summary}</pre></div>'

    return ""


def _build_html(entries: list[dict[str, Any]], session_id: str, theme_name: str | None) -> str:
    """Build complete HTML document from session entries."""
    css_vars = _generate_css_vars(theme_name)
    messages_html = "\n".join(
        r for e in entries if (r := _render_message_html(e))
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Session {html.escape(session_id)}</title>
<style>
  :root {{
        {css_vars}
  }}
  body {{
    background: var(--exportPageBg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    margin: 0;
    padding: 24px;
  }}
  .session-container {{ max-width: 900px; margin: 0 auto; }}
  .session-header {{ margin-bottom: 24px; color: var(--textMuted); font-size: 0.9em; }}
  .message {{ background: var(--exportCardBg); border-radius: 8px; padding: 16px; margin-bottom: 12px; }}
  .message-user {{ border-left: 3px solid var(--accent); }}
  .message-assistant {{ border-left: 3px solid var(--textMuted); }}
  .message-role {{ font-weight: bold; font-size: 0.85em; color: var(--textMuted); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .message-content pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-family: inherit; }}
  .tool-call {{ background: var(--toolBg); border-radius: 4px; padding: 8px; margin: 8px 0; }}
  .tool-call code {{ color: var(--accent); font-weight: bold; }}
  .tool-result {{ background: var(--toolBg); border-radius: 4px; padding: 8px; margin: 8px 0; border-left: 2px solid var(--textMuted); }}
  .tool-result.error {{ border-left-color: var(--error); }}
  .tool-result pre {{ margin: 0; font-size: 0.9em; }}
  .compaction {{ background: var(--exportInfoBg); border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
  .compaction-label {{ font-weight: bold; color: var(--warning); margin-bottom: 8px; }}
  .compaction pre {{ margin: 0; font-size: 0.9em; white-space: pre-wrap; }}
</style>
</head>
<body>
<div class="session-container">
  <div class="session-header">Session ID: {html.escape(session_id)}</div>
  {messages_html}
</div>
</body>
</html>"""


async def export_session_to_html(
    session_manager: Any,
    session_id: str | None = None,
    options: ExportOptions | None = None,
) -> str:
    """
    Export a session to an HTML file.
    Mirrors exportSessionToHtml() in TypeScript.

    Returns the output file path.
    """
    opts = options or ExportOptions()

    # Load entries from session manager
    entries: list[dict[str, Any]] = []
    sid = session_id or (session_manager.get_session_id() if hasattr(session_manager, "get_session_id") else "unknown")

    if hasattr(session_manager, "get_entries"):
        raw_entries = session_manager.get_entries()
        for e in raw_entries:
            if hasattr(e, "__dict__"):
                entries.append(e.__dict__)
            elif isinstance(e, dict):
                entries.append(e)

    html_content = _build_html(entries, sid, opts.theme_name)

    output_path = opts.output_path
    if output_path is None:
        output_path = os.path.join(os.getcwd(), f"session-{sid}.html")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


async def export_from_file(
    session_file_path: str,
    output_path: str | None = None,
    theme_name: str | None = None,
) -> str:
    """
    Export a session from a JSONL file to HTML.
    Mirrors exportFromFile() in TypeScript.
    """
    entries: list[dict[str, Any]] = []
    session_id = "unknown"

    if os.path.exists(session_file_path):
        with open(session_file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("type") == "session":
                        session_id = obj.get("id", session_id)
                    else:
                        entries.append(obj)
                except json.JSONDecodeError:
                    pass

    html_content = _build_html(entries, session_id, theme_name)

    if output_path is None:
        base = os.path.splitext(session_file_path)[0]
        output_path = f"{base}.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
