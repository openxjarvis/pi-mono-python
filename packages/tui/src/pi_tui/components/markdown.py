"""Markdown component — mirrors components/markdown.ts"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..terminal_image import is_image_line
from ..utils import apply_background_to_line, visible_width, wrap_text_with_ansi


@dataclass
class DefaultTextStyle:
    """Default text styling applied to all markdown text unless overridden."""
    color: Callable[[str], str] | None = None
    bg_color: Callable[[str], str] | None = None
    bold: bool = False
    italic: bool = False
    strikethrough: bool = False
    underline: bool = False


@dataclass
class MarkdownTheme:
    heading: Callable[[str], str] = field(default=lambda x: x)
    link: Callable[[str], str] = field(default=lambda x: x)
    link_url: Callable[[str], str] = field(default=lambda x: x)
    code: Callable[[str], str] = field(default=lambda x: x)
    code_block: Callable[[str], str] = field(default=lambda x: x)
    code_block_border: Callable[[str], str] = field(default=lambda x: x)
    quote: Callable[[str], str] = field(default=lambda x: x)
    quote_border: Callable[[str], str] = field(default=lambda x: x)
    hr: Callable[[str], str] = field(default=lambda x: x)
    list_bullet: Callable[[str], str] = field(default=lambda x: x)
    bold: Callable[[str], str] = field(default=lambda x: f"\x1b[1m{x}\x1b[22m")
    italic: Callable[[str], str] = field(default=lambda x: f"\x1b[3m{x}\x1b[23m")
    strikethrough: Callable[[str], str] = field(default=lambda x: f"\x1b[9m{x}\x1b[29m")
    underline: Callable[[str], str] = field(default=lambda x: f"\x1b[4m{x}\x1b[24m")
    highlight_code: Callable[[str, str | None], list[str]] | None = None
    code_block_indent: str = "  "


class Markdown:
    """
    Renders markdown text to ANSI-styled terminal output.
    Uses mistune for parsing. Mirrors Markdown in components/markdown.ts.
    """

    def __init__(
        self,
        text: str,
        padding_x: int,
        padding_y: int,
        theme: MarkdownTheme,
        default_text_style: DefaultTextStyle | None = None,
    ) -> None:
        self._text = text
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._theme = theme
        self._default_text_style = default_text_style
        self._default_style_prefix: str | None = None

        self._cached_text: str | None = None
        self._cached_width: int | None = None
        self._cached_lines: list[str] | None = None

        self._md = self._create_parser()

    def _create_parser(self) -> Any:
        try:
            import mistune
            return mistune.create_markdown(renderer=None)  # AST renderer
        except ImportError:
            return None

    def set_text(self, text: str) -> None:
        self._text = text
        self.invalidate()

    def invalidate(self) -> None:
        self._cached_text = None
        self._cached_width = None
        self._cached_lines = None

    def handle_input(self, _data: str) -> None:
        pass

    def render(self, width: int) -> list[str]:
        if self._cached_lines is not None and self._cached_text == self._text and self._cached_width == width:
            return self._cached_lines

        content_width = max(1, width - self._padding_x * 2)

        if not self._text or not self._text.strip():
            result: list[str] = []
            self._cached_text = self._text
            self._cached_width = width
            self._cached_lines = result
            return result

        normalized = self._text.replace("\t", "   ")
        rendered_lines = self._render_markdown(normalized, content_width)

        wrapped_lines: list[str] = []
        for line in rendered_lines:
            if is_image_line(line):
                wrapped_lines.append(line)
            else:
                wrapped_lines.extend(wrap_text_with_ansi(line, content_width))

        left_margin = " " * self._padding_x
        right_margin = " " * self._padding_x
        bg_fn = self._default_text_style.bg_color if self._default_text_style else None
        content_lines: list[str] = []

        for line in wrapped_lines:
            if is_image_line(line):
                content_lines.append(line)
                continue
            line_with_margins = left_margin + line + right_margin
            if bg_fn:
                content_lines.append(apply_background_to_line(line_with_margins, width, bg_fn))
            else:
                vis = visible_width(line_with_margins)
                content_lines.append(line_with_margins + " " * max(0, width - vis))

        empty_line = " " * width
        empty_lines_list: list[str] = []
        for _ in range(self._padding_y):
            ln = apply_background_to_line(empty_line, width, bg_fn) if bg_fn else empty_line
            empty_lines_list.append(ln)

        result = [*empty_lines_list, *content_lines, *empty_lines_list]

        self._cached_text = self._text
        self._cached_width = width
        self._cached_lines = result

        return result if result else [""]

    def _apply_default_style(self, text: str) -> str:
        if not self._default_text_style:
            return text
        styled = text
        s = self._default_text_style
        if s.color:
            styled = s.color(styled)
        if s.bold:
            styled = self._theme.bold(styled)
        if s.italic:
            styled = self._theme.italic(styled)
        if s.strikethrough:
            styled = self._theme.strikethrough(styled)
        if s.underline:
            styled = self._theme.underline(styled)
        return styled

    def _get_default_style_prefix(self) -> str:
        if not self._default_text_style:
            return ""
        if self._default_style_prefix is not None:
            return self._default_style_prefix
        sentinel = "\u0000"
        styled = self._apply_default_style(sentinel)
        idx = styled.find(sentinel)
        self._default_style_prefix = styled[:idx] if idx >= 0 else ""
        return self._default_style_prefix

    def _get_style_prefix(self, style_fn: Callable[[str], str]) -> str:
        sentinel = "\u0000"
        styled = style_fn(sentinel)
        idx = styled.find(sentinel)
        return styled[:idx] if idx >= 0 else ""

    def _render_markdown(self, text: str, width: int) -> list[str]:
        """Parse markdown and render to styled terminal lines."""
        if self._md is None:
            return self._render_plain_fallback(text)

        try:
            import mistune
            tokens = self._md(text)  # AST
            if tokens is None:
                return self._render_plain_fallback(text)
            return self._render_tokens(tokens, width)
        except Exception:
            return self._render_plain_fallback(text)

    def _render_plain_fallback(self, text: str) -> list[str]:
        return [self._apply_default_style(line) for line in text.split("\n")]

    def _render_tokens(self, tokens: list[dict], width: int) -> list[str]:
        lines: list[str] = []
        for i, token in enumerate(tokens):
            next_type = tokens[i + 1]["type"] if i + 1 < len(tokens) else None
            lines.extend(self._render_token(token, width, next_type))
        return lines

    def _render_token(self, token: dict, width: int, next_token_type: str | None) -> list[str]:
        lines: list[str] = []
        t = token.get("type", "")

        if t == "heading":
            level = token.get("attrs", {}).get("level", 1)
            heading_text = self._render_children(token.get("children", []))
            prefix = "#" * level + " "
            if level == 1:
                styled = self._theme.heading(self._theme.bold(self._theme.underline(heading_text)))
            elif level == 2:
                styled = self._theme.heading(self._theme.bold(heading_text))
            else:
                styled = self._theme.heading(self._theme.bold(prefix + heading_text))
            lines.append(styled)
            if next_token_type != "blank_line":
                lines.append("")

        elif t == "paragraph":
            para_text = self._render_children(token.get("children", []))
            lines.append(para_text)
            if next_token_type and next_token_type not in ("list", "blank_line"):
                lines.append("")

        elif t == "block_code":
            raw = token.get("raw", "")
            lang = token.get("attrs", {}).get("info", "") or ""
            indent = self._theme.code_block_indent
            lines.append(self._theme.code_block_border(f"```{lang}"))
            if self._theme.highlight_code:
                for hl_line in self._theme.highlight_code(raw, lang or None):
                    lines.append(f"{indent}{hl_line}")
            else:
                for code_line in raw.split("\n"):
                    lines.append(f"{indent}{self._theme.code_block(code_line)}")
            lines.append(self._theme.code_block_border("```"))
            if next_token_type != "blank_line":
                lines.append("")

        elif t == "list":
            ordered = token.get("attrs", {}).get("ordered", False)
            start = token.get("attrs", {}).get("start", 1) or 1
            lines.extend(self._render_list(token.get("children", []), ordered, start, 0))

        elif t == "block_quote":
            def quote_style(text: str) -> str:
                return self._theme.quote(self._theme.italic(text))
            children_text = self._render_children_with_style(token.get("children", []), quote_style)
            quote_content_width = max(1, width - 2)
            for quote_line in children_text.split("\n"):
                for wrapped_line in wrap_text_with_ansi(quote_line, quote_content_width):
                    lines.append(self._theme.quote_border("│ ") + wrapped_line)
            if next_token_type != "blank_line":
                lines.append("")

        elif t == "thematic_break":
            lines.append(self._theme.hr("─" * min(width, 80)))
            if next_token_type != "blank_line":
                lines.append("")

        elif t == "block_html":
            raw = token.get("raw", "")
            if raw:
                lines.append(self._apply_default_style(raw.strip()))

        elif t == "blank_line":
            lines.append("")

        elif t == "table":
            lines.extend(self._render_table(token, width))

        else:
            raw = token.get("raw", "")
            if raw:
                lines.append(self._apply_default_style(raw))

        return lines

    def _render_children(self, children: list[dict], style_fn: Callable[[str], str] | None = None) -> str:
        apply = style_fn or self._apply_default_style
        prefix = self._get_default_style_prefix()
        return self._render_inline_tokens(children, apply, prefix)

    def _render_children_with_style(self, children: list[dict], style_fn: Callable[[str], str]) -> str:
        prefix = self._get_style_prefix(style_fn)
        return self._render_inline_tokens(children, style_fn, prefix)

    def _render_inline_tokens(
        self,
        tokens: list[dict],
        apply_text: Callable[[str], str],
        style_prefix: str,
    ) -> str:
        result = ""

        def apply_with_newlines(text: str) -> str:
            return "\n".join(apply_text(seg) for seg in text.split("\n"))

        for token in tokens:
            t = token.get("type", "")
            if t in ("text", "softbreak"):
                raw = token.get("raw", "")
                children = token.get("children")
                if children:
                    result += self._render_inline_tokens(children, apply_text, style_prefix)
                else:
                    result += apply_with_newlines(raw)

            elif t == "paragraph":
                result += self._render_inline_tokens(token.get("children", []), apply_text, style_prefix)

            elif t == "strong":
                content = self._render_inline_tokens(token.get("children", []), apply_text, style_prefix)
                result += self._theme.bold(content) + style_prefix

            elif t == "emphasis":
                content = self._render_inline_tokens(token.get("children", []), apply_text, style_prefix)
                result += self._theme.italic(content) + style_prefix

            elif t == "codespan":
                result += self._theme.code(token.get("raw", "")) + style_prefix

            elif t == "link":
                link_text = self._render_inline_tokens(token.get("children", []), apply_text, style_prefix)
                attrs = token.get("attrs", {})
                href = attrs.get("url", "")
                raw_text = token.get("raw", "")
                href_for_cmp = href[7:] if href.startswith("mailto:") else href
                if raw_text == href or raw_text == href_for_cmp:
                    result += self._theme.link(self._theme.underline(link_text)) + style_prefix
                else:
                    result += (
                        self._theme.link(self._theme.underline(link_text)) +
                        self._theme.link_url(f" ({href})") +
                        style_prefix
                    )

            elif t == "linebreak":
                result += "\n"

            elif t in ("strikethrough", "del"):
                content = self._render_inline_tokens(token.get("children", []), apply_text, style_prefix)
                result += self._theme.strikethrough(content) + style_prefix

            elif t in ("inline_html", "html"):
                raw = token.get("raw", "")
                if raw:
                    result += apply_with_newlines(raw)

            else:
                raw = token.get("raw", "")
                if raw:
                    result += apply_with_newlines(raw)

        return result

    def _render_list(
        self,
        items: list[dict],
        ordered: bool,
        start_number: int,
        depth: int,
    ) -> list[str]:
        lines: list[str] = []
        indent = "  " * depth
        for i, item in enumerate(items):
            bullet = f"{start_number + i}. " if ordered else "- "
            item_lines = self._render_list_item(item.get("children", []), depth)
            if item_lines:
                first = item_lines[0]
                # Check if first line is a nested list (starts with spaces + bullet pattern)
                nested_list_re = re.compile(r"^\s+\x1b\[36m[-\d]")
                is_nested = bool(nested_list_re.match(first))
                if is_nested:
                    lines.append(first)
                else:
                    lines.append(indent + self._theme.list_bullet(bullet) + first)
                for ln in item_lines[1:]:
                    if bool(nested_list_re.match(ln)):
                        lines.append(ln)
                    else:
                        lines.append(f"{indent}  {ln}")
            else:
                lines.append(indent + self._theme.list_bullet(bullet))
        return lines

    def _render_list_item(self, tokens: list[dict], parent_depth: int) -> list[str]:
        lines: list[str] = []
        for token in tokens:
            t = token.get("type", "")
            if t == "list":
                ordered = token.get("attrs", {}).get("ordered", False)
                start = token.get("attrs", {}).get("start", 1) or 1
                lines.extend(self._render_list(token.get("children", []), ordered, start, parent_depth + 1))
            elif t in ("paragraph", "text"):
                children = token.get("children")
                if children:
                    lines.append(self._render_children(children))
                else:
                    raw = token.get("raw", "")
                    if raw:
                        lines.append(self._apply_default_style(raw))
            elif t == "block_code":
                raw = token.get("raw", "")
                lang = token.get("attrs", {}).get("info", "") or ""
                indent = self._theme.code_block_indent
                lines.append(self._theme.code_block_border(f"```{lang}"))
                if self._theme.highlight_code:
                    for hl_line in self._theme.highlight_code(raw, lang or None):
                        lines.append(f"{indent}{hl_line}")
                else:
                    for code_line in raw.split("\n"):
                        lines.append(f"{indent}{self._theme.code_block(code_line)}")
                lines.append(self._theme.code_block_border("```"))
            else:
                text = self._render_inline_tokens(
                    [token],
                    self._apply_default_style,
                    self._get_default_style_prefix(),
                )
                if text:
                    lines.append(text)
        return lines

    def _render_table(self, token: dict, available_width: int) -> list[str]:
        lines: list[str] = []
        children = token.get("children", [])
        if not children:
            return lines

        # Find head and body
        head_token = next((c for c in children if c.get("type") == "table_head"), None)
        body_token = next((c for c in children if c.get("type") == "table_body"), None)

        if not head_token:
            return lines

        header_rows = head_token.get("children", [])
        body_rows = body_token.get("children", []) if body_token else []
        all_header_cells = [cell for row in header_rows for cell in row.get("children", [])]
        num_cols = len(all_header_cells)

        if num_cols == 0:
            return lines

        border_overhead = 3 * num_cols + 1
        available_for_cells = available_width - border_overhead
        if available_for_cells < num_cols:
            raw = token.get("raw", "")
            if raw:
                lines.extend(wrap_text_with_ansi(raw, available_width))
                lines.append("")
            return lines

        # Compute column widths based on natural content
        natural_widths = [0] * num_cols
        for i, cell in enumerate(all_header_cells):
            text = self._render_children(cell.get("children", []))
            natural_widths[i] = max(natural_widths[i], visible_width(text))

        for row in body_rows:
            for i, cell in enumerate(row.get("children", [])):
                if i < num_cols:
                    text = self._render_children(cell.get("children", []))
                    natural_widths[i] = max(natural_widths[i], visible_width(text))

        # Simple proportional allocation if needed
        total_natural = sum(natural_widths) + border_overhead
        if total_natural <= available_width:
            col_widths = natural_widths[:]
        else:
            total_natural_cells = sum(natural_widths)
            if total_natural_cells <= 0:
                col_widths = [max(1, available_for_cells // num_cols)] * num_cols
            else:
                col_widths = [
                    max(1, int(w / total_natural_cells * available_for_cells))
                    for w in natural_widths
                ]
                allocated = sum(col_widths)
                leftover = available_for_cells - allocated
                for i in range(num_cols):
                    if leftover <= 0:
                        break
                    col_widths[i] += 1
                    leftover -= 1

        def render_row_lines(cells: list[dict], col_widths: list[int], bold: bool) -> list[str]:
            cell_lines: list[list[str]] = []
            for i, cell in enumerate(cells):
                text = self._render_children(cell.get("children", []))
                wrapped = wrap_text_with_ansi(text, max(1, col_widths[i]))
                if bold:
                    wrapped = [self._theme.bold(ln) for ln in wrapped]
                cell_lines.append(wrapped)
            row_line_count = max((len(cl) for cl in cell_lines), default=1)
            result: list[str] = []
            for li in range(row_line_count):
                parts = []
                for ci, cl in enumerate(cell_lines):
                    txt = cl[li] if li < len(cl) else ""
                    pad = " " * max(0, col_widths[ci] - visible_width(txt))
                    parts.append(txt + pad)
                result.append("│ " + " │ ".join(parts) + " │")
            return result

        # Top border
        top_cells = ["─" * w for w in col_widths]
        lines.append("┌─" + "─┬─".join(top_cells) + "─┐")

        # Header
        for header_row in header_rows:
            header_cells = header_row.get("children", [])
            lines.extend(render_row_lines(header_cells, col_widths, bold=True))

        # Separator
        sep_cells = ["─" * w for w in col_widths]
        separator = "├─" + "─┼─".join(sep_cells) + "─┤"
        lines.append(separator)

        # Body rows
        for ri, row in enumerate(body_rows):
            row_cells = row.get("children", [])
            lines.extend(render_row_lines(row_cells, col_widths, bold=False))
            if ri < len(body_rows) - 1:
                lines.append(separator)

        # Bottom border
        bot_cells = ["─" * w for w in col_widths]
        lines.append("└─" + "─┴─".join(bot_cells) + "─┘")
        lines.append("")

        return lines
