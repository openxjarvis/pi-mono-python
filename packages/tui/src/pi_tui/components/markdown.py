"""Markdown component — mirrors components/markdown.ts"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..latex import render_latex
from ..terminal_image import is_image_line
from ..utils import apply_background_to_line, visible_width, wrap_text_with_ansi

_LATEX_PLACEHOLDER_RE = re.compile(r"\ufdd0L(\d+)\ufdd1")
_CODE_PROTECT_RE = re.compile(r"(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`]+`)")
_PENDING_DOLLAR_MATH_RE = re.compile(r"\\[A-Za-z]+|[_^=+*/<>()[\]|±≤≥≠≈∈→⇒∞∫∑√-]")


def _is_escaped(source: str, index: int) -> bool:
    backslashes = 0
    position = index - 1
    while position >= 0 and source[position] == "\\":
        backslashes += 1
        position -= 1
    return backslashes % 2 == 1


def _find_closing_delimiter(source: str, closing: str, start: int) -> int:
    index = source.find(closing, start)
    while index >= 0 and _is_escaped(source, index):
        index = source.find(closing, index + len(closing))
    return index


def _looks_like_pending_dollar_math(source: str) -> bool:
    return bool(_PENDING_DOLLAR_MATH_RE.search(source))


def _tokenize_inline_latex(source: str) -> dict[str, Any] | None:
    if source.startswith("$$"):
        opening, closing = "$$", "$$"
    elif source.startswith("\\("):
        opening, closing = "\\(", "\\)"
    elif source.startswith("\\["):
        opening, closing = "\\[", "\\]"
    elif source.startswith("$") and not re.match(r"^\$\s", source):
        opening, closing = "$", "$"
    else:
        return None

    closing_index = _find_closing_delimiter(source, closing, len(opening))
    if (
        closing_index >= 0
        and opening == "$"
        and (
            re.search(r"\s$", source[len(opening):closing_index])
            or re.match(r"^\d", source[closing_index + 1:])
            or (
                re.fullmatch(r"[A-Z_][A-Z0-9_]*(?:[^A-Za-z0-9_\s])?", source[len(opening):closing_index])
                and re.match(r"^[A-Za-z_][A-Za-z0-9_]*", source[closing_index + 1:])
            )
            or "`" in source[len(opening):closing_index]
        )
    ):
        return None

    if closing_index < 0:
        pending_source = source[len(opening):]
        if opening.startswith("\\") or _looks_like_pending_dollar_math(pending_source):
            return {"type": "latex", "raw": source, "text": pending_source, "pending": True}
        return None

    text = source[len(opening):closing_index]
    if not text or "\n" in text:
        return None
    return {
        "type": "latex",
        "raw": source[: closing_index + len(closing)],
        "text": text,
        "pending": False,
    }


def _tokenize_block_latex(source: str) -> dict[str, Any] | None:
    dollar = re.match(r"^ {0,3}\$\$[ \t]*(?:\n)?([\s\S]*?)\$\$[ \t]*(?:\n|$)", source)
    if dollar and dollar.group(1):
        return {"type": "latexBlock", "raw": dollar.group(0), "text": dollar.group(1).strip(), "pending": False}
    bracket = re.match(r"^ {0,3}\\\[[ \t]*(?:\n)?([\s\S]*?)\\\][ \t]*(?:\n|$)", source)
    if bracket and bracket.group(1):
        return {"type": "latexBlock", "raw": bracket.group(0), "text": bracket.group(1).strip(), "pending": False}
    pending_bracket = re.match(r"^ {0,3}\\\[[ \t]*(?:\n)?([\s\S]*)$", source)
    if pending_bracket:
        return {"type": "latexBlock", "raw": pending_bracket.group(0), "text": pending_bracket.group(1), "pending": True}
    pending_dollar = re.match(r"^ {0,3}\$\$[ \t]*(?:\n)?([\s\S]*)$", source)
    if pending_dollar and pending_dollar.group(1) and _looks_like_pending_dollar_math(pending_dollar.group(1)):
        return {"type": "latexBlock", "raw": pending_dollar.group(0), "text": pending_dollar.group(1), "pending": True}
    return None


def _replace_latex_in_text(text: str, replacements: list[dict[str, Any]]) -> str:
    out: list[str] = []
    index = 0
    while index < len(text):
        if index == 0 or text[index - 1] == "\n":
            block = _tokenize_block_latex(text[index:])
            if block:
                out.append(f"\ufdd0L{len(replacements)}\ufdd1")
                replacements.append(block)
                index += len(block["raw"])
                continue
        inline = _tokenize_inline_latex(text[index:])
        if inline:
            out.append(f"\ufdd0L{len(replacements)}\ufdd1")
            replacements.append(inline)
            index += len(inline["raw"])
            continue
        out.append(text[index])
        index += 1
    return "".join(out)


def _protect_code_and_extract_latex(text: str) -> tuple[str, list[dict[str, Any]]]:
    replacements: list[dict[str, Any]] = []
    parts: list[str] = []
    last = 0
    for match in _CODE_PROTECT_RE.finditer(text):
        if match.start() > last:
            parts.append(_replace_latex_in_text(text[last:match.start()], replacements))
        parts.append(match.group(0))
        last = match.end()
    if last < len(text):
        parts.append(_replace_latex_in_text(text[last:], replacements))
    return "".join(parts), replacements


def _trim_partial_closing_fences(tokens: list[dict]) -> None:
    if not tokens:
        return
    token = tokens[-1]
    token_type = token.get("type")
    if token_type == "list":
        items = token.get("children") or []
        if items:
            _trim_partial_closing_fences(items[-1].get("children") or [])
        return
    if token_type == "block_quote":
        _trim_partial_closing_fences(token.get("children") or [])
        return
    if token_type != "block_code":
        return
    raw = token.get("raw", "")
    marker_match = re.match(r"(`{3,}|~{3,})", raw)
    if not marker_match:
        marker = token.get("marker") or ""
        last_line = raw.split("\n")[-1] if raw else ""
        if marker and last_line and len(last_line) < len(marker) and last_line == marker[0] * len(last_line):
            token["raw"] = raw[: -len(last_line)].rstrip("\n")
        return
    marker = marker_match.group(1)
    last_line = raw.split("\n")[-1]
    if not last_line or len(last_line) >= len(marker) or last_line != marker[0] * len(last_line):
        return
    token["raw"] = raw[: -len(last_line)].rstrip("\n")


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


@dataclass
class MarkdownOptions:
    """Optional Markdown renderer behavior — mirrors markdown.ts MarkdownOptions."""
    preserve_ordered_list_markers: bool = False
    preserve_backslash_escapes: bool = False
    transform: Callable[[str, int], str] | None = None
    render_latex: bool = True


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
        options: MarkdownOptions | None = None,
    ) -> None:
        self._text = text
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._theme = theme
        self._default_text_style = default_text_style
        self._options = options or MarkdownOptions()
        self._default_style_prefix: str | None = None
        self._latex_replacements: list[dict[str, Any]] = []

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

        padding_x = min(self._padding_x, max(0, (width - 1) // 2))
        content_width = max(1, width - padding_x * 2)
        source = self._text
        if self._options.transform:
            source = self._options.transform(source, content_width)

        if not source or not source.strip():
            result: list[str] = []
            self._cached_text = self._text
            self._cached_width = width
            self._cached_lines = result
            return result

        normalized = source.replace("\t", "   ")
        rendered_lines = self._render_markdown(normalized, content_width)

        wrapped_lines: list[str] = []
        for line in rendered_lines:
            if is_image_line(line):
                wrapped_lines.append(line)
            else:
                wrapped_lines.extend(wrap_text_with_ansi(line, content_width))

        left_margin = " " * padding_x
        right_margin = " " * padding_x
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

    def _expand_latex(self, text: str) -> str:
        if not self._latex_replacements or "\ufdd0L" not in text:
            return text

        def replace(match: re.Match[str]) -> str:
            item = self._latex_replacements[int(match.group(1))]
            is_block = item["type"] == "latexBlock"
            if item.get("pending") or self._options.render_latex is False:
                return item["raw"].strip() if is_block else item["raw"]
            rendered = render_latex(item["text"], display=is_block)
            if rendered is None:
                return item["raw"].strip() if is_block else item["raw"]
            return rendered

        return _LATEX_PLACEHOLDER_RE.sub(replace, text)

    def _render_markdown(self, text: str, width: int) -> list[str]:
        """Parse markdown and render to styled terminal lines."""
        if self._md is None:
            return self._render_plain_fallback(text)

        try:
            import mistune
            prepared, self._latex_replacements = _protect_code_and_extract_latex(text)
            tokens = self._md(prepared)  # AST
            if tokens is None:
                return self._render_plain_fallback(text)
            _trim_partial_closing_fences(tokens)
            return self._render_tokens(tokens, width)
        except Exception:
            return self._render_plain_fallback(text)

    def _render_plain_fallback(self, text: str) -> list[str]:
        return [self._apply_default_style(line) for line in self._expand_latex(text).split("\n")]

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
            prefix = "#" * level + " "
            if level == 1:
                heading_style_fn: Callable[[str], str] = (
                    lambda value: self._theme.heading(self._theme.bold(self._theme.underline(value)))
                )
            else:
                heading_style_fn = lambda value: self._theme.heading(self._theme.bold(value))
            heading_prefix = self._get_style_prefix(heading_style_fn)
            heading_text = self._render_inline_tokens(
                token.get("children", []), heading_style_fn, heading_prefix
            )
            styled = heading_style_fn(prefix) + heading_text if level >= 3 else heading_text
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
            lines.extend(self._render_list(token.get("children", []), ordered, start, 0, token))

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
            expanded = self._expand_latex(text)
            return "\n".join(apply_text(seg) for seg in expanded.split("\n"))

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
                styled_link = self._theme.link(self._theme.underline(link_text))
                try:
                    from ..terminal_image import get_capabilities, hyperlink
                    if href and get_capabilities().hyperlinks:
                        # Restore surrounding style after OSC 8 so table/blockquote colors do not leak.
                        result += hyperlink(styled_link, href) + style_prefix
                        continue
                except Exception:
                    pass
                if raw_text == href or raw_text == href_for_cmp:
                    result += styled_link + style_prefix
                else:
                    result += styled_link + self._theme.link_url(f" ({href})") + style_prefix

            elif t == "escape":
                raw = token.get("raw", "")
                escaped = raw if self._options.preserve_backslash_escapes else token.get("text", raw.lstrip("\\"))
                result += apply_with_newlines(escaped)

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

    def _list_item_bullet(self, item: dict, ordered: bool, number: int) -> str:
        raw = item.get("raw", "") or ""
        if ordered:
            if self._options.preserve_ordered_list_markers:
                match = re.match(r"^(?: {0,3})(\d{1,9}[.)])[ \t]+", raw)
                bullet = f"{match.group(1)} " if match else f"{number}. "
            else:
                bullet = f"{number}. "
        elif self._options.preserve_ordered_list_markers:
            match = re.match(r"^(?: {0,3})([-+*])(?:[ \t]+|(?=\r?\n|$))", raw)
            bullet = f"{match.group(1)} " if match else "- "
        else:
            bullet = "- "
        attrs = item.get("attrs") or {}
        checked = attrs.get("checked")
        if checked is not None or item.get("task"):
            bullet += f"[{'x' if checked else ' '}] "
        return bullet

    def _render_list(
        self,
        items: list[dict],
        ordered: bool,
        start_number: int,
        depth: int,
        parent: dict | None = None,
    ) -> list[str]:
        lines: list[str] = []
        indent = "    " * depth
        for i, item in enumerate(items):
            bullet = self._list_item_bullet(item, ordered, start_number + i)
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
                lines.extend(self._render_list(token.get("children", []), ordered, start, parent_depth + 1, token))
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

        def wrap_cell_text(text: str, max_width: int, style_prefix: str = "") -> list[str]:
            wrapped = wrap_text_with_ansi(text, max(1, max_width))
            # Reset text styles after each non-final fragment so link/color
            # SGR cannot leak into table borders or neighboring cells.
            return [
                f"{line}{'' if index == len(wrapped) - 1 else chr(27) + '[22;23;24;25;27;28;29;39m'}{style_prefix}"
                for index, line in enumerate(wrapped)
            ]

        style_prefix = self._get_default_style_prefix()

        def render_row_lines(cells: list[dict], col_widths: list[int], bold: bool) -> list[str]:
            cell_lines: list[list[str]] = []
            for i, cell in enumerate(cells):
                text = self._render_children(cell.get("children", []))
                wrapped = wrap_cell_text(text, col_widths[i] if i < len(col_widths) else 1, style_prefix)
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
