"""
Terminal text utilities — mirrors packages/tui/src/utils.ts

Provides:
- visible_width(): calculate terminal column width of a string
- truncate_to_width(): truncate with ellipsis, ANSI-aware
- wrap_text_with_ansi(): word-wrap preserving ANSI codes
- slice_by_column() / slice_with_width(): column-based slicing
- extract_segments(): single-pass before/after extraction for overlay compositing
- AnsiCodeTracker: track active SGR codes across line breaks
- apply_background_to_line(): pad and apply background color function
"""
from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple

try:
    from wcwidth import wcwidth as _wcwidth
    _HAS_WCWIDTH = True
except ImportError:
    _HAS_WCWIDTH = False

# ─────────────────────────────────────────────────────────────────────────────
# Width cache (mirrors widthCache in TS, LRU-style)
# ─────────────────────────────────────────────────────────────────────────────
_WIDTH_CACHE_SIZE = 512
_width_cache: dict[str, int] = {}
_width_cache_order: list[str] = []

# Strip ANSI escape sequences for width calculation
_ANSI_SGR_RE = re.compile(r"\x1b\[[0-9;]*[mGKHJA-Z]")
_ANSI_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_ANSI_APC_RE = re.compile(r"\x1b_[^\x07\x1b]*(?:\x07|\x1b\\)")

# Emoji codepoint ranges (fast pre-filter, mirrors couldBeEmoji)
def _could_be_emoji(cp: int, segment: str) -> bool:
    return (
        (0x1f000 <= cp <= 0x1fbff) or
        (0x2300 <= cp <= 0x23ff) or
        (0x2600 <= cp <= 0x27bf) or
        (0x2b50 <= cp <= 0x2b55) or
        "\ufe0f" in segment or
        len(segment) > 2
    )

def _grapheme_width(segment: str) -> int:
    """Calculate terminal width of a single grapheme cluster."""
    if not segment:
        return 0

    cp = ord(segment[0]) if segment else 0

    # Zero-width control / combining characters
    cat = unicodedata.category(segment[0])
    if cat in ("Mn", "Me", "Cf", "Cc", "Cs", "Co", "Cn"):
        # Variation selectors and ZWJ have no width
        if all(unicodedata.category(c) in ("Mn", "Me", "Cf", "Cc", "Cs") for c in segment):
            return 0

    # Emoji: 2 columns
    if _could_be_emoji(cp, segment):
        # Multi-codepoint (ZWJ sequences, flag pairs, skin tones) = 2
        if len(segment) > 1:
            return 2
        # Check east-asian-wide emoji
        if _HAS_WCWIDTH:
            w = _wcwidth(segment[0])
            if w == 2:
                return 2
        if 0x1f000 <= cp <= 0x1fbff:
            return 2

    # Use wcwidth when available
    if _HAS_WCWIDTH:
        w = _wcwidth(segment[0])
        if w < 0:
            return 0
        return w

    # Fallback: east-asian double-width ranges
    if (
        (0x1100 <= cp <= 0x115f) or
        (0x2e80 <= cp <= 0x9fff) or
        (0xac00 <= cp <= 0xd7af) or
        (0xf900 <= cp <= 0xfaff) or
        (0xfe10 <= cp <= 0xfe1f) or
        (0xfe30 <= cp <= 0xfe6f) or
        (0xff00 <= cp <= 0xff60) or
        (0xffe0 <= cp <= 0xffe6) or
        (0x1f004 <= cp <= 0x1f0cf) or
        (0x1f18e <= cp <= 0x1f9ff) or
        (0x20000 <= cp <= 0x2fffd) or
        (0x30000 <= cp <= 0x3fffd)
    ):
        return 2
    return 1


def _segment_graphemes(text: str) -> list[str]:
    """Segment text into grapheme clusters (simplified — uses unicodedata)."""
    if not text:
        return []
    # Python doesn't have Intl.Segmenter; we approximate by iterating codepoints
    # and grouping combining characters with their base
    clusters: list[str] = []
    i = 0
    chars = list(text)
    while i < len(chars):
        cluster = chars[i]
        i += 1
        # Consume combining marks
        while i < len(chars):
            cat = unicodedata.category(chars[i])
            if cat in ("Mn", "Me", "Cf") or ord(chars[i]) in (0x200D, 0xFE0F, 0x20E3):
                cluster += chars[i]
                i += 1
            else:
                break
        clusters.append(cluster)
    return clusters


def visible_width(s: str) -> int:
    """
    Calculate the visible terminal column width of a string.
    Handles ANSI escape codes, wide chars, emoji, tabs.
    Mirrors visibleWidth() in utils.ts.
    """
    if not s:
        return 0

    # Fast path: pure ASCII printable
    if all(0x20 <= ord(c) <= 0x7e for c in s):
        return len(s)

    # Cache check
    cached = _width_cache.get(s)
    if cached is not None:
        return cached

    # Normalize: tabs → 3 spaces
    clean = s.replace("\t", "   ") if "\t" in s else s

    # Strip ANSI escape codes
    if "\x1b" in clean:
        clean = _ANSI_SGR_RE.sub("", clean)
        clean = _ANSI_OSC_RE.sub("", clean)
        clean = _ANSI_APC_RE.sub("", clean)

    # Measure
    width = sum(_grapheme_width(g) for g in _segment_graphemes(clean))

    # Cache
    if len(_width_cache) >= _WIDTH_CACHE_SIZE:
        oldest = _width_cache_order.pop(0) if _width_cache_order else next(iter(_width_cache))
        _width_cache.pop(oldest, None)
    _width_cache[s] = width
    _width_cache_order.append(s)

    return width


# ─────────────────────────────────────────────────────────────────────────────
# ANSI code extraction
# ─────────────────────────────────────────────────────────────────────────────

class _AnsiExtract(NamedTuple):
    code: str
    length: int


def extract_ansi_code(s: str, pos: int) -> _AnsiExtract | None:
    """Extract ANSI escape sequence starting at pos. Returns None if not found."""
    if pos >= len(s) or s[pos] != "\x1b":
        return None
    if pos + 1 >= len(s):
        return None
    next_ch = s[pos + 1]

    # CSI: ESC [ ... m/G/K/H/J
    if next_ch == "[":
        j = pos + 2
        while j < len(s) and s[j] not in "mGKHJABCDEFSTfu~":
            j += 1
        if j < len(s):
            return _AnsiExtract(s[pos:j+1], j + 1 - pos)
        return None

    # OSC: ESC ] ... BEL or ESC ] ... ST
    if next_ch == "]":
        j = pos + 2
        while j < len(s):
            if s[j] == "\x07":
                return _AnsiExtract(s[pos:j+1], j + 1 - pos)
            if s[j] == "\x1b" and j + 1 < len(s) and s[j+1] == "\\":
                return _AnsiExtract(s[pos:j+2], j + 2 - pos)
            j += 1
        return None

    # APC: ESC _ ... BEL or ESC _ ... ST
    if next_ch == "_":
        j = pos + 2
        while j < len(s):
            if s[j] == "\x07":
                return _AnsiExtract(s[pos:j+1], j + 1 - pos)
            if s[j] == "\x1b" and j + 1 < len(s) and s[j+1] == "\\":
                return _AnsiExtract(s[pos:j+2], j + 2 - pos)
            j += 1
        return None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# ANSI SGR code tracker — mirrors AnsiCodeTracker in utils.ts
# ─────────────────────────────────────────────────────────────────────────────

class AnsiCodeTracker:
    """Track active ANSI SGR codes to preserve styling across line breaks."""

    __slots__ = (
        "_bold", "_dim", "_italic", "_underline", "_blink",
        "_inverse", "_hidden", "_strikethrough",
        "_fg_color", "_bg_color",
    )

    def __init__(self) -> None:
        self._bold = False
        self._dim = False
        self._italic = False
        self._underline = False
        self._blink = False
        self._inverse = False
        self._hidden = False
        self._strikethrough = False
        self._fg_color: str | None = None
        self._bg_color: str | None = None

    def _reset(self) -> None:
        self._bold = False
        self._dim = False
        self._italic = False
        self._underline = False
        self._blink = False
        self._inverse = False
        self._hidden = False
        self._strikethrough = False
        self._fg_color = None
        self._bg_color = None

    def clear(self) -> None:
        self._reset()

    def process(self, ansi_code: str) -> None:
        """Update state from an ANSI escape sequence."""
        if not ansi_code.endswith("m"):
            return
        m = re.match(r"\x1b\[([\d;]*)m", ansi_code)
        if not m:
            return
        params = m.group(1)
        if params in ("", "0"):
            self._reset()
            return

        parts = params.split(";")
        i = 0
        while i < len(parts):
            try:
                code = int(parts[i])
            except ValueError:
                i += 1
                continue

            if code in (38, 48):
                if i + 2 < len(parts) and parts[i+1] == "5":
                    color = f"{parts[i]};{parts[i+1]};{parts[i+2]}"
                    if code == 38:
                        self._fg_color = color
                    else:
                        self._bg_color = color
                    i += 3
                    continue
                elif i + 4 < len(parts) and parts[i+1] == "2":
                    color = f"{parts[i]};{parts[i+1]};{parts[i+2]};{parts[i+3]};{parts[i+4]}"
                    if code == 38:
                        self._fg_color = color
                    else:
                        self._bg_color = color
                    i += 5
                    continue

            if code == 0:
                self._reset()
            elif code == 1:
                self._bold = True
            elif code == 2:
                self._dim = True
            elif code == 3:
                self._italic = True
            elif code == 4:
                self._underline = True
            elif code == 5:
                self._blink = True
            elif code == 7:
                self._inverse = True
            elif code == 8:
                self._hidden = True
            elif code == 9:
                self._strikethrough = True
            elif code == 21:
                self._bold = False
            elif code == 22:
                self._bold = False
                self._dim = False
            elif code == 23:
                self._italic = False
            elif code == 24:
                self._underline = False
            elif code == 25:
                self._blink = False
            elif code == 27:
                self._inverse = False
            elif code == 28:
                self._hidden = False
            elif code == 29:
                self._strikethrough = False
            elif code == 39:
                self._fg_color = None
            elif code == 49:
                self._bg_color = None
            elif (30 <= code <= 37) or (90 <= code <= 97):
                self._fg_color = str(code)
            elif (40 <= code <= 47) or (100 <= code <= 107):
                self._bg_color = str(code)
            i += 1

    def get_active_codes(self) -> str:
        """Return ESC sequence to restore current SGR state, or empty string."""
        codes: list[str] = []
        if self._bold:
            codes.append("1")
        if self._dim:
            codes.append("2")
        if self._italic:
            codes.append("3")
        if self._underline:
            codes.append("4")
        if self._blink:
            codes.append("5")
        if self._inverse:
            codes.append("7")
        if self._hidden:
            codes.append("8")
        if self._strikethrough:
            codes.append("9")
        if self._fg_color:
            codes.append(self._fg_color)
        if self._bg_color:
            codes.append(self._bg_color)
        if not codes:
            return ""
        return f"\x1b[{';'.join(codes)}m"

    def has_active_codes(self) -> bool:
        return bool(
            self._bold or self._dim or self._italic or self._underline or
            self._blink or self._inverse or self._hidden or self._strikethrough or
            self._fg_color or self._bg_color
        )

    def get_line_end_reset(self) -> str:
        """Return reset code for attributes that bleed into padding (underline only)."""
        if self._underline:
            return "\x1b[24m"
        return ""


def _update_tracker_from_text(text: str, tracker: AnsiCodeTracker) -> None:
    i = 0
    while i < len(text):
        result = extract_ansi_code(text, i)
        if result:
            tracker.process(result.code)
            i += result.length
        else:
            i += 1


# ─────────────────────────────────────────────────────────────────────────────
# Token splitting (mirrors splitIntoTokensWithAnsi)
# ─────────────────────────────────────────────────────────────────────────────

def _split_into_tokens_with_ansi(text: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    pending_ansi = ""
    in_whitespace = False
    i = 0

    while i < len(text):
        ansi = extract_ansi_code(text, i)
        if ansi:
            pending_ansi += ansi.code
            i += ansi.length
            continue

        ch = text[i]
        ch_is_space = ch == " "

        if ch_is_space != in_whitespace and current:
            tokens.append(current)
            current = ""

        if pending_ansi:
            current += pending_ansi
            pending_ansi = ""

        in_whitespace = ch_is_space
        current += ch
        i += 1

    if pending_ansi:
        current += pending_ansi
    if current:
        tokens.append(current)
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Word wrapping — mirrors wrapTextWithAnsi / wrapSingleLine
# ─────────────────────────────────────────────────────────────────────────────

def wrap_text_with_ansi(text: str, width: int) -> list[str]:
    """
    Wrap text to width, preserving ANSI codes across line breaks.
    Mirrors wrapTextWithAnsi() in utils.ts.
    """
    if not text:
        return [""]

    input_lines = text.split("\n")
    result: list[str] = []
    tracker = AnsiCodeTracker()

    for line in input_lines:
        prefix = tracker.get_active_codes() if result else ""
        result.extend(_wrap_single_line(prefix + line, width, tracker))
        _update_tracker_from_text(line, tracker)

    return result if result else [""]


def _break_long_word(word: str, width: int, tracker: AnsiCodeTracker) -> list[str]:
    lines: list[str] = []
    current_line = tracker.get_active_codes()
    current_width = 0

    # Parse word into (type, value) segments
    segments: list[tuple[str, str]] = []
    i = 0
    while i < len(word):
        ansi = extract_ansi_code(word, i)
        if ansi:
            segments.append(("ansi", ansi.code))
            i += ansi.length
        else:
            # collect non-ANSI text
            end = i
            while end < len(word) and not extract_ansi_code(word, end):
                end += 1
            for g in _segment_graphemes(word[i:end]):
                segments.append(("grapheme", g))
            i = end

    for seg_type, seg_val in segments:
        if seg_type == "ansi":
            current_line += seg_val
            tracker.process(seg_val)
            continue

        gw = _grapheme_width(seg_val)
        if current_width + gw > width:
            reset = tracker.get_line_end_reset()
            if reset:
                current_line += reset
            lines.append(current_line)
            current_line = tracker.get_active_codes()
            current_width = 0

        current_line += seg_val
        current_width += gw

    if current_line:
        lines.append(current_line)

    return lines if lines else [""]


def _wrap_single_line(line: str, width: int, tracker: AnsiCodeTracker | None = None) -> list[str]:
    if not line:
        return [""]
    if visible_width(line) <= width:
        return [line]

    if tracker is None:
        tracker = AnsiCodeTracker()

    wrapped: list[str] = []
    tokens = _split_into_tokens_with_ansi(line)
    current_line = ""
    current_visible = 0

    for token in tokens:
        token_visible = visible_width(token)
        is_whitespace = token.strip() == ""

        if token_visible > width and not is_whitespace:
            if current_line:
                reset = tracker.get_line_end_reset()
                if reset:
                    current_line += reset
                wrapped.append(current_line)
                current_line = ""
                current_visible = 0
            broken = _break_long_word(token, width, tracker)
            wrapped.extend(broken[:-1])
            current_line = broken[-1]
            current_visible = visible_width(current_line)
            continue

        total_needed = current_visible + token_visible
        if total_needed > width and current_visible > 0:
            line_to_wrap = current_line.rstrip()
            reset = tracker.get_line_end_reset()
            if reset:
                line_to_wrap += reset
            wrapped.append(line_to_wrap)
            if is_whitespace:
                current_line = tracker.get_active_codes()
                current_visible = 0
            else:
                current_line = tracker.get_active_codes() + token
                current_visible = token_visible
        else:
            current_line += token
            current_visible += token_visible

        _update_tracker_from_text(token, tracker)

    if current_line:
        wrapped.append(current_line)

    return [ln.rstrip() for ln in wrapped] if wrapped else [""]


# ─────────────────────────────────────────────────────────────────────────────
# Background application
# ─────────────────────────────────────────────────────────────────────────────

def apply_background_to_line(
    line: str,
    width: int,
    bg_fn: "Callable[[str], str]",  # noqa: F821
) -> str:
    """Pad line to width and apply background color function."""
    from typing import Callable  # noqa: F401
    visible_len = visible_width(line)
    padding = max(0, width - visible_len)
    return bg_fn(line + " " * padding)


# ─────────────────────────────────────────────────────────────────────────────
# Truncation — mirrors truncateToWidth()
# ─────────────────────────────────────────────────────────────────────────────

def truncate_to_width(
    text: str,
    max_width: int,
    ellipsis: str = "...",
    pad: bool = False,
) -> str:
    """
    Truncate text to max_width columns, adding ellipsis if needed.
    ANSI codes are preserved but don't count toward width.
    Mirrors truncateToWidth() in utils.ts.
    """
    text_visible = visible_width(text)
    if text_visible <= max_width:
        if pad:
            return text + " " * (max_width - text_visible)
        return text

    ellipsis_width = visible_width(ellipsis)
    target_width = max_width - ellipsis_width
    if target_width <= 0:
        return ellipsis[:max_width]

    result = ""
    current_width = 0
    i = 0
    while i < len(text):
        ansi = extract_ansi_code(text, i)
        if ansi:
            result += ansi.code
            i += ansi.length
            continue

        end = i
        while end < len(text) and not extract_ansi_code(text, end):
            end += 1

        for g in _segment_graphemes(text[i:end]):
            gw = _grapheme_width(g)
            if current_width + gw > target_width:
                i = end  # break outer
                break
            result += g
            current_width += gw
        i = end

    truncated = f"{result}\x1b[0m{ellipsis}"
    if pad:
        tw = visible_width(truncated)
        return truncated + " " * max(0, max_width - tw)
    return truncated


# ─────────────────────────────────────────────────────────────────────────────
# Column slicing — mirrors sliceByColumn / sliceWithWidth
# ─────────────────────────────────────────────────────────────────────────────

class _SliceResult(NamedTuple):
    text: str
    width: int


def slice_with_width(line: str, start_col: int, length: int, strict: bool = False) -> _SliceResult:
    """
    Extract visible columns [start_col, start_col+length) from a line.
    Returns (text, actual_width). Mirrors sliceWithWidth() in utils.ts.
    """
    if length <= 0:
        return _SliceResult("", 0)

    end_col = start_col + length
    result = ""
    result_width = 0
    current_col = 0
    i = 0
    pending_ansi = ""

    while i < len(line):
        ansi = extract_ansi_code(line, i)
        if ansi:
            if start_col <= current_col < end_col:
                result += ansi.code
            elif current_col < start_col:
                pending_ansi += ansi.code
            i += ansi.length
            continue

        end = i
        while end < len(line) and not extract_ansi_code(line, end):
            end += 1

        for g in _segment_graphemes(line[i:end]):
            w = _grapheme_width(g)
            in_range = start_col <= current_col < end_col
            fits = not strict or (current_col + w <= end_col)
            if in_range and fits:
                if pending_ansi:
                    result += pending_ansi
                    pending_ansi = ""
                result += g
                result_width += w
            current_col += w
            if current_col >= end_col:
                break

        i = end
        if current_col >= end_col:
            break

    return _SliceResult(result, result_width)


def slice_by_column(line: str, start_col: int, length: int, strict: bool = False) -> str:
    """Extract a column range from a line. Mirrors sliceByColumn() in utils.ts."""
    return slice_with_width(line, start_col, length, strict).text


# ─────────────────────────────────────────────────────────────────────────────
# extract_segments — mirrors extractSegments() for overlay compositing
# ─────────────────────────────────────────────────────────────────────────────

class _SegmentResult(NamedTuple):
    before: str
    before_width: int
    after: str
    after_width: int


# Pooled tracker (avoids allocation per call)
_pooled_style_tracker = AnsiCodeTracker()


def extract_segments(
    line: str,
    before_end: int,
    after_start: int,
    after_len: int,
    strict_after: bool = False,
) -> _SegmentResult:
    """
    Extract 'before' and 'after' segments in a single pass for overlay compositing.
    Preserves styling from before the overlay that should affect content after it.
    Mirrors extractSegments() in utils.ts.
    """
    before = ""
    before_width = 0
    after = ""
    after_width = 0
    current_col = 0
    i = 0
    pending_ansi_before = ""
    after_started = False
    after_end = after_start + after_len

    _pooled_style_tracker.clear()

    while i < len(line):
        ansi = extract_ansi_code(line, i)
        if ansi:
            _pooled_style_tracker.process(ansi.code)
            if current_col < before_end:
                pending_ansi_before += ansi.code
            elif after_start <= current_col < after_end and after_started:
                after += ansi.code
            i += ansi.length
            continue

        end = i
        while end < len(line) and not extract_ansi_code(line, end):
            end += 1

        for g in _segment_graphemes(line[i:end]):
            w = _grapheme_width(g)

            if current_col < before_end:
                if pending_ansi_before:
                    before += pending_ansi_before
                    pending_ansi_before = ""
                before += g
                before_width += w
            elif after_start <= current_col < after_end:
                fits = not strict_after or (current_col + w <= after_end)
                if fits:
                    if not after_started:
                        after += _pooled_style_tracker.get_active_codes()
                        after_started = True
                    after += g
                    after_width += w

            current_col += w
            if after_len <= 0:
                if current_col >= before_end:
                    break
            else:
                if current_col >= after_end:
                    break

        i = end
        if after_len <= 0:
            if current_col >= before_end:
                break
        else:
            if current_col >= after_end:
                break

    return _SegmentResult(before, before_width, after, after_width)


# ─────────────────────────────────────────────────────────────────────────────
# Misc character utilities
# ─────────────────────────────────────────────────────────────────────────────

_PUNCTUATION_RE = re.compile(r"[(){}\[\]<>.,;:'\"!?+\-=*/\\|&%^$#@~`]")


def is_whitespace_char(ch: str) -> bool:
    return ch.isspace()


def is_punctuation_char(ch: str) -> bool:
    return bool(_PUNCTUATION_RE.match(ch))
