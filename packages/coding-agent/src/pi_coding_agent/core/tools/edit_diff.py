"""
Shared diff computation utilities for the edit tool.

Used by both edit.py (for execution) and interactive mode (for preview rendering).

Mirrors core/tools/edit-diff.ts
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Line ending utilities
# ---------------------------------------------------------------------------

def detect_line_ending(content: str) -> str:
    """Detect whether content uses CRLF or LF line endings."""
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1:
        return "\n"
    if crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"


def normalize_to_lf(text: str) -> str:
    """Normalize all line endings to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    """Restore CRLF line endings if the original file used them."""
    if ending == "\r\n":
        return text.replace("\n", "\r\n")
    return text


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fuzzy matching.

    Applies progressive transformations:
    - Strip trailing whitespace from each line
    - Normalize smart quotes to ASCII
    - Normalize Unicode dashes/hyphens to ASCII hyphen
    - Normalize special Unicode spaces to regular space
    """
    lines = [line.rstrip() for line in text.split("\n")]
    result = "\n".join(lines)
    # Smart single quotes
    result = result.replace("\u2018", "'").replace("\u2019", "'").replace("\u201A", "'").replace("\u201B", "'")
    # Smart double quotes
    result = result.replace("\u201C", '"').replace("\u201D", '"').replace("\u201E", '"').replace("\u201F", '"')
    # Various dashes
    for ch in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212":
        result = result.replace(ch, "-")
    # Special spaces
    for ch in "\u00A0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u202F\u205F\u3000":
        result = result.replace(ch, " ")
    return result


@dataclass
class FuzzyMatchResult:
    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool
    content_for_replacement: str


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    """Find old_text in content, trying exact match first, then fuzzy match."""
    exact_idx = content.find(old_text)
    if exact_idx != -1:
        return FuzzyMatchResult(
            found=True,
            index=exact_idx,
            match_length=len(old_text),
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old_text)
    fuzzy_idx = fuzzy_content.find(fuzzy_old)

    if fuzzy_idx == -1:
        return FuzzyMatchResult(
            found=False,
            index=-1,
            match_length=0,
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    return FuzzyMatchResult(
        found=True,
        index=fuzzy_idx,
        match_length=len(fuzzy_old),
        used_fuzzy_match=True,
        content_for_replacement=fuzzy_content,
    )


# ---------------------------------------------------------------------------
# BOM handling
# ---------------------------------------------------------------------------

def strip_bom(content: str) -> tuple[str, str]:
    """Strip UTF-8 BOM if present. Returns (bom, text_without_bom)."""
    if content.startswith("\uFEFF"):
        return "\uFEFF", content[1:]
    return "", content


# ---------------------------------------------------------------------------
# Diff generation
# ---------------------------------------------------------------------------

@dataclass
class EditDiffResult:
    diff: str
    first_changed_line: int | None


@dataclass
class EditDiffError:
    error: str


def generate_diff_string(
    old_content: str,
    new_content: str,
    context_lines: int = 4,
) -> EditDiffResult:
    """Generate a unified diff string with line numbers and context."""
    old_lines = old_content.splitlines(keepends=False)
    new_lines = new_content.splitlines(keepends=False)

    max_line = max(len(old_lines), len(new_lines), 1)
    line_num_width = len(str(max_line))

    output: list[str] = []
    first_changed_line: int | None = None

    # Use Python's difflib to get structured differences
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    opcodes = matcher.get_opcodes()

    old_line = 1
    new_line = 1

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            lines = old_lines[i1:i2]
            count = len(lines)
            old_line += count
            new_line += count
            # Context lines handled by surrounding changes
        elif tag in ("replace", "delete", "insert"):
            if first_changed_line is None:
                first_changed_line = new_line
            for l in old_lines[i1:i2]:
                ln = str(old_line).rjust(line_num_width)
                output.append(f"-{ln} {l}")
                old_line += 1
            for l in new_lines[j1:j2]:
                ln = str(new_line).rjust(line_num_width)
                output.append(f"+{ln} {l}")
                new_line += 1

    # If there were no changes, return empty diff
    if not output and old_content == new_content:
        return EditDiffResult(diff="", first_changed_line=None)

    # For a richer diff with context, use unified_diff
    diff_lines = list(difflib.unified_diff(
        old_lines,
        new_lines,
        lineterm="",
        n=context_lines,
    ))
    # Convert to our numbered format
    output_numbered: list[str] = []
    first_line: int | None = None
    old_ln = 1
    new_ln = 1

    for line in diff_lines:
        if line.startswith("@@"):
            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            import re
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if m:
                old_ln = int(m.group(1))
                new_ln = int(m.group(2))
            output_numbered.append(f" {''.rjust(line_num_width)} ...")
        elif line.startswith("+") and not line.startswith("+++"):
            ln = str(new_ln).rjust(line_num_width)
            output_numbered.append(f"+{ln} {line[1:]}")
            if first_line is None:
                first_line = new_ln
            new_ln += 1
        elif line.startswith("-") and not line.startswith("---"):
            ln = str(old_ln).rjust(line_num_width)
            output_numbered.append(f"-{ln} {line[1:]}")
            old_ln += 1
        elif not line.startswith("---") and not line.startswith("+++"):
            ln = str(old_ln).rjust(line_num_width)
            output_numbered.append(f" {ln} {line[1:]}")
            old_ln += 1
            new_ln += 1

    return EditDiffResult(
        diff="\n".join(output_numbered),
        first_changed_line=first_line,
    )


async def compute_edit_diff(
    path: str,
    old_text: str,
    new_text: str,
    cwd: str,
) -> EditDiffResult | EditDiffError:
    """Compute the diff for an edit operation without applying it.

    Used for preview rendering in the TUI before the tool executes.
    """
    from pi_coding_agent.core.tools.path_utils import resolve_to_cwd

    absolute_path = resolve_to_cwd(path, cwd)

    try:
        if not os.path.isfile(absolute_path):
            return EditDiffError(error=f"File not found: {path}")

        with open(absolute_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        _bom, content = strip_bom(raw_content)
        normalized = normalize_to_lf(content)
        old_norm = normalize_to_lf(old_text)
        new_norm = normalize_to_lf(new_text)

        match = fuzzy_find_text(normalized, old_norm)
        if not match.found:
            return EditDiffError(
                error=(
                    f"Could not find the exact text in {path}. "
                    "The old text must match exactly including all whitespace and newlines."
                )
            )

        base = match.content_for_replacement
        fuzzy_old = normalize_for_fuzzy_match(old_norm)
        occurrences = base.count(fuzzy_old) if match.used_fuzzy_match else normalized.count(old_norm)
        if occurrences > 1:
            return EditDiffError(
                error=(
                    f"Found {occurrences} occurrences of the text in {path}. "
                    "The text must be unique. Please provide more context to make it unique."
                )
            )

        new_content = base[: match.index] + new_norm + base[match.index + match.match_length :]
        if base == new_content:
            return EditDiffError(
                error=f"No changes would be made to {path}. The replacement produces identical content."
            )

        return generate_diff_string(base, new_content)

    except Exception as exc:
        return EditDiffError(error=str(exc))
