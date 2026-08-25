"""Shared edit/diff utilities — mirrors harness/tools/edit-diff.ts"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal


def detect_line_ending(content: str) -> Literal["\r\n", "\n"]:
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1:
        return "\n"
    if crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: Literal["\r\n", "\n"]) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def normalize_for_fuzzy_match(text: str) -> str:
    result = unicodedata.normalize("NFKC", text)
    result = "\n".join(line.rstrip() for line in result.split("\n"))
    result = re.sub(r"[\u2018\u2019\u201A\u201B]", "'", result)
    result = re.sub(r"[\u201C\u201D\u201E\u201F]", '"', result)
    result = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", result)
    result = re.sub(r"[\u00A0\u2002-\u200A\u202F\u205F\u3000]", " ", result)
    return result


def _split_lines_with_endings(content: str) -> list[str]:
    return re.findall(r"[^\n]*\n|[^\n]+", content)


@dataclass
class _LineSpan:
    start: int
    end: int


@dataclass
class TextReplacement:
    match_index: int
    match_length: int
    new_text: str


def _get_line_spans(content: str) -> list[_LineSpan]:
    offset = 0
    spans: list[_LineSpan] = []
    for line in _split_lines_with_endings(content):
        spans.append(_LineSpan(offset, offset + len(line)))
        offset += len(line)
    return spans


def _get_replacement_line_range(lines: list[_LineSpan], replacement: TextReplacement) -> tuple[int, int]:
    replacement_start = replacement.match_index
    replacement_end = replacement.match_index + replacement.match_length
    start_line = next(
        (i for i, line in enumerate(lines) if replacement_start >= line.start and replacement_start < line.end),
        -1,
    )
    if start_line == -1:
        raise ValueError("Replacement range is outside the base content.")
    end_line = start_line
    while end_line < len(lines) and lines[end_line].end < replacement_end:
        end_line += 1
    if end_line >= len(lines):
        raise ValueError("Replacement range is outside the base content.")
    return start_line, end_line + 1


def apply_replacements(content: str, replacements: list[TextReplacement], offset: int = 0) -> str:
    result = content
    for replacement in reversed(replacements):
        match_index = replacement.match_index - offset
        result = (
            result[:match_index]
            + replacement.new_text
            + result[match_index + replacement.match_length :]
        )
    return result


def apply_replacements_preserving_unchanged_lines(
    original_content: str,
    base_content: str,
    replacements: list[TextReplacement],
) -> str:
    original_lines = _split_lines_with_endings(original_content)
    base_lines = _get_line_spans(base_content)
    if len(original_lines) != len(base_lines):
        raise ValueError("Cannot preserve unchanged lines because the base content has a different line count.")

    groups: list[dict[str, object]] = []
    for replacement in sorted(replacements, key=lambda item: item.match_index):
        start_line, end_line = _get_replacement_line_range(base_lines, replacement)
        current = groups[-1] if groups else None
        if current and start_line < int(current["end_line"]):
            current["end_line"] = max(int(current["end_line"]), end_line)
            current["replacements"].append(replacement)  # type: ignore[union-attr]
            continue
        groups.append({"start_line": start_line, "end_line": end_line, "replacements": [replacement]})

    original_line_index = 0
    result = ""
    for group in groups:
        start_line = int(group["start_line"])
        end_line = int(group["end_line"])
        result += "".join(original_lines[original_line_index:start_line])
        group_start = base_lines[start_line].start
        group_end = base_lines[end_line - 1].end
        result += apply_replacements(
            base_content[group_start:group_end],
            list(group["replacements"]),  # type: ignore[arg-type]
            group_start,
        )
        original_line_index = end_line
    result += "".join(original_lines[original_line_index:])
    return result


@dataclass
class FuzzyMatchResult:
    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool
    content_for_replacement: str


@dataclass
class Edit:
    old_text: str
    new_text: str


@dataclass
class AppliedEditsResult:
    base_content: str
    new_content: str


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    exact_index = content.find(old_text)
    if exact_index != -1:
        return FuzzyMatchResult(True, exact_index, len(old_text), False, content)
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old)
    if fuzzy_index == -1:
        return FuzzyMatchResult(False, -1, 0, False, content)
    return FuzzyMatchResult(True, fuzzy_index, len(fuzzy_old), True, fuzzy_content)


def strip_bom(content: str) -> dict[str, str]:
    if content.startswith("\ufeff"):
        return {"bom": "\ufeff", "text": content[1:]}
    return {"bom": "", "text": content}


def _count_occurrences(content: str, old_text: str) -> int:
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old_text)
    return len(fuzzy_content.split(fuzzy_old)) - 1


def apply_edits_to_normalized_content(normalized_content: str, edits: list[Edit], path: str) -> AppliedEditsResult:
    normalized_edits = [Edit(normalize_to_lf(edit.old_text), normalize_to_lf(edit.new_text)) for edit in edits]
    for index, edit in enumerate(normalized_edits):
        if not edit.old_text:
            if len(normalized_edits) == 1:
                raise ValueError(f"oldText must not be empty in {path}.")
            raise ValueError(f"edits[{index}].oldText must not be empty in {path}.")

    used_fuzzy = any(fuzzy_find_text(normalized_content, edit.old_text).used_fuzzy_match for edit in normalized_edits)
    replacement_base = normalize_for_fuzzy_match(normalized_content) if used_fuzzy else normalized_content
    matched: list[TextReplacement] = []
    for index, edit in enumerate(normalized_edits):
        match = fuzzy_find_text(replacement_base, edit.old_text)
        if not match.found:
            if len(normalized_edits) == 1:
                raise ValueError(
                    f"Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
                )
            raise ValueError(
                f"Could not find edits[{index}] in {path}. The oldText must match exactly including all whitespace and newlines."
            )
        occurrences = _count_occurrences(replacement_base, edit.old_text)
        if occurrences > 1:
            if len(normalized_edits) == 1:
                raise ValueError(
                    f"Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
                )
            raise ValueError(
                f"Found {occurrences} occurrences of edits[{index}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
            )
        matched.append(TextReplacement(match.index, match.match_length, edit.new_text))

    matched.sort(key=lambda item: item.match_index)
    for previous, current in zip(matched, matched[1:]):
        if previous.match_index + previous.match_length > current.match_index:
            raise ValueError(
                f"edits overlap in {path}. Merge them into one edit or target disjoint regions."
            )

    new_content = (
        apply_replacements_preserving_unchanged_lines(normalized_content, replacement_base, matched)
        if used_fuzzy
        else apply_replacements(replacement_base, matched)
    )
    if normalized_content == new_content:
        raise ValueError(
            f"No changes made to {path}. The replacement produced identical content."
            if len(normalized_edits) == 1
            else f"No changes made to {path}. The replacements produced identical content."
        )
    return AppliedEditsResult(normalized_content, new_content)


def generate_unified_patch(path: str, old_content: str, new_content: str, context_lines: int = 4) -> str:
    return "".join(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
            n=context_lines,
        )
    )


def generate_diff_string(
    old_content: str,
    new_content: str,
    context_lines: int = 4,
) -> dict[str, object]:
    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")
    max_line = max(len(old_lines), len(new_lines), 1)
    width = len(str(max_line))
    output: list[str] = []
    first_changed: int | None = None
    old_ln = 1
    new_ln = 1
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    last_was_change = False
    opcodes = matcher.get_opcodes()
    for index, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag in ("replace", "delete", "insert"):
            if first_changed is None:
                first_changed = new_ln
            for line in old_lines[i1:i2]:
                output.append(f"-{str(old_ln).rjust(width)} {line}")
                old_ln += 1
            for line in new_lines[j1:j2]:
                output.append(f"+{str(new_ln).rjust(width)} {line}")
                new_ln += 1
            last_was_change = True
            continue
        raw = old_lines[i1:i2]
        next_is_change = index < len(opcodes) - 1 and opcodes[index + 1][0] != "equal"
        if last_was_change and next_is_change:
            shown = raw if len(raw) <= context_lines * 2 else raw[:context_lines] + [None] + raw[-context_lines:]
            for line in shown:
                if line is None:
                    output.append(f" {'':>{width}} ...")
                    skipped = len(raw) - context_lines * 2
                    old_ln += skipped
                    new_ln += skipped
                    continue
                output.append(f" {str(old_ln).rjust(width)} {line}")
                old_ln += 1
                new_ln += 1
        elif last_was_change:
            for line in raw[:context_lines]:
                output.append(f" {str(old_ln).rjust(width)} {line}")
                old_ln += 1
                new_ln += 1
            skipped = max(0, len(raw) - context_lines)
            if skipped:
                output.append(f" {'':>{width}} ...")
                old_ln += skipped
                new_ln += skipped
        elif next_is_change:
            skipped = max(0, len(raw) - context_lines)
            if skipped:
                output.append(f" {'':>{width}} ...")
                old_ln += skipped
                new_ln += skipped
            for line in raw[skipped:]:
                output.append(f" {str(old_ln).rjust(width)} {line}")
                old_ln += 1
                new_ln += 1
        else:
            old_ln += len(raw)
            new_ln += len(raw)
        last_was_change = False
    return {"diff": "\n".join(output), "first_changed_line": first_changed}


# Back-compat aliases used by earlier harness tests
def apply_edit(content: str, old_text: str, new_text: str) -> str:
    return apply_edits_to_normalized_content(content, [Edit(old_text, new_text)], "file").new_content


def format_diff(before: str, after: str, path: str = "file") -> str:
    return generate_unified_patch(path, before, after)
