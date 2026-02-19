"""
Changelog parsing utilities.

Parses CHANGELOG.md files in the conventional "Keep a Changelog" format,
extracts version entries, and compares version strings.

Mirrors utils/changelog.ts
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ChangelogEntry:
    """A single version entry in a changelog."""

    version: str
    date: str | None
    content: str


def parse_changelog(text: str) -> list[ChangelogEntry]:
    """Parse a CHANGELOG.md string into a list of ChangelogEntry objects."""
    entries: list[ChangelogEntry] = []
    # Match headings like "## [1.2.3] - 2024-01-01" or "## 1.2.3"
    pattern = re.compile(
        r"^## \[?([^\]\n]+)\]?"        # version
        r"(?:\s*-\s*(\d{4}-\d{2}-\d{2}))?"  # optional date
        r"\s*$",
        re.MULTILINE,
    )

    matches = list(pattern.finditer(text))
    for idx, match in enumerate(matches):
        version = match.group(1).strip()
        date = match.group(2)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        entries.append(ChangelogEntry(version=version, date=date, content=content))

    return entries


def compare_versions(a: str, b: str) -> int:
    """Compare two semver-like version strings.

    Returns:
        Negative if a < b, 0 if a == b, positive if a > b.
    """
    def _split(v: str) -> list[int]:
        # Take only the numeric prefix, e.g. "1.2.3-alpha" -> [1, 2, 3]
        parts = re.split(r"[.\-]", v)
        result = []
        for part in parts:
            if part.isdigit():
                result.append(int(part))
            else:
                break
        return result

    a_parts = _split(a)
    b_parts = _split(b)

    for ap, bp in zip(a_parts, b_parts):
        if ap != bp:
            return ap - bp

    return len(a_parts) - len(b_parts)


def get_new_entries(
    old_version: str | None,
    entries: list[ChangelogEntry],
) -> list[ChangelogEntry]:
    """Return changelog entries newer than ``old_version``.

    If ``old_version`` is None, returns all entries.
    """
    if old_version is None:
        return list(entries)

    new_entries: list[ChangelogEntry] = []
    for entry in entries:
        if compare_versions(entry.version, old_version) > 0:
            new_entries.append(entry)

    return new_entries
