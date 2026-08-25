"""
Windows self-update quarantine — omitted on purpose.

TypeScript ``utils/windows-self-update.ts`` is a Bun/Node installer helper.
The four-package correspondence plan keeps it out of Python (no Bun SEA,
no npm installer, no Windows quarantine updater).
"""

WINDOWS_SELF_UPDATE_SUPPORTED = False
