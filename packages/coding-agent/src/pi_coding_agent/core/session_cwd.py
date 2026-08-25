"""
Session working-directory checks — mirrors packages/coding-agent/src/core/session-cwd.ts
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


class SessionCwdSource(Protocol):
    def get_cwd(self) -> str: ...
    def get_session_file(self) -> str | None: ...


@dataclass
class SessionCwdIssue:
    session_cwd: str
    fallback_cwd: str
    session_file: str | None = None


def get_missing_session_cwd_issue(
    session_manager: SessionCwdSource,
    fallback_cwd: str,
) -> SessionCwdIssue | None:
    session_file = session_manager.get_session_file()
    if not session_file:
        return None
    session_cwd = session_manager.get_cwd()
    if not session_cwd or os.path.exists(session_cwd):
        return None
    return SessionCwdIssue(
        session_file=session_file,
        session_cwd=session_cwd,
        fallback_cwd=fallback_cwd,
    )


def format_missing_session_cwd_error(issue: SessionCwdIssue) -> str:
    session_file = f"\nSession file: {issue.session_file}" if issue.session_file else ""
    return (
        f"Stored session working directory does not exist: {issue.session_cwd}"
        f"{session_file}\nCurrent working directory: {issue.fallback_cwd}"
    )


def format_missing_session_cwd_prompt(issue: SessionCwdIssue) -> str:
    return (
        f"cwd from session file does not exist\n{issue.session_cwd}\n\n"
        f"continue in current cwd\n{issue.fallback_cwd}"
    )


class MissingSessionCwdError(Exception):
    def __init__(self, issue: SessionCwdIssue) -> None:
        super().__init__(format_missing_session_cwd_error(issue))
        self.issue = issue
        self.name = "MissingSessionCwdError"


def assert_session_cwd_exists(session_manager: SessionCwdSource, fallback_cwd: str) -> None:
    issue = get_missing_session_cwd_issue(session_manager, fallback_cwd)
    if issue:
        raise MissingSessionCwdError(issue)
