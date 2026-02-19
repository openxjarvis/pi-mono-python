"""
Footer data provider — mirrors packages/coding-agent/src/core/footer-data-provider.ts

Provides git branch watching, extension statuses, and available provider count
for the interactive TUI footer.
"""
from __future__ import annotations

import asyncio
import os
from typing import Callable


def _find_git_head_path(cwd: str | None = None) -> str | None:
    """
    Walk up from cwd to find .git/HEAD (handles worktrees).
    Mirrors findGitHeadPath() in TypeScript.
    """
    directory = cwd or os.getcwd()
    while True:
        git_path = os.path.join(directory, ".git")
        if os.path.exists(git_path):
            try:
                if os.path.isfile(git_path):
                    # Worktree: .git is a file like "gitdir: ../"
                    with open(git_path, encoding="utf-8") as f:
                        content = f.read().strip()
                    if content.startswith("gitdir: "):
                        git_dir = content[8:]
                        head_path = os.path.normpath(os.path.join(directory, git_dir, "HEAD"))
                        if os.path.exists(head_path):
                            return head_path
                elif os.path.isdir(git_path):
                    head_path = os.path.join(git_path, "HEAD")
                    if os.path.exists(head_path):
                        return head_path
            except Exception:
                return None
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


class FooterDataProvider:
    """
    Provides git branch and extension statuses for the footer.
    Mirrors FooterDataProvider in TypeScript.

    Uses asyncio polling instead of fs.watch (more portable).
    """

    def __init__(self, cwd: str | None = None) -> None:
        self._extension_statuses: dict[str, str] = {}
        self._unset_sentinel = object()
        self._cached_branch: object = self._unset_sentinel
        self._branch_cache: object = self._unset_sentinel
        self._branch_change_callbacks: list[Callable[[], None]] = []
        self._available_provider_count = 0
        self._cwd = cwd
        self._poll_task: asyncio.Task | None = None
        self._last_head_content: str | None = None
        self._start_polling()

    def _start_polling(self) -> None:
        """Start background polling for git HEAD changes."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._poll_task = loop.create_task(self._poll_git_head())
        except RuntimeError:
            pass

    async def _poll_git_head(self) -> None:
        """Poll .git/HEAD for branch changes every 2 seconds."""
        while True:
            try:
                await asyncio.sleep(2)
                head_path = _find_git_head_path(self._cwd)
                if head_path:
                    try:
                        with open(head_path, encoding="utf-8") as f:
                            content = f.read().strip()
                        if content != self._last_head_content:
                            self._last_head_content = content
                            self._branch_cache = self._unset_sentinel  # invalidate
                            for cb in list(self._branch_change_callbacks):
                                try:
                                    cb()
                                except Exception:
                                    pass
                    except Exception:
                        pass
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def get_git_branch(self) -> str | None:
        """
        Get current git branch. Returns None if not in a repo,
        'detached' if in detached HEAD state.
        Mirrors getGitBranch() in TypeScript.
        """
        if self._branch_cache is not self._unset_sentinel:
            return self._branch_cache  # type: ignore

        head_path = _find_git_head_path(self._cwd)
        if not head_path:
            self._branch_cache = None
            return None

        try:
            with open(head_path, encoding="utf-8") as f:
                content = f.read().strip()
            self._last_head_content = content
            if content.startswith("ref: refs/heads/"):
                self._branch_cache = content[16:]
            else:
                self._branch_cache = "detached"
        except Exception:
            self._branch_cache = None

        return self._branch_cache  # type: ignore

    def get_extension_statuses(self) -> dict[str, str]:
        """Get extension status texts (set via ctx.ui.setStatus())."""
        return dict(self._extension_statuses)

    def on_branch_change(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Subscribe to git branch changes. Returns unsubscribe function."""
        self._branch_change_callbacks.append(callback)

        def unsubscribe():
            try:
                self._branch_change_callbacks.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def set_extension_status(self, key: str, text: str | None) -> None:
        """Internal: set extension status."""
        if text is None:
            self._extension_statuses.pop(key, None)
        else:
            self._extension_statuses[key] = text

    def clear_extension_statuses(self) -> None:
        """Internal: clear all extension statuses."""
        self._extension_statuses.clear()

    def get_available_provider_count(self) -> int:
        """Number of unique providers with available models."""
        return self._available_provider_count

    def set_available_provider_count(self, count: int) -> None:
        """Internal: update available provider count."""
        self._available_provider_count = count

    def dispose(self) -> None:
        """Cleanup: stop polling and clear callbacks."""
        if self._poll_task:
            self._poll_task.cancel()
            self._poll_task = None
        self._branch_change_callbacks.clear()
