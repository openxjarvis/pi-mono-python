"""
Filesystem watch helpers — mirrors packages/coding-agent/src/utils/fs-watch.ts
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

FS_WATCH_RETRY_DELAY_MS = 5000


def close_watcher(watcher: Any | None) -> None:
    if watcher is None:
        return
    close = getattr(watcher, "close", None) or getattr(watcher, "stop", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def watch_with_error_handler(
    path: str,
    listener: Callable[..., Any],
    on_error: Callable[[], None],
) -> Any | None:
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class _Handler(FileSystemEventHandler):
            def on_any_event(self, event: Any) -> None:
                listener(event)

        observer = Observer()
        observer.schedule(_Handler(), path, recursive=False)
        observer.start()
        return observer
    except Exception:
        on_error()
        return None
