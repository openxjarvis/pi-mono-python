"""
Session persistence — mirrors packages/coding-agent/src/core/session-manager.ts

JSONL file format with tree structure supporting branching sessions.
Each line is a JSON-encoded entry. The first line is the session header.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

CURRENT_SESSION_VERSION = 3

# Entry types (mirrors TypeScript session entry types)
SessionEntryType = Literal[
    "session",
    "message",
    "compaction",
    "branch_summary",
    "model_change",
    "thinking_level_change",
    "custom_message",
    "custom",
    "session_info",
    "label",
]


@dataclass
class SessionEntry:
    """A single entry in a JSONL session file."""
    id: str
    type: str
    timestamp: int
    parent_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionHeader:
    """Session file header (first JSONL line)."""
    type: str
    id: str
    timestamp: str
    cwd: str
    version: int = CURRENT_SESSION_VERSION
    parent_session: str | None = None


@dataclass
class SessionTreeNode:
    """Tree node for get_tree() - defensive copy of session structure."""
    entry: SessionEntry
    children: list["SessionTreeNode"] = field(default_factory=list)
    label: str | None = None


@dataclass
class SessionContext:
    """Built session context for the agent."""
    messages: list[dict[str, Any]]
    thinking_level: str
    model: dict[str, str] | None  # {"provider": ..., "model_id": ...}


@dataclass
class SessionInfo:
    """Metadata about a session."""
    session_id: str
    file_path: str
    created_at: int
    updated_at: int
    label: str | None = None
    entry_count: int = 0
    cwd_path: str = ""
    parent_session_path: str | None = None
    first_message: str = ""
    all_messages_text: str = ""

    @property
    def id(self) -> str:
        return self.session_id

    @property
    def path(self) -> str:
        return self.file_path

    @property
    def name(self) -> str:
        return self.label or self.session_id

    @property
    def cwd(self) -> str:
        return self.cwd_path or str(Path(self.file_path).parent)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone utility functions (exported for tests)
# ─────────────────────────────────────────────────────────────────────────────

def generate_id(existing_ids: set[str]) -> str:
    """Generate a unique 8-hex-char ID, checking for collisions."""
    for _ in range(100):
        candidate = str(uuid.uuid4()).replace("-", "")[:8]
        if candidate not in existing_ids:
            return candidate
    return str(uuid.uuid4())


def migrate_v1_to_v2(entries: list[dict[str, Any]]) -> None:
    """Migrate v1 → v2: add id/parentId tree structure. Mutates in place."""
    ids: set[str] = set()
    prev_id: str | None = None

    for entry in entries:
        if entry.get("type") == "session":
            entry["version"] = 2
            continue

        entry["id"] = generate_id(ids)
        ids.add(entry["id"])
        entry["parentId"] = prev_id
        prev_id = entry["id"]

        # Convert firstKeptEntryIndex → firstKeptEntryId for compaction
        if entry.get("type") == "compaction":
            idx = entry.pop("firstKeptEntryIndex", None)
            if isinstance(idx, int) and 0 <= idx < len(entries):
                target = entries[idx]
                if target.get("type") != "session":
                    entry["firstKeptEntryId"] = target.get("id")


def migrate_v2_to_v3(entries: list[dict[str, Any]]) -> None:
    """Migrate v2 → v3: rename hookMessage role to custom. Mutates in place."""
    for entry in entries:
        if entry.get("type") == "session":
            entry["version"] = 3
            continue
        if entry.get("type") == "message":
            msg = entry.get("message", {})
            if isinstance(msg, dict) and msg.get("role") == "hookMessage":
                msg["role"] = "custom"


def migrate_to_current_version(entries: list[dict[str, Any]]) -> bool:
    """Run all necessary migrations. Mutates in place. Returns True if any applied."""
    header = next((e for e in entries if e.get("type") == "session"), None)
    version = header.get("version", 1) if header else 1

    if version >= CURRENT_SESSION_VERSION:
        return False

    if version < 2:
        migrate_v1_to_v2(entries)
    if version < 3:
        migrate_v2_to_v3(entries)

    return True


# Alias for tests
def migrate_session_entries(entries: list[dict[str, Any]]) -> None:
    migrate_to_current_version(entries)


def get_latest_compaction_entry(entries: list[SessionEntry]) -> SessionEntry | None:
    """Get the most recent compaction entry, if any."""
    for entry in reversed(entries):
        if entry.type == "compaction":
            return entry
    return None


def build_session_context(
    entries: list[SessionEntry],
    leaf_id: str | None = None,
    by_id: dict[str, SessionEntry] | None = None,
) -> SessionContext:
    """
    Build the session context from entries using tree traversal.
    Mirrors buildSessionContext() in TypeScript.

    If leaf_id is provided, walks from that entry to root.
    Handles compaction and branch summaries along the path.
    """
    if by_id is None:
        by_id = {e.id: e for e in entries}

    # Find leaf
    leaf: SessionEntry | None = None
    if leaf_id is None:
        # Explicitly None — return empty (navigated before first entry)
        return SessionContext(messages=[], thinking_level="off", model=None)

    if leaf_id:
        leaf = by_id.get(leaf_id)
    if not leaf and entries:
        leaf = entries[-1]

    if not leaf:
        return SessionContext(messages=[], thinking_level="off", model=None)

    # Walk from leaf to root
    path: list[SessionEntry] = []
    current: SessionEntry | None = leaf
    while current:
        path.insert(0, current)
        current = by_id.get(current.parent_id) if current.parent_id else None

    # Extract settings and find compaction
    thinking_level = "off"
    model: dict[str, str] | None = None
    compaction: SessionEntry | None = None

    for entry in path:
        if entry.type == "thinking_level_change":
            thinking_level = entry.data.get("thinkingLevel") or entry.data.get("level", "off")
        elif entry.type == "model_change":
            model = {
                "provider": entry.data.get("provider", ""),
                "model_id": entry.data.get("modelId") or entry.data.get("model_id", ""),
            }
        elif entry.type == "message":
            msg = entry.data.get("message", {})
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                provider = msg.get("provider", "")
                model_id = msg.get("model", "")
                if provider:
                    model = {"provider": provider, "model_id": model_id}
        elif entry.type == "compaction":
            compaction = entry

    # Build messages list
    messages: list[dict[str, Any]] = []

    def append_message(entry: SessionEntry) -> None:
        if entry.type == "message":
            msg = entry.data.get("message", {})
            if isinstance(msg, dict):
                messages.append(msg)
        elif entry.type == "custom_message":
            content = entry.data.get("content", "")
            custom_type = entry.data.get("customType", "")
            display = entry.data.get("display", True)
            messages.append({
                "role": "custom",
                "customType": custom_type,
                "content": content,
                "display": display,
                "timestamp": entry.timestamp,
            })
        elif entry.type == "branch_summary":
            summary = entry.data.get("summary", "")
            if summary:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": f"[Branch summary: {summary}]"}],
                    "timestamp": entry.timestamp,
                })

    if compaction:
        first_kept = compaction.data.get("firstKeptEntryId")
        summary = compaction.data.get("summary", "")
        tokens_before = compaction.data.get("tokensBefore", 0)

        # Emit compaction summary
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": f"[Context compacted. Summary:\n{summary}]"}],
            "timestamp": compaction.timestamp,
            "_tokens_before": tokens_before,
        })

        comp_idx = next(
            (i for i, e in enumerate(path) if e.type == "compaction" and e.id == compaction.id),
            -1,
        )

        # Emit kept messages before compaction
        found_first_kept = False
        for i in range(comp_idx):
            entry = path[i]
            if entry.id == first_kept:
                found_first_kept = True
            if found_first_kept:
                append_message(entry)

        # Emit messages after compaction
        for i in range(comp_idx + 1, len(path)):
            append_message(path[i])
    else:
        for entry in path:
            append_message(entry)

    return SessionContext(messages=messages, thinking_level=thinking_level, model=model)


class SessionManager:
    """
    Manages session persistence in JSONL files.
    Mirrors SessionManager in TypeScript.

    Sessions are stored as JSONL files where each line is a JSON-encoded entry.
    The first line is the session header. Tree structure via parentId relationships.
    """

    def __init__(
        self,
        session_file: str | None = None,
        sessions_dir: str | None = None,
        cwd: str | None = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()

        if session_file:
            self._session_file_path = os.path.abspath(session_file)
            self.sessions_dir = os.path.dirname(self._session_file_path)
        else:
            self.sessions_dir = sessions_dir or self._default_sessions_dir()
            os.makedirs(self.sessions_dir, exist_ok=True)
            self._session_file_path: str | None = None

        self._entries: list[dict[str, Any]] = []
        self._header: dict[str, Any] | None = None
        self._leaf_id: str | None = None
        self._by_id: dict[str, dict[str, Any]] = {}
        self._dirty = False

        if self._session_file_path and os.path.exists(self._session_file_path):
            self._load_file()

    @staticmethod
    def _default_sessions_dir() -> str:
        return os.path.join(os.path.expanduser("~"), ".pi", "agent", "sessions")

    @staticmethod
    def _resolve_sessions_dir(cwd: str, session_dir: str | None = None) -> str:
        if session_dir:
            return os.path.abspath(session_dir)
        safe = f"--{cwd.lstrip('/').lstrip(os.sep).replace('/', '-').replace(os.sep, '-').replace(':', '-')}--"
        base = os.path.join(os.path.expanduser("~"), ".pi", "agent", "sessions", safe)
        os.makedirs(base, exist_ok=True)
        return base

    def _load_file(self) -> None:
        """Load entries from the session file."""
        if not self._session_file_path:
            return
        self._entries = []
        self._by_id = {}
        try:
            with open(self._session_file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if obj.get("type") == "session":
                            self._header = obj
                        else:
                            self._entries.append(obj)
                            if "id" in obj:
                                self._by_id[obj["id"]] = obj
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass

        # Run migrations if needed
        all_entries = ([self._header] if self._header else []) + self._entries
        if migrate_to_current_version(all_entries):
            self._persist_all()

    def _persist_all(self) -> None:
        """Rewrite entire session file after migration."""
        if not self._session_file_path:
            return
        lines = []
        if self._header:
            lines.append(json.dumps(self._header, ensure_ascii=False))
        for entry in self._entries:
            lines.append(json.dumps(entry, ensure_ascii=False))
        with open(self._session_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _append_raw(self, obj: dict[str, Any]) -> None:
        """Append a raw dict as a JSONL line."""
        path = self._session_file_path
        if not path:
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # ── Factory classmethods ──────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        cwd: str,
        session_dir: str | None = None,
        parent_session: str | None = None,
    ) -> "SessionManager":
        """Create a new session."""
        sessions_dir = cls._resolve_sessions_dir(cwd, session_dir)
        session_id = str(uuid.uuid4()).replace("-", "")[:8]
        session_file = os.path.join(sessions_dir, f"{session_id}.jsonl")
        mgr = cls(session_file=session_file, cwd=cwd)

        now_ms = int(time.time() * 1000)
        header: dict[str, Any] = {
            "type": "session",
            "id": session_id,
            "version": CURRENT_SESSION_VERSION,
            "timestamp": str(now_ms),
            "cwd": cwd,
        }
        if parent_session:
            header["parentSession"] = parent_session

        mgr._header = header
        mgr._append_raw(header)
        return mgr

    @classmethod
    def in_memory(cls, cwd: str | None = None) -> "SessionManager":
        """Create an in-memory session (uses temp sessions dir)."""
        actual_cwd = cwd or os.getcwd()
        tmp = os.path.join(os.path.expanduser("~"), ".pi", "agent", "sessions", "in-memory")
        os.makedirs(tmp, exist_ok=True)
        return cls.create(actual_cwd, session_dir=tmp)

    @classmethod
    def open(cls, path: str) -> "SessionManager":
        """Open an existing session file."""
        resolved = os.path.abspath(path)
        mgr = cls(session_file=resolved)
        return mgr

    @classmethod
    def continue_recent(cls, cwd: str, session_dir: str | None = None) -> "SessionManager":
        """Open the most recent session, or create a new one."""
        sessions = cls.list_sync(cwd, session_dir)
        if sessions:
            return cls.open(sessions[0].file_path)
        return cls.create(cwd, session_dir)

    @classmethod
    def fork_from(
        cls,
        source_path: str,
        target_cwd: str,
        session_dir: str | None = None,
    ) -> "SessionManager":
        """Fork a session by copying its entries into a new session."""
        source = cls.open(source_path)
        target = cls.create(target_cwd, session_dir, parent_session=source_path)
        # Copy source entries into target
        for entry in source._entries:
            target._entries.append(entry)
            target._append_raw(entry)
        return target

    @classmethod
    async def list(
        cls,
        cwd: str,
        session_dir: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[SessionInfo]:
        """List sessions (async, with optional progress callback)."""
        return cls.list_sync(cwd, session_dir, on_progress=on_progress)

    @classmethod
    def list_sync(
        cls,
        cwd: str,
        session_dir: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[SessionInfo]:
        """List sessions synchronously."""
        sessions_dir = cls._resolve_sessions_dir(cwd, session_dir)
        return cls._list_sessions_from_dir(sessions_dir, on_progress)

    @classmethod
    async def list_all(
        cls,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[SessionInfo]:
        """List all sessions across the global sessions directory."""
        roots = [os.path.join(os.path.expanduser("~"), ".pi", "agent", "sessions")]
        seen: set[str] = set()
        out: list[SessionInfo] = []
        for root in roots:
            if not os.path.isdir(root):
                continue
            for info in cls._list_sessions_from_dir(root, on_progress):
                if info.file_path not in seen:
                    seen.add(info.file_path)
                    out.append(info)
        out.sort(key=lambda s: s.updated_at, reverse=True)
        return out

    @classmethod
    def _list_sessions_from_dir(
        cls,
        sessions_dir: str,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[SessionInfo]:
        """List sessions from a specific directory."""
        if not os.path.isdir(sessions_dir):
            return []

        files = [
            os.path.join(sessions_dir, f)
            for f in os.listdir(sessions_dir)
            if f.endswith(".jsonl")
        ]
        total = len(files)
        out: list[SessionInfo] = []

        for i, fpath in enumerate(files):
            if on_progress:
                on_progress(i, total)
            info = cls._build_session_info_sync(fpath)
            if info:
                out.append(info)

        if on_progress:
            on_progress(total, total)

        out.sort(key=lambda s: s.updated_at, reverse=True)
        return out

    @classmethod
    def _build_session_info_sync(cls, file_path: str) -> SessionInfo | None:
        """Build SessionInfo from a session file path."""
        if not os.path.exists(file_path):
            return None

        try:
            stat = os.stat(file_path)
        except OSError:
            return None

        header: dict[str, Any] | None = None
        entries: list[dict[str, Any]] = []
        label: str | None = None
        cwd_path = ""
        parent_session = None
        first_message = ""
        all_messages: list[str] = []

        try:
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if obj.get("type") == "session":
                            header = obj
                            cwd_path = obj.get("cwd", "")
                            parent_session = obj.get("parentSession")
                        else:
                            entries.append(obj)
                            etype = obj.get("type")
                            if etype == "session_info":
                                label = obj.get("data", {}).get("name") or label
                            elif etype == "label":
                                lbl = obj.get("data", {}).get("label")
                                if lbl is not None:
                                    label = lbl
                            elif etype == "message":
                                msg = obj.get("data", {}).get("message", {})
                                if isinstance(msg, dict) and msg.get("role") == "user":
                                    text = ""
                                    content = msg.get("content", [])
                                    if isinstance(content, str):
                                        text = content
                                    elif isinstance(content, list):
                                        for block in content:
                                            if isinstance(block, dict) and block.get("type") == "text":
                                                text += block.get("text", "")
                                    if text:
                                        if not first_message:
                                            first_message = text[:200]
                                        all_messages.append(text[:100])
                    except json.JSONDecodeError:
                        pass
        except OSError:
            return None

        if header is None:
            return None

        session_id = header.get("id", Path(file_path).stem)
        created_at = int(header.get("timestamp", int(stat.st_ctime * 1000)))
        if isinstance(created_at, str):
            try:
                created_at = int(created_at)
            except (ValueError, TypeError):
                created_at = int(stat.st_ctime * 1000)

        return SessionInfo(
            session_id=session_id,
            file_path=file_path,
            created_at=created_at,
            updated_at=int(stat.st_mtime * 1000),
            label=label,
            entry_count=len(entries),
            cwd_path=cwd_path,
            parent_session_path=parent_session,
            first_message=first_message,
            all_messages_text=" ".join(all_messages),
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self._header.get("id", "") if self._header else ""

    def get_session_file(self) -> str | None:
        """Get the session file path."""
        return self._session_file_path

    def get_cwd(self) -> str:
        """Get the working directory."""
        if self._header:
            return self._header.get("cwd", self._cwd)
        return self._cwd

    def get_session_dir(self) -> str:
        """Get the sessions directory."""
        return self.sessions_dir

    def get_header(self) -> dict[str, Any] | None:
        """Get the session header."""
        return self._header

    def get_entries(self) -> list[SessionEntry]:
        """Get all session entries as SessionEntry objects."""
        return [
            SessionEntry(
                id=e.get("id", str(uuid.uuid4())),
                type=e.get("type", "unknown"),
                timestamp=e.get("timestamp", 0),
                parent_id=e.get("parentId") or e.get("parent_id"),
                data=e,
            )
            for e in self._entries
        ]

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        """Get a single entry by ID."""
        raw = self._by_id.get(entry_id)
        if not raw:
            return None
        return SessionEntry(
            id=raw.get("id", entry_id),
            type=raw.get("type", "unknown"),
            timestamp=raw.get("timestamp", 0),
            parent_id=raw.get("parentId") or raw.get("parent_id"),
            data=raw,
        )

    def get_leaf_id(self) -> str | None:
        """Get the current leaf entry ID (current tree position)."""
        return self._leaf_id

    def set_leaf_id(self, entry_id: str | None) -> None:
        """Set the current leaf entry ID."""
        self._leaf_id = entry_id

    def get_leaf_entry(self) -> SessionEntry | None:
        """Get the leaf entry (current position in tree)."""
        if self._leaf_id:
            return self.get_entry(self._leaf_id)
        entries = self.get_entries()
        return entries[-1] if entries else None

    def get_label(self, entry_id: str) -> str | None:
        """Get the resolved label for an entry ID."""
        label: str | None = None
        for raw in self._entries:
            if raw.get("type") == "label" and raw.get("targetId") == entry_id:
                label = raw.get("label")
        return label

    def get_branch(self) -> list[SessionEntry]:
        """Get all entries on the path from root to leaf."""
        leaf = self.get_leaf_entry()
        if not leaf:
            return []
        path: list[SessionEntry] = []
        current: SessionEntry | None = leaf
        while current:
            path.insert(0, current)
            current = self.get_entry(current.parent_id) if current.parent_id else None
        return path

    def get_session_name(self) -> str | None:
        """Get user-defined session name from session_info entries."""
        for raw in reversed(self._entries):
            if raw.get("type") == "session_info":
                name = raw.get("name") or raw.get("data", {}).get("name")
                if name:
                    return name
        return None

    def get_tree(self) -> list[SessionTreeNode]:
        """Return the full tree structure of entries."""
        entries = self.get_entries()
        by_id = {e.id: e for e in entries}
        nodes: dict[str, SessionTreeNode] = {}
        roots: list[SessionTreeNode] = []

        for entry in entries:
            node = SessionTreeNode(
                entry=entry,
                label=self.get_label(entry.id),
            )
            nodes[entry.id] = node

        for entry in entries:
            node = nodes[entry.id]
            if entry.parent_id and entry.parent_id in nodes:
                nodes[entry.parent_id].children.append(node)
            else:
                roots.append(node)

        return roots

    def build_context(self, leaf_id: str | None = None) -> SessionContext:
        """Build session context for the agent from the entry tree."""
        entries = self.get_entries()
        by_id = {e.id: e for e in entries}
        return build_session_context(entries, leaf_id or self._leaf_id, by_id)

    # ── Append methods ─────────────────────────────────────────────────────

    def _new_id(self) -> str:
        existing = {e.get("id", "") for e in self._entries}
        return generate_id(existing)

    def _make_entry(self, entry_type: str, extra: dict[str, Any]) -> dict[str, Any]:
        """Build an entry dict with id, type, timestamp, parentId."""
        leaf = self.get_leaf_entry()
        entry: dict[str, Any] = {
            "id": self._new_id(),
            "type": entry_type,
            "timestamp": int(time.time() * 1000),
            "parentId": leaf.id if leaf else None,
        }
        entry.update(extra)
        return entry

    def _append_entry(self, entry: dict[str, Any]) -> str:
        """Store and persist a new entry. Returns the entry ID."""
        self._entries.append(entry)
        if "id" in entry:
            self._by_id[entry["id"]] = entry
        self._leaf_id = entry.get("id")
        self._append_raw(entry)
        return entry["id"]

    def append_message(self, message: dict[str, Any], parent_id: str | None = None) -> str:
        """Append a message entry."""
        entry = {
            "id": self._new_id(),
            "type": "message",
            "timestamp": int(time.time() * 1000),
            "parentId": parent_id or (self.get_leaf_entry().id if self.get_leaf_entry() else None),
            "message": message,
        }
        return self._append_entry(entry)

    def append_model_change(self, model_id: str, provider: str) -> str:
        """Append a model change entry."""
        return self._append_entry(self._make_entry("model_change", {
            "provider": provider,
            "modelId": model_id,
        }))

    def append_thinking_level_change(self, level: str) -> str:
        """Append a thinking level change entry."""
        return self._append_entry(self._make_entry("thinking_level_change", {
            "thinkingLevel": level,
        }))

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int = 0,
        details: Any = None,
        from_hook: bool = False,
    ) -> str:
        """Append a compaction entry."""
        extra: dict[str, Any] = {
            "summary": summary,
            "firstKeptEntryId": first_kept_entry_id,
            "tokensBefore": tokens_before,
        }
        if details is not None:
            extra["details"] = details
        if from_hook:
            extra["fromHook"] = True
        return self._append_entry(self._make_entry("compaction", extra))

    def append_branch_summary(
        self,
        summary: str,
        from_id: str,
        details: Any = None,
        from_hook: bool = False,
    ) -> str:
        """Append a branch summary entry."""
        extra: dict[str, Any] = {
            "fromId": from_id,
            "summary": summary,
        }
        if details is not None:
            extra["details"] = details
        if from_hook:
            extra["fromHook"] = True
        return self._append_entry(self._make_entry("branch_summary", extra))

    def append_session_info(self, name: str | None = None) -> str:
        """Append a session_info entry (user-defined display name)."""
        extra: dict[str, Any] = {}
        if name is not None:
            extra["name"] = name
        return self._append_entry(self._make_entry("session_info", extra))

    def append_custom_message_entry(
        self,
        custom_type: str,
        content: Any,
        display: bool = True,
        details: Any = None,
    ) -> str:
        """
        Append a custom_message entry (injected into LLM context).
        Mirrors appendCustomMessageEntry() in TypeScript.
        """
        extra: dict[str, Any] = {
            "customType": custom_type,
            "content": content,
            "display": display,
        }
        if details is not None:
            extra["details"] = details
        return self._append_entry(self._make_entry("custom_message", extra))

    def append_custom_entry(
        self,
        custom_type: str,
        data: Any = None,
    ) -> str:
        """
        Append a custom entry (extension state storage, NOT in LLM context).
        Mirrors appendCustomEntry() in TypeScript.
        """
        extra: dict[str, Any] = {"customType": custom_type}
        if data is not None:
            extra["data"] = data
        return self._append_entry(self._make_entry("custom", extra))

    def append_label_change(self, target_id: str, label: str | None) -> str:
        """Append a label entry (bookmark/marker)."""
        return self._append_entry(self._make_entry("label", {
            "targetId": target_id,
            "label": label,
        }))

    def set_label(self, session_id: str, label: str) -> str:
        """Set session label via a label_change entry (legacy API compatibility)."""
        leaf = self._leaf_id or (self._entries[-1].get("id") if self._entries else None)
        target = leaf or session_id
        return self.append_label_change(target, label)

    # ── Branch / Fork ──────────────────────────────────────────────────────

    def branch(
        self,
        branch_point_id: str | None = None,
        cwd: str | None = None,
    ) -> "SessionManager":
        """
        Create a branched session from a specific entry point.
        Mirrors branch() in TypeScript.
        """
        parent_file = self._session_file_path or ""
        target_cwd = cwd or self.get_cwd()
        new_mgr = SessionManager.create(
            target_cwd,
            session_dir=self.sessions_dir,
            parent_session=parent_file,
        )
        # Copy entries up to branch_point
        for raw in self._entries:
            new_mgr._entries.append(raw)
            new_mgr._append_raw(raw)
            if branch_point_id and raw.get("id") == branch_point_id:
                break
        new_mgr._leaf_id = branch_point_id
        return new_mgr

    def branch_with_summary(
        self,
        summary: str,
        branch_point_id: str | None = None,
        cwd: str | None = None,
    ) -> "SessionManager":
        """Create a branched session and add a branch summary entry."""
        new_mgr = self.branch(branch_point_id, cwd)
        from_id = branch_point_id or (new_mgr.get_leaf_entry().id if new_mgr.get_leaf_entry() else "")
        new_mgr.append_branch_summary(summary, from_id)
        return new_mgr

    # ── Query helpers ──────────────────────────────────────────────────────

    def create_session(self, label: str | None = None) -> str:
        """
        Create a new session and return its ID.
        Backwards-compatibility shim for agent_session.py.
        """
        # If no file is set yet, create one
        if not self._session_file_path:
            cwd = self._cwd or os.getcwd()
            new_mgr = SessionManager.create(cwd, session_dir=self.sessions_dir)
            self._session_file_path = new_mgr._session_file_path
            self._header = new_mgr._header
            self._entries = new_mgr._entries
            self._by_id = new_mgr._by_id

        if label:
            self.append_session_info(name=label)

        return self.get_session_id()

    def get_messages(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """Extract messages from session in chronological order."""
        context = self.build_context()
        return context.messages

    def load_entries(self, session_id: str | None = None) -> list[SessionEntry]:
        """Load entries (compatibility shim)."""
        return self.get_entries()

    def delete_session(self, session_id: str | None = None) -> None:
        """Delete the current session file."""
        if self._session_file_path and os.path.exists(self._session_file_path):
            os.remove(self._session_file_path)

    def list_sessions(self) -> list[SessionInfo]:
        """List all sessions in the current sessions directory."""
        return self._list_sessions_from_dir(self.sessions_dir)
