"""Session backend conformance cases. Mirrors harness/session/testing/conformance.ts."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pi_agent.harness.session.types import SessionError

from .types import SessionBackendConformanceCase, SessionBackendFixture, SessionBackendFixtureFactory


def _user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}], "timestamp": 1}


def _assistant_message(text: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "usage": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 0,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
        },
        "stopReason": "stop",
        "timestamp": 1,
    }


def _operation_started(identifier: str, lane: str, kind: str) -> dict[str, Any]:
    if kind == "run":
        intent: dict[str, Any] = {"kind": kind, "originalPrompt": [], "initialMessages": []}
    elif kind == "compaction":
        intent = {"kind": kind, "resultEntryId": f"{identifier}-result"}
    else:
        intent = {"kind": kind, "targetId": None, "summarize": False}
    return {"type": "operation_started", "id": identifier, "lane": lane, "sourceLeafId": None, "intent": intent}


async def _entry_ids(entries: Awaitable[list[dict[str, Any]]]) -> list[str]:
    return [entry["id"] for entry in await entries]


async def _rejects_with_code(operation: Awaitable[object], code: str) -> None:
    try:
        await operation
    except SessionError as error:
        if error.code != code:
            raise AssertionError(f"Expected SessionError with code {code}, got {error.code}") from error
        return
    raise AssertionError(f"Expected SessionError with code {code}")


def _create_case(
    factory: SessionBackendFixtureFactory,
    group: str,
    name: str,
    test: Callable[[Any], Awaitable[None]],
) -> SessionBackendConformanceCase:
    async def run() -> None:
        fixture = await factory()
        try:
            await test(fixture.repository)
        finally:
            closer = getattr(fixture, "aclose", None)
            if closer:
                await closer()

    return SessionBackendConformanceCase(group, name, run)


def create_session_backend_conformance(
    factory: SessionBackendFixtureFactory,
) -> list[SessionBackendConformanceCase]:
    async def assigns_parents(repository: Any) -> None:
        session = await repository.create({"id": "session"})
        root = await session.append_entry({"type": "message", "id": "root", "message": _user_message("root")}, "main")
        await session.create_lane("thread", root["id"])
        child = await session.append_entry({"type": "custom", "id": "child", "custom_type": "note", "data": {"value": 1}}, "thread")
        record = await session.append_record(_operation_started("run", "thread", "run"))
        await session.set_name("Example")
        await session.set_label(root["id"], "checkpoint")
        await session.move_lane("main", child["id"])
        assert {"parent_id": root.get("parent_id"), "seq": root["seq"]} == {"parent_id": None, "seq": 1}
        assert {"parent_id": child.get("parent_id"), "seq": child["seq"]} == {"parent_id": "root", "seq": 3}
        assert record["seq"] == 4
        kinds = [(item["kind"], item["seq"]) for item in await session.get_log()]
        assert kinds == [
            ("entry", 1),
            ("lane", 2),
            ("entry", 3),
            ("record", 4),
            ("fact", 5),
            ("fact", 6),
            ("lane", 7),
        ]

    async def commits_records(repository: Any) -> None:
        session = await repository.create({"id": "session"})
        root = await session.append_entry({"type": "message", "id": "root", "message": _user_message("root")}, "main")
        finished = await session.append_record(
            {"type": "operation_finished", "id": "finish", "lane": "main", "runId": "run", "outcome": "completed"}
        )
        assert finished["seq"] == 2
        assert await session.get_lanes() == [{"lane": "main", "leaf_id": "root"}]
        await session.move_lane("main", None)
        assert await session.get_lanes() == [{"lane": "main", "leaf_id": None}]
        await _rejects_with_code(session.move_lane("main", "missing"), "not_found")
        assert len(await session.find_records()) == 1

    async def rejects_duplicate_ids(repository: Any) -> None:
        session = await repository.create({"id": "session"})
        await session.append_entry({"type": "message", "id": "shared", "message": _user_message("root")}, "main")
        await _rejects_with_code(session.append_record(_operation_started("shared", "main", "run")), "already_exists")
        await session.append_record(_operation_started("run", "main", "run"))
        await _rejects_with_code(
            session.append_entry({"type": "custom", "id": "run", "custom_type": "note"}, "main"),
            "already_exists",
        )
        assert [item["seq"] for item in await session.get_log()] == [1, 2]

    async def isolates_lanes(repository: Any) -> None:
        session = await repository.create({"id": "session"})
        await session.append_entry({"type": "message", "id": "root", "message": _user_message("root")}, "main")
        await session.create_lane("thread", "root")
        await session.append_entry({"type": "message", "id": "main-child", "message": _user_message("main")}, "main")
        await session.append_entry({"type": "message", "id": "thread-child", "message": _user_message("thread")}, "thread")
        assert await session.get_lanes() == [
            {"lane": "main", "leaf_id": "main-child"},
            {"lane": "thread", "leaf_id": "thread-child"},
        ]
        assert await _entry_ids(session.find_entries_on_branch({"start": "main-child", "order": "oldestFirst"})) == [
            "root",
            "main-child",
        ]
        assert await _entry_ids(session.find_entries_on_branch({"start": "thread-child", "order": "oldestFirst"})) == [
            "root",
            "thread-child",
        ]

    async def rejects_invalid_queries(repository: Any) -> None:
        session = await repository.create({"id": "invalid-queries"})
        await session.create_lane("thread", None)
        await _rejects_with_code(session.find_entries({"limit": 0}), "invalid_query")
        await _rejects_with_code(session.find_entry({"limit": 0}), "invalid_query")
        await _rejects_with_code(session.find_records({"limit": 0}), "invalid_query")
        await _rejects_with_code(session.get_log({"after_seq": -1}), "invalid_query")

    async def bounded_queries(repository: Any) -> None:
        session = await repository.create({"id": "session"})
        await session.append_entry({"type": "message", "id": "root", "message": _user_message("root")}, "main")
        await session.append_entry({"type": "custom", "id": "old-note", "custom_type": "note", "data": 1}, "main")
        await session.append_entry(
            {"type": "compaction", "id": "compact", "summary": "summary", "retainedTail": [], "tokensBefore": 10},
            "main",
        )
        await session.append_entry({"type": "custom", "id": "new-note", "custom_type": "note", "data": 2}, "main")
        await session.append_entry({"type": "message", "id": "tail", "message": _assistant_message("tail")}, "main")
        assert await _entry_ids(session.find_entries()) == ["tail", "new-note", "compact", "old-note", "root"]
        assert await _entry_ids(session.find_entries({"order": "oldestFirst", "cursor": {"after_seq": 2}, "limit": 2})) == [
            "compact",
            "new-note",
        ]

    return [
        _create_case(factory, "entries and lanes", "assigns parents and one sequence across every mutation", assigns_parents),
        _create_case(factory, "records and log", "commits records and lane moves as separate mutations", commits_records),
        _create_case(factory, "entries and lanes", "rejects duplicate ids without changing state", rejects_duplicate_ids),
        _create_case(factory, "entries and lanes", "isolates lanes while sharing the tree", isolates_lanes),
        _create_case(factory, "queries and facts", "rejects invalid queries before empty reads", rejects_invalid_queries),
        _create_case(factory, "queries and facts", "supports bounded filtered and cursor-based queries", bounded_queries),
    ]


def run_storage_conformance(factory: Callable[[], Any]) -> None:
    storage = factory()
    if hasattr(storage, "write_header"):
        storage.write_header({"id": "test", "version": 4})
        storage.append({"type": "message", "id": "e1"})
        header, mutations = storage.read()
        assert header is not None
        assert len(mutations) == 1
