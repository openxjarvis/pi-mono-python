"""
Persistent model catalogs keyed by provider ID.
Mirrors packages/ai/src/models-store.ts
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol

from pi_ai.types import Model


@dataclass
class ModelsStoreEntry:
    models: list[Model] = field(default_factory=list)
    last_modified: int | None = None
    checked_at: int | None = None
    etag: str | None = None


@dataclass
class ModelsStoreOperationOptions:
    cancel_event: asyncio.Event | None = None


class ModelsStore(Protocol):
    async def read(
        self, provider_id: str, options: ModelsStoreOperationOptions | None = None
    ) -> ModelsStoreEntry | None: ...

    async def write(
        self,
        provider_id: str,
        entry: ModelsStoreEntry,
        options: ModelsStoreOperationOptions | None = None,
    ) -> None: ...

    async def delete(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> None: ...


class InMemoryModelsStore:
    def __init__(self) -> None:
        self._entries: dict[str, ModelsStoreEntry] = {}

    async def read(
        self, provider_id: str, options: ModelsStoreOperationOptions | None = None
    ) -> ModelsStoreEntry | None:
        if options and options.cancel_event and options.cancel_event.is_set():
            raise asyncio.CancelledError("The operation was aborted")
        entry = self._entries.get(provider_id)
        return ModelsStoreEntry(
            models=list(entry.models),
            last_modified=entry.last_modified,
            checked_at=entry.checked_at,
            etag=entry.etag,
        ) if entry else None

    async def write(
        self,
        provider_id: str,
        entry: ModelsStoreEntry,
        options: ModelsStoreOperationOptions | None = None,
    ) -> None:
        if options and options.cancel_event and options.cancel_event.is_set():
            raise asyncio.CancelledError("The operation was aborted")
        self._entries[provider_id] = ModelsStoreEntry(
            models=list(entry.models),
            last_modified=entry.last_modified,
            checked_at=entry.checked_at,
            etag=entry.etag,
        )

    async def delete(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> None:
        if options and options.cancel_event and options.cancel_event.is_set():
            raise asyncio.CancelledError("The operation was aborted")
        self._entries.pop(provider_id, None)
