"""
In-memory credential store.
Mirrors packages/ai/src/auth/credential-store.ts
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from pi_ai.utils.abort import operation_signal, race_with_abort_signal

from .types import AuthOperationOptions, Credential, CredentialInfo


class InMemoryCredentialStore:
    def __init__(self) -> None:
        self._credentials: dict[str, Credential] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, provider_id: str) -> asyncio.Lock:
        lock = self._locks.get(provider_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[provider_id] = lock
        return lock

    async def read(self, provider_id: str, options: AuthOperationOptions | None = None) -> Credential | None:
        cancel = options.cancel_event if options else None
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError("The operation was aborted")
        return self._credentials.get(provider_id)

    async def list(self, options: AuthOperationOptions | None = None) -> list[CredentialInfo]:
        cancel = options.cancel_event if options else None
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError("The operation was aborted")
        return [
            CredentialInfo(provider_id=provider_id, type=credential.type)
            for provider_id, credential in self._credentials.items()
        ]

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        cancel = operation_signal(options.cancel_event if options else None)

        async def _task() -> Credential | None:
            async with self._lock(provider_id):
                current = self._credentials.get(provider_id)
                nxt = await fn(current)
                if cancel.is_set():
                    raise asyncio.CancelledError("The operation was aborted")
                if nxt is not None:
                    self._credentials[provider_id] = nxt
                return nxt if nxt is not None else current

        return await race_with_abort_signal(_task(), cancel)

    async def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> None:
        cancel = operation_signal(options.cancel_event if options else None)

        async def _task() -> None:
            async with self._lock(provider_id):
                self._credentials.pop(provider_id, None)

        await race_with_abort_signal(_task(), cancel)
