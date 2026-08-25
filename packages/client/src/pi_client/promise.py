from __future__ import annotations

import asyncio
from typing import Generic, TypeVar

T = TypeVar("T")


class PromiseResolvers(Generic[T]):
    def __init__(self) -> None:
        loop = asyncio.get_event_loop()
        self.future: asyncio.Future[T] = loop.create_future()

    @property
    def promise(self) -> asyncio.Future[T]:
        return self.future

    def resolve(self, value: T) -> None:
        if not self.future.done():
            self.future.set_result(value)

    def reject(self, reason: object = None) -> None:
        if not self.future.done():
            error = reason if isinstance(reason, Exception) else Exception(str(reason))
            self.future.set_exception(error)


def create_promise_resolvers() -> PromiseResolvers:
    return PromiseResolvers()
