"""
llama.cpp extension UI — mirrors packages/coding-agent/src/extensions/llama/ui.ts
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from .client import LlamaModelInfo, LlamaProgress
from .huggingface import HuggingFaceModel

LlamaManagerActionType = Literal["model", "download", "close"]


@dataclass
class LlamaManagerAction:
    type: LlamaManagerActionType
    model: LlamaModelInfo | None = None


class LlamaUi:
    async def show_models(self, server_url: str, models: list[LlamaModelInfo]) -> LlamaManagerAction:
        return LlamaManagerAction(type="close")

    async def select(self, title: str, options: list[str]) -> str | None:
        return options[0] if options else None

    async def confirm(self, title: str, message: str) -> bool:
        return True

    async def search_models(
        self,
        search: Callable[[str, Any], Awaitable[list[HuggingFaceModel]]],
    ) -> str | None:
        return None

    async def connection_error(self, server_url: str, message: str) -> str:
        return "close"

    def show_status(self, title: str, detail: str | None = None) -> None:
        return None


async def show_llama_ui(ctx: Any, run: Callable[[LlamaUi], Awaitable[None]]) -> None:
    await run(LlamaUi())


async def run_with_progress(
    ui: LlamaUi,
    *,
    title: str,
    model: str,
    initial_message: str,
    cancel_title: str,
    cancel_message: str,
    run: Callable[[Any, Callable[[LlamaProgress], None]], Awaitable[Any]],
    cancel: Callable[[], Awaitable[Any] | Any],
) -> dict[str, Any]:
    def update(_progress: LlamaProgress) -> None:
        return None

    try:
        value = await run(None, update)
        return {"cancelled": False, "value": value}
    except Exception:
        return {"cancelled": True}
