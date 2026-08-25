"""Remote catalog refresh. Mirrors model-catalog-refresh.ts"""
from __future__ import annotations

from typing import Any, Callable


async def refresh_model_catalogs(runtime: Any = None, signal: Any = None) -> Any:
    """TS-compatible alias used by InteractiveMode.run."""
    if runtime is None:
        return []
    refresh = getattr(runtime, "refresh", None) or getattr(runtime, "refresh_catalogs", None)
    if refresh is None:
        return []
    result = refresh()
    if hasattr(result, "__await__"):
        return await result
    return result


async def refresh_model_catalog(loader: Callable[[], Any] | None = None) -> Any:
    if loader is None:
        return []
    result = loader()
    if hasattr(result, "__await__"):
        return await result
    return result
