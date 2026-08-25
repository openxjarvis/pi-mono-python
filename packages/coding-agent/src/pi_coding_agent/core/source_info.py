"""
Resource source metadata — mirrors packages/coding-agent/src/core/source-info.ts
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .package_manager import PathMetadata

SourceScope = Literal["user", "project", "temporary"]
SourceOrigin = Literal["package", "top-level"]


@dataclass
class SourceInfo:
    path: str
    source: str
    scope: SourceScope
    origin: SourceOrigin
    base_dir: str | None = None


def create_source_info(path: str, metadata: PathMetadata) -> SourceInfo:
    return SourceInfo(
        path=path,
        source=metadata.source,
        scope=metadata.scope,
        origin=metadata.origin,
        base_dir=metadata.base_dir,
    )


def create_synthetic_source_info(
    path: str,
    *,
    source: str,
    scope: SourceScope = "temporary",
    origin: SourceOrigin = "top-level",
    base_dir: str | None = None,
) -> SourceInfo:
    return SourceInfo(
        path=path,
        source=source,
        scope=scope,
        origin=origin,
        base_dir=base_dir,
    )
