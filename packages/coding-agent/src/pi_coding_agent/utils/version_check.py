"""Version check helper. Mirrors packages/coding-agent/src/utils/version-check.ts"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def get_package_version(name: str = "pi-coding-agent") -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "0.0.0"
