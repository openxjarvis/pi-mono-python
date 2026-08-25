"""
Pi package manifest reader — mirrors packages/coding-agent/src/core/pi-manifest.ts
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pi_coding_agent.utils.text import strip_bom

RESOURCE_FIELDS = ("extensions", "skills", "prompts", "themes")


@dataclass
class PiManifest:
    extensions: list[str] | None = None
    skills: list[str] | None = None
    prompts: list[str] | None = None
    themes: list[str] | None = None


def _is_object(value: Any) -> bool:
    return isinstance(value, dict)


def read_pi_manifest(package_json_path: str) -> PiManifest | None:
    try:
        with open(package_json_path, encoding="utf-8") as f:
            pkg = json.loads(strip_bom(f.read()))
    except (OSError, json.JSONDecodeError):
        return None
    if not _is_object(pkg) or not _is_object(pkg.get("pi")):
        return None
    pi = pkg["pi"]
    manifest = PiManifest()
    for field in RESOURCE_FIELDS:
        entries = pi.get(field)
        if isinstance(entries, list) and all(isinstance(entry, str) for entry in entries):
            setattr(manifest, field, list(entries))
    return manifest
