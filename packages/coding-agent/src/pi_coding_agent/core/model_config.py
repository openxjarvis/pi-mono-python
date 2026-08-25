"""
models.json snapshot — mirrors packages/coding-agent/src/core/model-config.ts
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from pi_coding_agent.utils.text import strip_bom


def strip_json_comments(text: str) -> str:
    """Remove // and /* */ comments from JSONC without touching strings."""
    result: list[str] = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        ch = text[i]
        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            result.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            while i < len(text) and text[i] not in "\n\r":
                i += 1
            continue
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        result.append(ch)
        i += 1
    return "".join(result)


@dataclass
class ModelConfig:
    """Immutable, credential-blind models.json snapshot."""

    path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    providers: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | None) -> "ModelConfig":
        config = cls(path=path)
        if not path or not os.path.exists(path):
            return config
        try:
            with open(path, encoding="utf-8") as f:
                text = strip_json_comments(strip_bom(f.read()))
            parsed = json.loads(text or "{}")
        except (OSError, json.JSONDecodeError) as exc:
            config.errors.append(f"Failed to read models.json {path}: {exc}")
            return config
        if not isinstance(parsed, dict):
            config.errors.append(f"Invalid models.json {path}: expected an object")
            return config
        config.raw = parsed
        providers = parsed.get("providers")
        if isinstance(providers, dict):
            config.providers = providers
        elif "providers" not in parsed:
            # Allow a flat provider map at the top level.
            config.providers = {
                key: value for key, value in parsed.items() if isinstance(value, dict)
            }
        return config

    def get_provider(self, provider_id: str) -> dict[str, Any] | None:
        value = self.providers.get(provider_id)
        return value if isinstance(value, dict) else None

    def get_model_override(self, provider_id: str, model_id: str) -> dict[str, Any] | None:
        provider = self.get_provider(provider_id)
        if not provider:
            return None
        models = provider.get("models")
        if isinstance(models, dict):
            override = models.get(model_id)
            return override if isinstance(override, dict) else None
        if isinstance(models, list):
            for item in models:
                if isinstance(item, dict) and item.get("id") == model_id:
                    return item
        return None
