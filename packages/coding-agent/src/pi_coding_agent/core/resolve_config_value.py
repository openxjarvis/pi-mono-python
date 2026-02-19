"""
Resolve configuration values (API keys, header values) that may be
shell commands, environment variables, or literals.

Mirrors core/resolve-config-value.ts
"""

from __future__ import annotations

import os
import subprocess

# Cache for shell command results (persists for process lifetime)
_command_result_cache: dict[str, str | None] = {}


def resolve_config_value(config: str) -> str | None:
    """Resolve a config value to an actual string value.

    - If starts with "!", executes the rest as a shell command (result cached).
    - Otherwise checks if it matches an environment variable, then treats as literal.
    """
    if config.startswith("!"):
        return _execute_command(config)
    env_value = os.environ.get(config)
    return env_value or config


def _execute_command(command_config: str) -> str | None:
    if command_config in _command_result_cache:
        return _command_result_cache[command_config]

    command = command_config[1:]
    result: str | None = None
    try:
        output = subprocess.check_output(
            command,
            shell=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
        stripped = output.decode("utf-8", errors="replace").strip()
        result = stripped or None
    except Exception:
        result = None

    _command_result_cache[command_config] = result
    return result


def resolve_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    """Resolve all header values using the same logic as API keys."""
    if not headers:
        return None
    resolved: dict[str, str] = {}
    for key, value in headers.items():
        resolved_value = resolve_config_value(value)
        if resolved_value:
            resolved[key] = resolved_value
    return resolved or None


def clear_config_value_cache() -> None:
    """Clear the config value command cache (for testing)."""
    _command_result_cache.clear()
