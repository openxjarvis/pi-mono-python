"""
One-time migrations that run on startup.

Mirrors packages/coding-agent/src/migrations.ts
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

from pi_coding_agent.config import CONFIG_DIR_NAME, get_agent_dir, get_bin_dir

MIGRATION_GUIDE_URL = (
    "https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/CHANGELOG.md#extensions-migration"
)
EXTENSIONS_DOC_URL = (
    "https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md"
)


def migrate_auth_to_auth_json() -> list[str]:
    """
    Migrate legacy oauth.json and settings.json apiKeys to auth.json.

    Returns list of provider names that were migrated.
    """
    agent_dir = get_agent_dir()
    auth_path = os.path.join(agent_dir, "auth.json")
    oauth_path = os.path.join(agent_dir, "oauth.json")
    settings_path = os.path.join(agent_dir, "settings.json")

    if os.path.exists(auth_path):
        return []

    migrated: dict[str, object] = {}
    providers: list[str] = []

    if os.path.exists(oauth_path):
        try:
            with open(oauth_path, encoding="utf-8") as f:
                oauth = json.load(f)
            for provider, cred in oauth.items():
                migrated[provider] = {"type": "oauth", **(cred if isinstance(cred, dict) else {})}
                providers.append(provider)
            os.rename(oauth_path, f"{oauth_path}.migrated")
        except Exception:
            pass

    if os.path.exists(settings_path):
        try:
            with open(settings_path, encoding="utf-8") as f:
                settings = json.load(f)
            api_keys = settings.get("apiKeys", {})
            if isinstance(api_keys, dict):
                for provider, key in api_keys.items():
                    if provider not in migrated and isinstance(key, str):
                        migrated[provider] = {"type": "api_key", "key": key}
                        providers.append(provider)
                del settings["apiKeys"]
                with open(settings_path, "w", encoding="utf-8") as f:
                    json.dump(settings, f, indent=2)
        except Exception:
            pass

    if migrated:
        os.makedirs(os.path.dirname(auth_path), exist_ok=True)
        with open(auth_path, "w", encoding="utf-8") as f:
            json.dump(migrated, f, indent=2)
        os.chmod(auth_path, 0o600)

    return providers


def migrate_sessions_from_agent_root() -> None:
    """
    Migrate sessions from ~/.pi/agent/*.jsonl to proper session directories.

    Bug in v0.30.0: Sessions were saved to ~/.pi/agent/ instead of
    ~/.pi/agent/sessions/<encoded-cwd>/. This migration moves them
    to the correct location based on the cwd in their session header.
    """
    agent_dir = get_agent_dir()

    try:
        files = [
            os.path.join(agent_dir, f)
            for f in os.listdir(agent_dir)
            if f.endswith(".jsonl") and os.path.isfile(os.path.join(agent_dir, f))
        ]
    except OSError:
        return

    if not files:
        return

    for file_path in files:
        try:
            with open(file_path, encoding="utf-8") as f:
                first_line = f.readline()
            if not first_line.strip():
                continue
            header = json.loads(first_line)
            if header.get("type") != "session" or not header.get("cwd"):
                continue

            cwd: str = header["cwd"]
            # Compute the correct session directory (same encoding as session-manager.py)
            safe_path = "--" + cwd.lstrip("/\\").replace("/", "-").replace("\\", "-").replace(":", "-") + "--"
            correct_dir = os.path.join(agent_dir, "sessions", safe_path)

            os.makedirs(correct_dir, exist_ok=True)

            file_name = os.path.basename(file_path)
            new_path = os.path.join(correct_dir, file_name)

            if os.path.exists(new_path):
                continue

            os.rename(file_path, new_path)
        except Exception:
            pass


def _migrate_commands_to_prompts(base_dir: str, label: str) -> bool:
    """Migrate commands/ to prompts/ if needed. Returns True if migrated."""
    commands_dir = os.path.join(base_dir, "commands")
    prompts_dir = os.path.join(base_dir, "prompts")

    if os.path.exists(commands_dir) and not os.path.exists(prompts_dir):
        try:
            os.rename(commands_dir, prompts_dir)
            print(f"Migrated {label} commands/ → prompts/")
            return True
        except OSError as err:
            print(f"Warning: Could not migrate {label} commands/ to prompts/: {err}", file=sys.stderr)
    return False


def _migrate_tools_to_bin() -> None:
    """Move fd/rg binaries from tools/ to bin/ if they exist."""
    agent_dir = get_agent_dir()
    tools_dir = os.path.join(agent_dir, "tools")
    bin_dir = get_bin_dir()

    if not os.path.exists(tools_dir):
        return

    binaries = ["fd", "rg", "fd.exe", "rg.exe"]
    moved_any = False

    for binary in binaries:
        old_path = os.path.join(tools_dir, binary)
        new_path = os.path.join(bin_dir, binary)

        if os.path.exists(old_path):
            os.makedirs(bin_dir, exist_ok=True)
            if not os.path.exists(new_path):
                try:
                    os.rename(old_path, new_path)
                    moved_any = True
                except OSError:
                    pass
            else:
                try:
                    os.remove(old_path)
                except OSError:
                    pass

    if moved_any:
        print("Migrated managed binaries tools/ → bin/")


def _check_deprecated_extension_dirs(base_dir: str, label: str) -> list[str]:
    """Check for deprecated hooks/ and tools/ directories."""
    hooks_dir = os.path.join(base_dir, "hooks")
    tools_dir = os.path.join(base_dir, "tools")
    warnings: list[str] = []

    if os.path.exists(hooks_dir):
        warnings.append(f"{label} hooks/ directory found. Hooks have been renamed to extensions.")

    if os.path.exists(tools_dir):
        try:
            entries = os.listdir(tools_dir)
            custom_tools = [
                e for e in entries
                if e.lower() not in ("fd", "rg", "fd.exe", "rg.exe") and not e.startswith(".")
            ]
            if custom_tools:
                warnings.append(
                    f"{label} tools/ directory contains custom tools. "
                    "Custom tools have been merged into extensions."
                )
        except OSError:
            pass

    return warnings


def _migrate_extension_system(cwd: str) -> list[str]:
    """Run extension system migrations and collect warnings about deprecated directories."""
    agent_dir = get_agent_dir()
    project_dir = os.path.join(cwd, CONFIG_DIR_NAME)

    _migrate_commands_to_prompts(agent_dir, "Global")
    _migrate_commands_to_prompts(project_dir, "Project")

    warnings = [
        *_check_deprecated_extension_dirs(agent_dir, "Global"),
        *_check_deprecated_extension_dirs(project_dir, "Project"),
    ]

    return warnings


async def show_deprecation_warnings(warnings: list[str]) -> None:
    """Print deprecation warnings and wait for keypress."""
    if not warnings:
        return

    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    print(f"\nMove your extensions to the extensions/ directory.", file=sys.stderr)
    print(f"Migration guide: {MIGRATION_GUIDE_URL}", file=sys.stderr)
    print(f"Documentation: {EXTENSIONS_DOC_URL}", file=sys.stderr)
    print("\nPress Enter to continue...", end="", flush=True, file=sys.stderr)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, sys.stdin.readline)
    print()


def run_migrations(cwd: str | None = None) -> dict[str, list[str]]:
    """
    Run all migrations. Called once on startup.

    Returns dict with migration results and deprecation warnings.
    """
    if cwd is None:
        cwd = os.getcwd()

    migrated_auth_providers = migrate_auth_to_auth_json()
    migrate_sessions_from_agent_root()
    _migrate_tools_to_bin()
    deprecation_warnings = _migrate_extension_system(cwd)

    return {
        "migratedAuthProviders": migrated_auth_providers,
        "deprecationWarnings": deprecation_warnings,
    }
