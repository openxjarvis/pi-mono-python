"""
Config selector — mirrors packages/coding-agent/src/cli/config-selector.ts

Provides a readline-based interactive config selector for `pi config` command.
"""
from __future__ import annotations

import asyncio
import os
import readline
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class ConfigSelectorOptions:
    resolved_paths: Any  # ResolvedPaths from package_manager
    settings_manager: Any  # SettingsManager
    cwd: str
    agent_dir: str


async def select_config(options: ConfigSelectorOptions) -> None:
    """
    Show an interactive config selector.
    Mirrors selectConfig() in TypeScript.

    Falls back to readline-based selection (no full TUI).
    """
    resolved = options.resolved_paths
    settings = options.settings_manager

    # Build config items list
    items = _build_config_items(resolved, settings, options.cwd, options.agent_dir)

    if not items:
        print("No configuration options available.")
        return

    await _run_readline_selector(items, options)


def _build_config_items(
    resolved_paths: Any,
    settings_manager: Any,
    cwd: str,
    agent_dir: str,
) -> list[dict[str, Any]]:
    """Build list of configurable items."""
    items = []

    # Global settings file
    global_settings_path = os.path.join(agent_dir, "settings.json")
    items.append({
        "type": "file",
        "label": "Global Settings",
        "path": global_settings_path,
        "description": "Edit global pi agent settings",
    })

    # Project settings file
    project_settings_path = os.path.join(cwd, ".pi", "settings.json")
    items.append({
        "type": "file",
        "label": "Project Settings",
        "path": project_settings_path,
        "description": f"Edit project-specific settings in {cwd}",
    })

    # Custom models file
    models_path = os.path.join(agent_dir, "models.json")
    items.append({
        "type": "file",
        "label": "Custom Models",
        "path": models_path,
        "description": "Add custom models or provider overrides",
    })

    # Keybindings file
    keybindings_path = os.path.join(agent_dir, "keybindings.json")
    items.append({
        "type": "file",
        "label": "Keybindings",
        "path": keybindings_path,
        "description": "Customize keyboard shortcuts",
    })

    # Extensions (from resolved_paths if available)
    if resolved_paths and hasattr(resolved_paths, "extension_paths"):
        for ext_path in (resolved_paths.extension_paths or []):
            items.append({
                "type": "extension",
                "label": f"Extension: {os.path.basename(ext_path)}",
                "path": ext_path,
                "description": f"Manage extension at {ext_path}",
            })

    return items


async def _run_readline_selector(
    items: list[dict[str, Any]],
    options: ConfigSelectorOptions,
) -> None:
    """Readline-based config selector fallback."""
    print("\nConfiguration options:")
    print("-" * 40)
    for i, item in enumerate(items, 1):
        exists_mark = "✓" if os.path.exists(item["path"]) else " "
        print(f"  [{exists_mark}] {i}. {item['label']}")
        print(f"         {item['description']}")

    print("\nOptions:")
    print("  [number] Open/edit item")
    print("  [q]      Quit")
    print()

    while True:
        try:
            choice = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: input("Choice: ").strip(),
            )
        except (EOFError, KeyboardInterrupt):
            break

        if choice.lower() in ("q", "quit", "exit"):
            break

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                item = items[idx]
                await _open_item(item, options)
            else:
                print(f"Invalid choice. Enter 1-{len(items)} or q.")
        except ValueError:
            print(f"Invalid choice. Enter 1-{len(items)} or q.")


async def _open_item(item: dict[str, Any], options: ConfigSelectorOptions) -> None:
    """Open a config item for editing."""
    path = item["path"]

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create default content if file doesn't exist
    if not os.path.exists(path):
        if item["type"] == "file":
            _create_default_config(path, item["label"])
            print(f"Created: {path}")
        elif item["type"] == "extension":
            print(f"Extension directory: {path}")
            return

    # Open in editor
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))
    print(f"Opening {path} in {editor}...")
    try:
        proc = await asyncio.create_subprocess_exec(editor, path)
        await proc.wait()
    except Exception as e:
        print(f"Could not open editor: {e}")
        print(f"File path: {path}")


def _create_default_config(path: str, label: str) -> None:
    """Create a default empty config file."""
    import json

    if "models" in label.lower():
        default = {"providers": {}}
    elif "keybinding" in label.lower():
        default = {}
    else:
        default = {}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(default, f, indent=2)
        f.write("\n")
