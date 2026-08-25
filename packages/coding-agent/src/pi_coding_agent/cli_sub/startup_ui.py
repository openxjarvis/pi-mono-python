"""
Startup TUI helpers — mirrors packages/coding-agent/src/cli/startup-ui.ts
"""
from __future__ import annotations

import os
from typing import Any

from pi_coding_agent.config import APP_NAME, CONFIG_DIR_NAME, get_agent_dir
from pi_coding_agent.core.experimental import are_experimental_features_enabled
from pi_coding_agent.core.keybindings import KeybindingsManager
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.modes.interactive.theme.theme import (
    detect_terminal_background_from_env,
    init_theme,
    load_theme_from_path,
    resolve_theme_setting,
    set_registered_themes,
)


OFFICIAL_PACKAGE_NAME = "@earendil-works/pi-coding-agent"
OFFICIAL_APP_NAME = "pi"
OFFICIAL_CONFIG_DIR_NAME = ".pi"


def is_official_distribution(
    package_name: str = OFFICIAL_PACKAGE_NAME,
    app_name: str = APP_NAME,
    config_dir_name: str = CONFIG_DIR_NAME,
) -> bool:
    return (
        package_name == OFFICIAL_PACKAGE_NAME
        and app_name == OFFICIAL_APP_NAME
        and config_dir_name == OFFICIAL_CONFIG_DIR_NAME
    )


def load_themes(resources: list[Any]) -> list[Any]:
    themes = []
    seen: set[str] = set()
    for resource in resources:
        enabled = getattr(resource, "enabled", True)
        path = getattr(resource, "path", None) or (resource.get("path") if isinstance(resource, dict) else None)
        if not enabled or not path:
            continue
        try:
            loaded = load_theme_from_path(path)
        except Exception:
            continue
        name = getattr(loaded, "name", None)
        if name:
            if name in seen:
                continue
            seen.add(name)
        themes.append(loaded)
    return themes


async def load_startup_themes(settings_manager: SettingsManager) -> list[Any]:
    return []


async def create_startup_tui(settings_manager: SettingsManager) -> dict[str, Any]:
    set_registered_themes(await load_startup_themes(settings_manager))
    terminal_theme = detect_terminal_background_from_env().get("theme", "dark")
    init_theme(resolve_theme_setting(getattr(settings_manager, "get_theme", lambda: None)(), terminal_theme) or terminal_theme)
    return {
        "theme": terminal_theme,
        "experimental": are_experimental_features_enabled(),
        "keybindings": KeybindingsManager(),
        "agentDir": get_agent_dir(),
        "exists": os.path.exists(get_agent_dir()),
    }
