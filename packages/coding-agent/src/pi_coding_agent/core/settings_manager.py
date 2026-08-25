"""
Settings management — mirrors packages/coding-agent/src/core/settings-manager.ts

Manages global (~/.pi/agent/settings.json) and project (.pi/settings.json) settings.
Supports deep merge, write queue, settings migration, and all getter/setter methods.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


# ─── Sub-settings dataclasses ─────────────────────────────────────────────────

@dataclass
class CompactionSettings:
    enabled: bool | None = None          # default: True
    reserve_tokens: int | None = None    # default: 16384
    keep_recent_tokens: int | None = None  # default: 20000


@dataclass
class BranchSummarySettings:
    reserve_tokens: int | None = None    # default: 16384
    skip_prompt: bool | None = None      # default: False


@dataclass
class RetrySettings:
    enabled: bool | None = None          # default: True
    max_retries: int | None = None       # default: 3
    base_delay_ms: int | None = None     # default: 2000
    max_delay_ms: int | None = None      # default: 60000


@dataclass
class TerminalSettings:
    show_images: bool | None = None      # default: True
    clear_on_shrink: bool | None = None  # default: False


@dataclass
class ImageSettings:
    auto_resize: bool | None = None      # default: True
    block_images: bool | None = None     # default: False


@dataclass
class ThinkingBudgetsSettings:
    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


@dataclass
class MarkdownSettings:
    code_block_indent: str | None = None  # default: "  "


# ─── Full settings ─────────────────────────────────────────────────────────────

@dataclass
class Settings:
    """
    Agent settings — mirrors the Settings interface in TypeScript.
    All fields optional (merged from global + project).
    """
    # Core
    last_changelog_version: str | None = None
    default_provider: str | None = None
    default_model: str | None = None
    default_thinking_level: str | None = None  # off|minimal|low|medium|high|xhigh
    model_thinking_levels: dict[str, str] | None = None  # keyed by "provider/modelId"
    transport: str | None = None               # "auto" | "sse" | "websocket"
    default_tools: list[str] | None = None
    steering_mode: str | None = "one-at-a-time"  # "all" | "one-at-a-time"
    follow_up_mode: str | None = "one-at-a-time"  # "all" | "one-at-a-time"
    theme: str | None = None
    hide_thinking_block: bool | None = None
    shell_path: str | None = None
    quiet_startup: bool | None = None
    shell_command_prefix: str | None = None
    collapse_changelog: bool | None = None
    enable_skill_commands: bool | None = None  # default: True
    double_escape_action: str | None = None    # "fork" | "tree" | "none"
    editor_padding_x: int | None = None
    autocomplete_max_visible: int | None = None
    show_hardware_cursor: bool | None = None
    tree_filter_mode: str | None = None        # "default"|"no-tools"|"user-only"|"labeled-only"|"all"

    # Nested settings objects (stored as dicts in JSON)
    compaction: dict[str, Any] | None = None
    branch_summary: dict[str, Any] | None = None
    retry: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None
    images: dict[str, Any] | None = None
    thinking_budgets: dict[str, Any] | None = None
    markdown: dict[str, Any] | None = None

    # Array fields
    packages: list[Any] | None = None
    extensions: list[str] | None = None
    skills: list[str] | None = None
    prompts: list[str] | None = None
    themes: list[str] | None = None
    enabled_models: list[str] | None = None

    # Legacy/compat fields
    thinking_level: str = "off"
    auto_compact: bool = True
    compact_threshold: float = 0.8
    max_retries: int = 3
    retry_delay_ms: int = 1000
    include_images: bool = True
    max_image_size_kb: int = 5000
    model_id: str | None = None
    provider: str | None = None
    image_auto_resize: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        known = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def merge(self, other: "Settings") -> "Settings":
        """Merge another Settings into this one (other wins for non-None values)."""
        base = self.to_dict()
        for k, v in other.to_dict().items():
            if v is None:
                continue
            # Deep merge nested dicts
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = {**base[k], **v}
            else:
                base[k] = v
        return Settings.from_dict(base)


# ─── Deep merge helper ────────────────────────────────────────────────────────

def deep_merge_settings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge override into base. Nested objects merge recursively.
    Arrays and primitives: override wins.
    Mirrors deepMergeSettings() in TypeScript.
    """
    result = dict(base)
    for key, val in override.items():
        if val is None:
            continue
        base_val = result.get(key)
        if (
            isinstance(val, dict)
            and isinstance(base_val, dict)
        ):
            result[key] = {**base_val, **val}
        else:
            result[key] = val
    return result


# ─── Settings migration ───────────────────────────────────────────────────────

def migrate_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Migrate old settings format to current.
    Mirrors migrateSettings() in TypeScript.
    """
    # queueMode → steeringMode
    if "queueMode" in raw and "steeringMode" not in raw:
        raw["steeringMode"] = raw.pop("queueMode")

    # legacy websockets bool → transport enum
    if "transport" not in raw and "websockets" in raw:
        raw["transport"] = "websocket" if raw.pop("websockets") else "sse"

    # Old skills object → array
    skills = raw.get("skills")
    if isinstance(skills, dict) and skills is not None:
        ec = skills.get("enableSkillCommands")
        dirs = skills.get("customDirectories")
        if ec is not None and "enableSkillCommands" not in raw:
            raw["enableSkillCommands"] = ec
        if isinstance(dirs, list) and dirs:
            raw["skills"] = dirs
        else:
            raw.pop("skills", None)

    # retry.maxDelayMs → retry.provider.maxRetryDelayMs
    retry = raw.get("retry")
    if isinstance(retry, dict):
        provider = retry.get("provider")
        provider_settings = provider if isinstance(provider, dict) else {}
        if (
            isinstance(retry.get("maxDelayMs"), (int, float))
            and provider_settings.get("maxRetryDelayMs") is None
        ):
            retry["provider"] = {**provider_settings, "maxRetryDelayMs": retry["maxDelayMs"]}
        retry.pop("maxDelayMs", None)

    return raw


# ─── SettingsManager ──────────────────────────────────────────────────────────

class SettingsManager:
    """
    Manages global and project-level settings.
    Mirrors SettingsManager in TypeScript.

    Global settings:  ~/.pi/agent/settings.json
    Project settings: <project_root>/.pi/settings.json

    Provides:
    - deep merge (project overrides global)
    - asyncio.Lock write queue to prevent race conditions
    - settings migration on load
    - all getter/setter methods matching TypeScript API
    """

    GLOBAL_SETTINGS_DIR = os.path.join(os.path.expanduser("~"), ".pi", "agent")

    def __init__(
        self,
        project_root: str | None = None,
        global_settings_file: str | None = None,
    ) -> None:
        self.project_root = project_root or os.getcwd()
        self._project_settings_file = os.path.join(self.project_root, ".pi", "settings.json")
        self._global_settings_file = (
            global_settings_file
            or os.path.join(self.GLOBAL_SETTINGS_DIR, "settings.json")
        )
        self._global_raw: dict[str, Any] = {}
        self._project_raw: dict[str, Any] = {}
        self._merged: dict[str, Any] = {}
        self._errors: list[dict[str, Any]] = []
        self._write_lock = asyncio.Lock()
        self._runtime_overrides: dict[str, Any] = {}
        self._loaded = False
        self._project_trusted = True

    @classmethod
    def create(
        cls,
        cwd: str | None = None,
        agent_dir: str | None = None,
    ) -> "SettingsManager":
        """Factory matching TypeScript SettingsManager.create(cwd, agentDir)."""
        global_file = (
            os.path.join(agent_dir, "settings.json") if agent_dir else None
        )
        mgr = cls(project_root=cwd, global_settings_file=global_file)
        mgr.load()
        return mgr

    @classmethod
    def in_memory(cls, settings: dict[str, Any] | None = None) -> "SettingsManager":
        """Create an in-memory settings manager (no file I/O)."""
        mgr = cls()
        if settings:
            mgr._global_raw = dict(settings)
            mgr._rebuild()
        mgr._loaded = True
        return mgr

    # ── Load / Save ───────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load settings from disk and run migration."""
        self._global_raw = self._load_file(self._global_settings_file)
        self._project_raw = self._load_file(self._project_settings_file)
        self._rebuild()
        self._loaded = True

    def _load_file(self, path: str) -> dict[str, Any]:
        """Load raw settings dict from JSON file, running migration."""
        if not os.path.exists(path):
            return {}
        try:
            from pi_coding_agent.utils.text import load_json_file
            raw = load_json_file(path)
            if not isinstance(raw, dict):
                return {}
            return migrate_settings(raw)
        except (json.JSONDecodeError, OSError) as e:
            self._errors.append({"scope": "global" if "agent" in path else "project", "error": str(e)})
            return {}

    def _rebuild(self) -> None:
        """Recompute merged settings from global + project + runtime overrides."""
        self._merged = deep_merge_settings(self._global_raw, self._project_raw)
        if self._runtime_overrides:
            self._merged = deep_merge_settings(self._merged, self._runtime_overrides)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def reload(self) -> None:
        """Reload settings from disk."""
        self.load()

    def _write_file(self, path: str, data: dict[str, Any]) -> None:
        """Write settings dict to JSON file, creating dirs as needed."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    def save_global(self, key: str, value: Any) -> None:
        """Update and persist a single global settings key."""
        self._ensure_loaded()
        self._global_raw[key] = value
        self._rebuild()
        self._write_file(self._global_settings_file, self._global_raw)

    def save_project(self, key: str, value: Any) -> None:
        """Update and persist a single project settings key."""
        self._ensure_loaded()
        self._project_raw[key] = value
        self._rebuild()
        self._write_file(self._project_settings_file, self._project_raw)

    async def save_global_async(self, key: str, value: Any) -> None:
        """Thread-safe async version of save_global."""
        async with self._write_lock:
            self.save_global(key, value)

    async def save_project_async(self, key: str, value: Any) -> None:
        """Thread-safe async version of save_project."""
        async with self._write_lock:
            self.save_project(key, value)

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply runtime overrides (not persisted to disk)."""
        self._runtime_overrides = deep_merge_settings(self._runtime_overrides, overrides)
        self._rebuild()

    # ── Read access ───────────────────────────────────────────────────────────

    def get(self) -> Settings:
        """Get merged Settings object (project overrides global)."""
        self._ensure_loaded()
        merged = self._map_raw_to_settings(self._merged)
        return Settings.from_dict(merged)

    def get_merged_raw(self) -> dict[str, Any]:
        """Get merged raw dict."""
        self._ensure_loaded()
        return dict(self._merged)

    def get_global_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        return dict(self._global_raw)

    def get_project_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        return dict(self._project_raw)

    def is_project_trusted(self) -> bool:
        return self._project_trusted

    def set_project_trusted(self, trusted: bool) -> None:
        if self._project_trusted == trusted:
            return
        self._project_trusted = trusted
        if not trusted:
            self._project_raw = {}
            self._rebuild()
            return
        self._project_raw = self._load_file(self._project_settings_file)
        self._rebuild()

    def drain_errors(self) -> list[dict[str, Any]]:
        """Drain and return all accumulated settings errors."""
        drained = list(self._errors)
        self._errors = []
        return drained

    @staticmethod
    def _map_raw_to_settings(raw: dict[str, Any]) -> dict[str, Any]:
        """Convert camelCase JSON keys to snake_case for Settings dataclass."""
        mapping = {
            "defaultProvider": "default_provider",
            "defaultModel": "default_model",
            "defaultThinkingLevel": "default_thinking_level",
            "modelThinkingLevels": "model_thinking_levels",
            "defaultTools": "default_tools",
            "steeringMode": "steering_mode",
            "followUpMode": "follow_up_mode",
            "hideThinkingBlock": "hide_thinking_block",
            "shellPath": "shell_path",
            "quietStartup": "quiet_startup",
            "shellCommandPrefix": "shell_command_prefix",
            "collapseChangelog": "collapse_changelog",
            "enableSkillCommands": "enable_skill_commands",
            "doubleEscapeAction": "double_escape_action",
            "editorPaddingX": "editor_padding_x",
            "autocompleteMaxVisible": "autocomplete_max_visible",
            "showHardwareCursor": "show_hardware_cursor",
            "treeFilterMode": "tree_filter_mode",
            "branchSummary": "branch_summary",
            "thinkingBudgets": "thinking_budgets",
            "enabledModels": "enabled_models",
            "lastChangelogVersion": "last_changelog_version",
        }
        result = {}
        for k, v in raw.items():
            py_key = mapping.get(k, k)
            result[py_key] = v
        return result

    # ── Typed getters (matching TypeScript API) ────────────────────────────────

    def get_default_provider(self) -> str | None:
        self._ensure_loaded()
        return self._merged.get("defaultProvider") or self._merged.get("default_provider")

    def get_default_model(self) -> str | None:
        self._ensure_loaded()
        return self._merged.get("defaultModel") or self._merged.get("default_model")

    def get_default_thinking_level(self) -> str | None:
        self._ensure_loaded()
        return self._merged.get("defaultThinkingLevel") or self._merged.get("default_thinking_level")

    def set_default_thinking_level(self, level: str) -> None:
        """Persist the global default thinking level."""
        self.save_global("defaultThinkingLevel", level)

    def get_model_thinking_level(self, provider: str, model_id: str) -> str | None:
        """Per-model thinking override keyed by provider/modelId."""
        levels = self.get_all_model_thinking_levels()
        return levels.get(f"{provider}/{model_id}")

    def get_all_model_thinking_levels(self) -> dict[str, str]:
        self._ensure_loaded()
        raw = self._merged.get("modelThinkingLevels") or self._merged.get("model_thinking_levels") or {}
        return dict(raw) if isinstance(raw, dict) else {}

    def set_model_thinking_level(self, provider: str, model_id: str, level: str) -> None:
        self._ensure_loaded()
        levels = dict(self._global_raw.get("modelThinkingLevels") or {})
        levels[f"{provider}/{model_id}"] = level
        self.save_global("modelThinkingLevels", levels)

    def remove_model_thinking_level(self, provider: str, model_id: str) -> None:
        self._ensure_loaded()
        levels = dict(self._global_raw.get("modelThinkingLevels") or {})
        levels.pop(f"{provider}/{model_id}", None)
        if levels:
            self.save_global("modelThinkingLevels", levels)
        elif "modelThinkingLevels" in self._global_raw:
            del self._global_raw["modelThinkingLevels"]
            self._rebuild()
            self._write_file(self._global_settings_file, self._global_raw)

    def get_default_tools(self) -> list[str] | None:
        self._ensure_loaded()
        val = self._merged.get("defaultTools") or self._merged.get("default_tools")
        return list(val) if isinstance(val, list) else None

    def get_theme(self) -> str | None:
        self._ensure_loaded()
        return self._merged.get("theme")

    def get_transport(self) -> str:
        self._ensure_loaded()
        return self._merged.get("transport") or "auto"

    def get_steering_mode(self) -> str:
        self._ensure_loaded()
        return self._merged.get("steeringMode") or self._merged.get("steering_mode", "one-at-a-time")

    def get_follow_up_mode(self) -> str:
        self._ensure_loaded()
        return self._merged.get("followUpMode") or self._merged.get("follow_up_mode", "one-at-a-time")

    def set_steering_mode(self, mode: str) -> None:
        self.save_global("steeringMode", mode)

    def set_follow_up_mode(self, mode: str) -> None:
        self.save_global("followUpMode", mode)

    def get_quiet_startup(self) -> bool:
        self._ensure_loaded()
        return bool(self._merged.get("quietStartup") or self._merged.get("quiet_startup", False))

    def get_shell_path(self) -> str | None:
        self._ensure_loaded()
        return self._merged.get("shellPath") or self._merged.get("shell_path")

    def get_shell_command_prefix(self) -> str | None:
        self._ensure_loaded()
        return self._merged.get("shellCommandPrefix") or self._merged.get("shell_command_prefix")

    def get_enable_skill_commands(self) -> bool:
        self._ensure_loaded()
        val = self._merged.get("enableSkillCommands") or self._merged.get("enable_skill_commands")
        return val if val is not None else True

    def get_double_escape_action(self) -> str:
        self._ensure_loaded()
        return self._merged.get("doubleEscapeAction") or self._merged.get("double_escape_action", "tree")

    def get_compaction_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        defaults: dict[str, Any] = {"enabled": True, "reserveTokens": 16384, "keepRecentTokens": 20000}
        override = self._merged.get("compaction") or {}
        return {**defaults, **override}

    def get_block_images(self) -> bool:
        """Get whether images should be blocked from being sent to LLM providers."""
        self._ensure_loaded()
        images = self._merged.get("images") or {}
        return bool(images.get("blockImages", False))

    def set_block_images(self, blocked: bool) -> None:
        """Set whether images should be blocked from being sent to LLM providers."""
        self._ensure_loaded()
        images = dict(self._global_raw.get("images") or {})
        images["blockImages"] = blocked
        self.save_global("images", images)

    def get_retry_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        defaults: dict[str, Any] = {"enabled": True, "maxRetries": 3, "baseDelayMs": 2000, "maxDelayMs": 60000}
        override = self._merged.get("retry") or {}
        return {**defaults, **override}

    def get_terminal_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        defaults: dict[str, Any] = {"showImages": True, "clearOnShrink": False}
        override = self._merged.get("terminal") or {}
        return {**defaults, **override}

    def get_image_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        defaults: dict[str, Any] = {"autoResize": True, "blockImages": False}
        override = self._merged.get("images") or {}
        return {**defaults, **override}

    def get_image_auto_resize(self) -> bool:
        return bool(self.get_image_settings().get("autoResize", True))

    def get_thinking_budgets(self) -> dict[str, Any]:
        self._ensure_loaded()
        return dict(self._merged.get("thinkingBudgets") or self._merged.get("thinking_budgets") or {})

    def get_markdown_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        defaults: dict[str, Any] = {"codeBlockIndent": "  "}
        override = self._merged.get("markdown") or {}
        return {**defaults, **override}

    def get_branch_summary_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        defaults: dict[str, Any] = {"reserveTokens": 16384, "skipPrompt": False}
        override = self._merged.get("branchSummary") or self._merged.get("branch_summary") or {}
        return {**defaults, **override}

    def get_branch_summary_skip_prompt(self) -> bool:
        return bool(self.get_branch_summary_settings().get("skipPrompt", False))

    _TREE_FILTER_VALID = {"default", "no-tools", "user-only", "labeled-only", "all"}

    def get_tree_filter_mode(self) -> str:
        self._ensure_loaded()
        mode = self._merged.get("treeFilterMode") or self._merged.get("tree_filter_mode")
        return mode if mode in self._TREE_FILTER_VALID else "default"

    def get_enabled_models(self) -> list[str] | None:
        self._ensure_loaded()
        val = self._merged.get("enabledModels") or self._merged.get("enabled_models")
        return list(val) if isinstance(val, list) else None

    def get_packages(self) -> list[Any]:
        self._ensure_loaded()
        val = self._merged.get("packages") or []
        return list(val) if isinstance(val, list) else []

    def get_extensions(self) -> list[str]:
        self._ensure_loaded()
        val = self._merged.get("extensions") or []
        return list(val) if isinstance(val, list) else []

    def get_skills(self) -> list[str]:
        self._ensure_loaded()
        val = self._merged.get("skills") or []
        return list(val) if isinstance(val, list) else []

    def get_prompts(self) -> list[str]:
        self._ensure_loaded()
        val = self._merged.get("prompts") or []
        return list(val) if isinstance(val, list) else []

    def get_themes(self) -> list[str]:
        self._ensure_loaded()
        val = self._merged.get("themes") or []
        return list(val) if isinstance(val, list) else []

    def _merged_get(self, camel: str, snake: str, default: Any = None) -> Any:
        self._ensure_loaded()
        if camel in self._merged and self._merged[camel] is not None:
            return self._merged[camel]
        if snake in self._merged and self._merged[snake] is not None:
            return self._merged[snake]
        return default

    def get_tui_mode(self) -> str:
        mode = self._merged_get("tuiMode", "tui_mode", "regular")
        return mode if mode in ("regular", "fullscreen") else "regular"

    def get_show_hardware_cursor(self) -> bool:
        return bool(self._merged_get("showHardwareCursor", "show_hardware_cursor", False))

    def get_clear_on_shrink(self) -> bool:
        terminal = self.get_terminal_settings()
        return bool(terminal.get("clearOnShrink", False))

    def get_hide_thinking_block(self) -> bool:
        return bool(self._merged_get("hideThinkingBlock", "hide_thinking_block", False))

    def get_output_pad(self) -> int:
        value = self._merged_get("outputPad", "output_pad", 1)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 1

    def get_editor_padding_x(self) -> int:
        value = self._merged_get("editorPaddingX", "editor_padding_x", 1)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 1

    def get_autocomplete_max_visible(self) -> int:
        value = self._merged_get("autocompleteMaxVisible", "autocomplete_max_visible", 5)
        try:
            return max(3, min(20, int(value)))
        except (TypeError, ValueError):
            return 5

    def get_fullscreen_scrollbar(self) -> str:
        value = self._merged_get("fullscreenScrollbar", "fullscreen_scrollbar", "auto")
        return value if value in ("hidden", "auto", "always") else "auto"

    def get_fullscreen_exit_output(self) -> str:
        value = self._merged_get("fullscreenExitOutput", "fullscreen_exit_output", "none")
        return value if value in ("none", "transcript") else "none"

    def get_collapse_changelog(self) -> bool:
        return bool(self._merged_get("collapseChangelog", "collapse_changelog", False))

    def get_last_changelog_version(self) -> str | None:
        return self._merged_get("lastChangelogVersion", "last_changelog_version")

    def get_show_terminal_progress(self) -> bool:
        terminal = self.get_terminal_settings()
        return bool(terminal.get("showProgress", True))

    def get_mermaid_rendering_mode(self) -> str:
        markdown = self.get_markdown_settings()
        return str(markdown.get("mermaid", "off"))

    # ── Typed setters ─────────────────────────────────────────────────────────

    def set_packages(self, packages: list[Any]) -> None:
        """Set global package sources list."""
        self.save_global("packages", list(packages))

    def set_project_packages(self, packages: list[Any]) -> None:
        """Set project package sources list."""
        self.save_project("packages", list(packages))

    def update_global(self, **kwargs: Any) -> None:
        """Update specific global settings fields."""
        self._ensure_loaded()
        for k, v in kwargs.items():
            self._global_raw[k] = v
        self._rebuild()
        self._write_file(self._global_settings_file, self._global_raw)

    def update_project(self, **kwargs: Any) -> None:
        """Update specific project settings fields."""
        self._ensure_loaded()
        for k, v in kwargs.items():
            self._project_raw[k] = v
        self._rebuild()
        self._write_file(self._project_settings_file, self._project_raw)
