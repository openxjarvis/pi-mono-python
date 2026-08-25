"""Package resource enable/disable selector. Mirrors config-selector.ts."""
from __future__ import annotations

import os
from typing import Any, Callable, Literal

from pi_coding_agent.config import CONFIG_DIR_NAME

from .component import Component

ResourceType = Literal["extensions", "skills", "prompts", "themes"]
RESOURCE_TYPES: tuple[ResourceType, ...] = ("extensions", "skills", "prompts", "themes")
RESOURCE_TYPE_LABELS = {
    "extensions": "Extensions",
    "skills": "Skills",
    "prompts": "Prompts",
    "themes": "Themes",
}


def format_base_dir(base_dir: str) -> str:
    home = os.path.expanduser("~")
    if base_dir == home:
        display = "~"
    elif base_dir.startswith(home):
        display = f"~{base_dir[len(home):].replace(os.sep, '/')}"
    else:
        display = base_dir.replace(os.sep, "/")
    return display if display.endswith("/") else f"{display}/"


def get_group_label(metadata: Any, agent_dir: str) -> str:
    origin = getattr(metadata, "origin", None) or (metadata.get("origin") if isinstance(metadata, dict) else None)
    source = getattr(metadata, "source", None) or (metadata.get("source") if isinstance(metadata, dict) else None)
    scope = getattr(metadata, "scope", None) or (metadata.get("scope") if isinstance(metadata, dict) else None)
    base_dir = getattr(metadata, "base_dir", None) or (metadata.get("baseDir") if isinstance(metadata, dict) else None) or (
        metadata.get("base_dir") if isinstance(metadata, dict) else None
    )
    if origin == "package":
        return f"{source} ({scope})"
    if source == "auto":
        if base_dir:
            return f"User ({format_base_dir(base_dir)})" if scope == "user" else f"Project ({format_base_dir(base_dir)})"
        return f"User ({format_base_dir(agent_dir)})" if scope == "user" else f"Project ({CONFIG_DIR_NAME}/)"
    return "User settings" if scope == "user" else "Project settings"


def _resource_fields(resource: Any) -> tuple[str, bool, Any]:
    if isinstance(resource, dict):
        return resource["path"], bool(resource.get("enabled")), resource.get("metadata")
    return resource.path, bool(resource.enabled), resource.metadata


def build_groups(resolved: Any, agent_dir: str) -> list[dict[str, Any]]:
    group_map: dict[str, dict[str, Any]] = {}

    def add(resources: list[Any], resource_type: ResourceType) -> None:
        for resource in resources:
            path, enabled, metadata = _resource_fields(resource)
            origin = getattr(metadata, "origin", None) or metadata.get("origin")
            scope = getattr(metadata, "scope", None) or metadata.get("scope")
            source = getattr(metadata, "source", None) or metadata.get("source")
            base_dir = getattr(metadata, "base_dir", None) or metadata.get("baseDir") or metadata.get("base_dir") or ""
            group_key = f"{origin}:{scope}:{source}:{base_dir}"
            if group_key not in group_map:
                group_map[group_key] = {
                    "key": group_key,
                    "label": get_group_label(metadata, agent_dir),
                    "scope": scope,
                    "origin": origin,
                    "source": source,
                    "subgroups": [],
                }
            group = group_map[group_key]
            subgroup_key = f"{group_key}:{resource_type}"
            subgroup = next((item for item in group["subgroups"] if item["key"] == subgroup_key), None)
            if subgroup is None:
                subgroup = {"key": subgroup_key, "type": resource_type, "label": RESOURCE_TYPE_LABELS[resource_type], "items": []}
                group["subgroups"].append(subgroup)
            subgroup["items"].append(
                {
                    "path": path,
                    "enabled": enabled,
                    "metadata": metadata,
                    "resourceType": resource_type,
                    "displayName": os.path.basename(path),
                    "groupKey": group_key,
                    "subgroupKey": subgroup_key,
                }
            )

    for resource_type in RESOURCE_TYPES:
        resources = getattr(resolved, resource_type, None)
        if resources is None and isinstance(resolved, dict):
            resources = resolved.get(resource_type) or []
        add(list(resources or []), resource_type)
    return list(group_map.values())


def flatten_items(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for group in groups:
        for subgroup in group["subgroups"]:
            items.extend(subgroup["items"])
    return items


class ConfigSelectorComponent(Component):
    name = "config_selector"

    def __init__(
        self,
        resolved_paths: Any | None = None,
        agent_dir: str | None = None,
        on_toggle: Callable[[dict[str, Any]], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        items: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.agent_dir = agent_dir or os.path.expanduser("~/.pi/agent")
        self.on_toggle = on_toggle or kwargs.get("on_select")
        self.on_cancel = on_cancel
        self.selected_index = 0
        if resolved_paths is not None:
            self.groups = build_groups(resolved_paths, self.agent_dir)
        else:
            self.groups = [{"key": "flat", "label": "Resources", "subgroups": [{"key": "all", "type": "extensions", "label": "Resources", "items": items or []}]}]

    @property
    def items(self) -> list[dict[str, Any]]:
        return flatten_items(self.groups)

    def set_items(self, items: list[dict[str, Any]]) -> None:
        self.groups = [{"key": "flat", "label": "Resources", "subgroups": [{"key": "all", "type": "extensions", "label": "Resources", "items": items}]}]
        self.invalidate()

    def select(self, index: int) -> Any:
        items = self.items
        if not (0 <= index < len(items)):
            return None
        self.selected_index = index
        item = items[index]
        item["enabled"] = not item.get("enabled", False)
        if self.on_toggle:
            self.on_toggle(item)
        self.invalidate()
        return item

    def _render_body(self, width: int) -> str:
        lines = ["Config"]
        index = 0
        for group in self.groups:
            lines.append(group["label"])
            for subgroup in group["subgroups"]:
                lines.append(f"  {subgroup['label']}")
                for item in subgroup["items"]:
                    marker = ">" if index == self.selected_index else " "
                    check = "[x]" if item.get("enabled") else "[ ]"
                    lines.append(f"    {marker} {check} {item.get('displayName') or item.get('path')}")
                    index += 1
        if index == 0:
            lines.append("  (no resources)")
        return "\n".join(lines)
