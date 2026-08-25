"""Scoped models enable/order selector. Mirrors scoped-models-selector.ts."""
from __future__ import annotations

from typing import Any, Callable

from pi_tui.fuzzy import fuzzy_filter

from .component import Component

EnabledIds = list[str] | None


def is_enabled(enabled_ids: EnabledIds, identifier: str) -> bool:
    return enabled_ids is None or identifier in enabled_ids


def toggle(enabled_ids: EnabledIds, identifier: str) -> EnabledIds:
    if enabled_ids is None:
        return [identifier]
    if identifier in enabled_ids:
        return [item for item in enabled_ids if item != identifier]
    return [*enabled_ids, identifier]


def enable_all(enabled_ids: EnabledIds, all_ids: list[str], target_ids: list[str] | None = None) -> EnabledIds:
    if enabled_ids is None:
        return None
    result = list(enabled_ids)
    for identifier in target_ids or all_ids:
        if identifier not in result:
            result.append(identifier)
    if len(result) == len(all_ids) and all(identifier in all_ids for identifier in result):
        return None
    return result


def clear_all(enabled_ids: EnabledIds, all_ids: list[str], target_ids: list[str] | None = None) -> EnabledIds:
    if enabled_ids is None:
        return [identifier for identifier in all_ids if identifier not in (target_ids or all_ids)] if target_ids else []
    targets = set(target_ids or enabled_ids)
    return [identifier for identifier in enabled_ids if identifier not in targets]


def move(enabled_ids: EnabledIds, identifier: str, delta: int) -> EnabledIds:
    if enabled_ids is None:
        return None
    result = list(enabled_ids)
    try:
        index = result.index(identifier)
    except ValueError:
        return result
    new_index = index + delta
    if new_index < 0 or new_index >= len(result):
        return result
    result[index], result[new_index] = result[new_index], result[index]
    return result


def get_sorted_ids(enabled_ids: EnabledIds, all_ids: list[str]) -> list[str]:
    if enabled_ids is None:
        return list(all_ids)
    enabled = set(enabled_ids)
    return [*enabled_ids, *[identifier for identifier in all_ids if identifier not in enabled]]


def _full_id(model: Any) -> str:
    if isinstance(model, dict):
        provider = model.get("provider", "")
        identifier = model.get("id", "")
        return f"{provider}/{identifier}" if provider else str(identifier)
    provider = getattr(model, "provider", "")
    identifier = getattr(model, "id", str(model))
    return f"{provider}/{identifier}" if provider else str(identifier)


class ScopedModelsSelectorComponent(Component):
    name = "scoped_models_selector"

    def __init__(
        self,
        all_models: list[Any] | None = None,
        enabled_model_ids: EnabledIds = None,
        on_change: Callable[[EnabledIds], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        query: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.all_models = list(all_models or kwargs.get("items") or [])
        self.enabled_model_ids = enabled_model_ids
        self.on_change = on_change
        self.on_cancel = on_cancel
        self.query = query
        self.selected_index = 0

    @property
    def items(self) -> list[dict[str, Any]]:
        ids = get_sorted_ids(self.enabled_model_ids, [_full_id(model) for model in self.all_models])
        by_id = {_full_id(model): model for model in self.all_models}
        models = [{"fullId": identifier, "model": by_id.get(identifier), "enabled": is_enabled(self.enabled_model_ids, identifier)} for identifier in ids]
        if not self.query.strip():
            return models
        return fuzzy_filter(models, self.query, lambda item: item["fullId"])

    def set_items(self, items: list[Any]) -> None:
        self.all_models = list(items)
        self.invalidate()

    def toggle_current(self) -> EnabledIds:
        items = self.items
        if not items:
            return self.enabled_model_ids
        self.enabled_model_ids = toggle(self.enabled_model_ids, items[self.selected_index]["fullId"])
        if self.on_change:
            self.on_change(self.enabled_model_ids)
        self.invalidate()
        return self.enabled_model_ids

    def enable_all(self) -> EnabledIds:
        self.enabled_model_ids = enable_all(self.enabled_model_ids, [_full_id(model) for model in self.all_models])
        if self.on_change:
            self.on_change(self.enabled_model_ids)
        self.invalidate()
        return self.enabled_model_ids

    def clear_all(self) -> EnabledIds:
        self.enabled_model_ids = clear_all(self.enabled_model_ids, [_full_id(model) for model in self.all_models])
        if self.on_change:
            self.on_change(self.enabled_model_ids)
        self.invalidate()
        return self.enabled_model_ids

    def move_current(self, delta: int) -> EnabledIds:
        items = self.items
        if not items:
            return self.enabled_model_ids
        self.enabled_model_ids = move(self.enabled_model_ids, items[self.selected_index]["fullId"], delta)
        if self.on_change:
            self.on_change(self.enabled_model_ids)
        self.invalidate()
        return self.enabled_model_ids

    def select(self, index: int) -> Any:
        self.selected_index = index
        return self.toggle_current()

    def _render_body(self, width: int) -> str:
        lines = ["Scoped models"]
        items = self.items
        if not items:
            lines.append("  (none)")
            return "\n".join(lines)
        for index, item in enumerate(items):
            marker = ">" if index == self.selected_index else " "
            check = "[x]" if item["enabled"] else "[ ]"
            lines.append(f"  {marker} {check} {item['fullId']}")
        return "\n".join(lines)
