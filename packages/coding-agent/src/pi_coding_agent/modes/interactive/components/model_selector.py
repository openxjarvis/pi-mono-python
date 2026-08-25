"""Model selector with search and scoped/all views. Mirrors model-selector.ts."""
from __future__ import annotations

from typing import Any, Callable, Literal

from pi_tui.fuzzy import fuzzy_filter

from .component import Component

ModelScope = Literal["all", "scoped"]


def _model_id(model: Any) -> str:
    if isinstance(model, dict):
        return str(model.get("id") or "")
    return str(getattr(model, "id", model))


def _model_provider(model: Any) -> str:
    if isinstance(model, dict):
        return str(model.get("provider") or "")
    return str(getattr(model, "provider", ""))


def _search_text(model: Any) -> str:
    return f"{_model_provider(model)} {_model_id(model)}".lower()


class ModelSelectorComponent(Component):
    name = "model_selector"

    def __init__(
        self,
        models: list[Any] | None = None,
        current: Any | None = None,
        on_select: Callable[[Any], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        query: str = "",
        scoped_models: list[Any] | None = None,
        default_model: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.all_models = list(models or kwargs.get("models") or [])
        self.scoped_models = list(scoped_models or [])
        self.current = current
        self.on_select = on_select
        self.on_cancel = on_cancel
        self.query = query
        self.default_model = default_model
        self.scope: ModelScope = "scoped" if self.scoped_models else "all"
        self.selected_index = 0
        self._sync_selection()

    @property
    def models(self) -> list[Any]:
        return self.active_models

    @models.setter
    def models(self, value: list[Any]) -> None:
        self.all_models = list(value)
        self._sync_selection()

    @property
    def active_models(self) -> list[Any]:
        source = self.scoped_models if self.scope == "scoped" and self.scoped_models else self.all_models
        if not self.query.strip():
            return source
        return fuzzy_filter(source, self.query, _search_text)

    def toggle_scope(self) -> None:
        if not self.scoped_models:
            return
        self.scope = "all" if self.scope == "scoped" else "scoped"
        self.selected_index = 0
        self._sync_selection()
        self.invalidate()

    def move(self, delta: int) -> None:
        models = self.active_models
        if not models:
            return
        self.selected_index = (self.selected_index + delta) % len(models)
        self.invalidate()

    def set_query(self, query: str) -> None:
        self.query = query
        self.selected_index = 0
        self.invalidate()

    def select_current(self) -> Any | None:
        models = self.active_models
        if not models:
            if self.on_cancel:
                self.on_cancel()
            return None
        model = models[self.selected_index]
        if self.on_select:
            self.on_select(model)
        return model

    def _sync_selection(self) -> None:
        if self.current is None:
            return
        current_id = _model_id(self.current)
        for index, model in enumerate(self.active_models):
            if _model_id(model) == current_id:
                self.selected_index = index
                break

    def _render_body(self, width: int) -> str:
        models = self.active_models
        header = f"Select model  [{self.scope}]"
        if self.query:
            header += f"  /{self.query}"
        lines = [header]
        if not models:
            lines.append("  (none available)")
            return "\n".join(lines)
        for index, model in enumerate(models):
            marker = ">" if index == self.selected_index else " "
            default = ""
            if self.default_model and _model_id(model) == self.default_model.get("id") and _model_provider(model) == self.default_model.get("provider"):
                default = " *"
            lines.append(f"  {marker} {_model_id(model)} ({_model_provider(model)}){default}")
        return "\n".join(lines)
