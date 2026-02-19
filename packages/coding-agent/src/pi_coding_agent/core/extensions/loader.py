"""
Extension loader for Python extensions.

Python extensions are Python modules (files or packages) that export an
`extension_factory(api)` function. Unlike the TS version which uses jiti to
dynamically load TypeScript, here we use importlib to load .py files.

Mirrors core/extensions/loader.ts
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Callable

from pi_coding_agent.core.extensions.types import (
    Extension,
    ExtensionAPI,
    ExtensionFactory,
    ExtensionFlag,
    ExtensionRuntime,
    ExtensionShortcut,
    LoadExtensionsResult,
    ProviderConfig,
    RegisteredCommand,
    ToolDefinition,
)


def create_extension_runtime() -> ExtensionRuntime:
    """Create a bare ExtensionRuntime with stub actions that raise if called prematurely."""
    return ExtensionRuntime()


def _expand_path(p: str) -> str:
    home = os.path.expanduser("~")
    p = p.strip()
    if p.startswith("~/"):
        return os.path.join(home, p[2:])
    if p.startswith("~"):
        return os.path.join(home, p[1:])
    return p


def _resolve_path(ext_path: str, cwd: str) -> str:
    expanded = _expand_path(ext_path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(cwd, expanded))


class _ConcreteExtensionAPI:
    """Concrete implementation of ExtensionAPI for extension loading."""

    def __init__(self, extension: Extension, runtime: ExtensionRuntime, event_bus: Any) -> None:
        self._extension = extension
        self._runtime = runtime
        self._event_bus = event_bus

    def on(self, event: str, handler: Callable) -> None:
        if event not in self._extension.handlers:
            self._extension.handlers[event] = []
        self._extension.handlers[event].append(handler)

    def register_tool(self, tool: ToolDefinition) -> None:
        self._extension.tools[tool.name] = tool

    def register_command(self, name: str, options: dict) -> None:
        cmd = RegisteredCommand(
            name=name,
            description=options.get("description"),
            get_argument_completions=options.get("get_argument_completions"),
            handler=options.get("handler", lambda args, ctx: None),
        )
        self._extension.commands[name] = cmd

    def register_shortcut(self, shortcut: str, options: dict) -> None:
        sc = ExtensionShortcut(
            shortcut=shortcut,
            description=options.get("description"),
            handler=options.get("handler", lambda ctx: None),
            extension_path=self._extension.path,
        )
        self._extension.shortcuts[shortcut] = sc

    def register_flag(self, name: str, options: dict) -> None:
        flag = ExtensionFlag(
            name=name,
            description=options.get("description"),
            type=options.get("type", "boolean"),
            default=options.get("default"),
            extension_path=self._extension.path,
        )
        self._extension.flags[name] = flag
        if name not in self._runtime.flag_values and flag.default is not None:
            self._runtime.flag_values[name] = flag.default

    def get_flag(self, name: str) -> bool | str | None:
        return self._runtime.flag_values.get(name)

    def register_message_renderer(self, custom_type: str, renderer: Callable) -> None:
        self._extension.message_renderers[custom_type] = renderer

    def send_message(self, message: Any, options: dict | None = None) -> None:
        self._runtime.send_message(message, options)

    def send_user_message(self, content: Any, options: dict | None = None) -> None:
        self._runtime.send_user_message(content, options)

    def append_entry(self, custom_type: str, data: Any = None) -> None:
        self._runtime.append_entry(custom_type, data)

    def set_session_name(self, name: str) -> None:
        self._runtime.set_session_name(name)

    def get_session_name(self) -> str | None:
        return self._runtime.get_session_name()

    def set_label(self, entry_id: str, label: str | None) -> None:
        self._runtime.set_label(entry_id, label)

    def register_provider(self, name: str, config: ProviderConfig) -> None:
        self._runtime.pending_provider_registrations.append({"name": name, "config": config})

    @property
    def events(self) -> Any:
        return self._event_bus


def _load_module_from_path(path: str) -> Any:
    """Dynamically import a Python module from a file path."""
    spec = importlib.util.spec_from_file_location("_pi_ext_" + os.path.basename(path).replace(".", "_"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load extension from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def load_extension_from_factory(
    factory: ExtensionFactory,
    cwd: str,
    event_bus: Any,
    runtime: ExtensionRuntime,
    extension_path: str,
) -> Extension:
    """Load an extension from a factory function."""
    extension = Extension(
        path=extension_path,
        resolved_path=extension_path,
    )
    api = _ConcreteExtensionAPI(extension, runtime, event_bus)
    result = factory(api)
    if hasattr(result, "__await__"):
        await result
    return extension


async def load_extension_from_path(
    ext_path: str,
    cwd: str,
    event_bus: Any,
    runtime: ExtensionRuntime,
) -> Extension:
    """Load a single Python extension file."""
    resolved = _resolve_path(ext_path, cwd)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Extension not found: {resolved}")

    mod = _load_module_from_path(resolved)
    factory = getattr(mod, "extension_factory", None)
    if not callable(factory):
        raise ImportError(f"Extension '{resolved}' has no `extension_factory` function")

    extension = Extension(path=ext_path, resolved_path=resolved)
    api = _ConcreteExtensionAPI(extension, runtime, event_bus)
    result = factory(api)
    if hasattr(result, "__await__"):
        await result
    return extension


async def load_extensions(
    paths: list[str],
    cwd: str,
    event_bus: Any,
) -> LoadExtensionsResult:
    """Load multiple extension modules."""
    from pi_coding_agent.core.event_bus import EventBus

    runtime = create_extension_runtime()
    extensions: list[Extension] = []
    errors: list[dict[str, str]] = []

    for ext_path in paths:
        try:
            ext = await load_extension_from_path(ext_path, cwd, event_bus, runtime)
            extensions.append(ext)
        except Exception as e:
            errors.append({"path": ext_path, "error": str(e)})

    return LoadExtensionsResult(extensions=extensions, errors=errors, runtime=runtime)


def discover_and_load_extensions(
    extensions_dir: str,
    cwd: str,
    event_bus: Any,
) -> list[str]:
    """Discover extension .py files in a directory."""
    paths: list[str] = []
    if not os.path.isdir(extensions_dir):
        return paths

    try:
        for entry in sorted(os.scandir(extensions_dir), key=lambda e: e.name):
            if entry.name.startswith(".") or entry.name.startswith("__"):
                continue
            if entry.is_file(follow_symlinks=True) and entry.name.endswith(".py"):
                paths.append(entry.path)
    except OSError:
        pass

    return paths
