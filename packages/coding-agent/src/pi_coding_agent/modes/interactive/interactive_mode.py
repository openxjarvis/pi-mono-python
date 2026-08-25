"""
Interactive mode — mirrors modes/interactive/interactive-mode.ts

Composition root for the coding-agent TUI: main-screen / alt-screen renderer,
editor, slash commands, selectors, and AgentSession event wiring.
"""
from __future__ import annotations

import asyncio
import os
import re
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pi_tui import (
    CombinedAutocompleteProvider,
    EditorTheme,
    ProcessTerminal,
    ScrollView,
    SlashCommand,
    Spacer,
    Text,
    TuiAltScreen,
    TuiAltScreenOptions,
    TuiMainScreen,
    VStack,
    is_viewport_tui,
)
from pi_tui.components.editor import EditorOptions
from pi_tui.tui import TUI, Container

from pi_coding_agent.config import APP_TITLE, VERSION, get_agent_dir
from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.agent_session_runtime import AgentSessionRuntime
from pi_coding_agent.core.agent_session_services import AgentSessionServices
from pi_coding_agent.core.defaults import THINKING_LEVEL_OPTIONS
from pi_coding_agent.core.footer_data_provider import FooterDataProvider
from pi_coding_agent.core.keybindings import KeybindingsManager
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.slash_commands import BUILTIN_SLASH_COMMANDS
from pi_coding_agent.core.trust_manager import ProjectTrustStore
from pi_coding_agent.utils.changelog import get_new_entries, parse_changelog
from pi_coding_agent.utils.clipboard import copy_to_clipboard, read_clipboard_text
from pi_coding_agent.utils.clipboard_image import extension_for_image_mime_type, read_clipboard_image
from pi_coding_agent.utils.git import parse_git_url
from pi_coding_agent.utils.open_browser import open_browser

from .components.assistant_message import AssistantMessageComponent
from .components.bash_execution import BashExecutionComponent
from .components.custom_editor import CustomEditor
from .components.dynamic_border import DynamicBorder
from .components.footer import FooterComponent
from .components.login_dialog import LoginDialogComponent
from .components.model_selector import ModelSelectorComponent
from .components.oauth_selector import OAuthSelectorComponent
from .components.scoped_models_selector import ScopedModelsSelectorComponent
from .components.session_selector import SessionSelectorComponent
from .components.settings_selector import SettingsSelectorComponent
from .components.status_indicator import IdleStatus, StatusIndicator, WorkingStatusIndicator
from .components.thinking_selector import ThinkingSelectorComponent
from .components.tree_selector import TreeSelectorComponent
from .components.trust_selector import TrustSelectorComponent
from .components.user_message import UserMessageComponent
from .components.user_message_selector import UserMessageSelectorComponent
from .model_catalog_refresh import refresh_model_catalogs
from .theme.theme import get_theme, set_registered_themes
from .theme.theme_controller import ThemeController


@dataclass
class InteractiveModeOptions:
    migrated_providers: list[str] | None = None
    startup_diagnostics: list[Any] | None = None
    model_fallback_message: str | None = None
    auto_trust_on_reload_cwd: str | None = None
    initial_message: str | None = None
    initial_images: list[Any] | None = None
    initial_messages: list[str] | None = None
    verbose: bool = False
    tui_mode: str | None = None
    initial_theme_setting: str | None = None


@dataclass
class InteractiveTuiOptions:
    tui_mode: str
    show_hardware_cursor: bool
    log_directory: str
    terminal: Any | None = None
    on_right_click_paste: Callable[[], None] | None = None


def format_resume_command(session_manager: SessionManager) -> str | None:
    session_file = session_manager.get_session_file()
    if not session_file:
        return None
    return f"pi --resume {session_file}"


def create_interactive_tui(options: InteractiveTuiOptions | dict[str, Any]) -> TuiMainScreen | TuiAltScreen:
    if isinstance(options, dict):
        options = InteractiveTuiOptions(
            tui_mode=options.get("tui_mode") or options.get("tuiMode") or "regular",
            show_hardware_cursor=bool(options.get("show_hardware_cursor") or options.get("showHardwareCursor")),
            log_directory=options.get("log_directory") or options.get("logDirectory") or get_agent_dir(),
            terminal=options.get("terminal"),
            on_right_click_paste=options.get("on_right_click_paste") or options.get("onRightClickPaste"),
        )
    terminal = options.terminal or ProcessTerminal()
    if options.tui_mode == "fullscreen":
        theme = get_theme()

        def style_search(text: str) -> str:
            return theme.fg("searchMatchText", text) if hasattr(theme, "fg") else text

        async def copy_selection(text: str) -> bool:
            try:
                copy_to_clipboard(text)
                return True
            except Exception:
                return False

        return TuiAltScreen(
            terminal,
            options.show_hardware_cursor,
            options.log_directory,
            TuiAltScreenOptions(
                search_match_style=lambda text: style_search(text),
                search_current_match_style=lambda text: theme.bold(style_search(text)) if hasattr(theme, "bold") else text,
                open_url=open_browser,
                on_right_click_paste=options.on_right_click_paste,
                copy_selection=copy_selection,
            ),
        )
    return TuiMainScreen(terminal, options.show_hardware_cursor, options.log_directory)


def create_interactive_tui_reference(get_tui: Callable[[], TUI]) -> TUI:
    class _TuiProxy:
        def __getattr__(self, name: str) -> Any:
            value = getattr(get_tui(), name)
            if not callable(value):
                return value

            def bound(*args: Any, **kwargs: Any) -> Any:
                current = getattr(get_tui(), name)
                return current(*args, **kwargs)

            return bound

        def __setattr__(self, name: str, value: Any) -> None:
            if name.startswith("_"):
                object.__setattr__(self, name, value)
                return
            setattr(get_tui(), name, value)

    return _TuiProxy()  # type: ignore[return-value]


def _as_runtime(session_or_runtime: AgentSession | AgentSessionRuntime) -> AgentSessionRuntime:
    if isinstance(session_or_runtime, AgentSessionRuntime):
        return session_or_runtime
    session = session_or_runtime
    services = AgentSessionServices(
        cwd=session.cwd,
        agent_dir=get_agent_dir(),
        model_runtime=session.model_runtime or ModelRuntime(),
        settings_manager=session.settings_manager,
        resource_loader=session.resource_loader,
    )
    return AgentSessionRuntime(session, services)


class InteractiveMode:
    """Interactive TUI host. Mirrors the TypeScript InteractiveMode class."""

    def __init__(
        self,
        runtime_host: AgentSession | AgentSessionRuntime,
        options: InteractiveModeOptions | dict[str, Any] | None = None,
    ) -> None:
        if isinstance(options, dict):
            options = InteractiveModeOptions(
                migrated_providers=options.get("migrated_providers") or options.get("migratedProviders"),
                startup_diagnostics=options.get("startup_diagnostics") or options.get("startupDiagnostics"),
                model_fallback_message=options.get("model_fallback_message") or options.get("modelFallbackMessage"),
                auto_trust_on_reload_cwd=options.get("auto_trust_on_reload_cwd") or options.get("autoTrustOnReloadCwd"),
                initial_message=options.get("initial_message") or options.get("initialMessage"),
                initial_images=options.get("initial_images") or options.get("initialImages"),
                initial_messages=options.get("initial_messages") or options.get("initialMessages"),
                verbose=bool(options.get("verbose")),
                tui_mode=options.get("tui_mode") or options.get("tuiMode"),
                initial_theme_setting=options.get("initial_theme_setting") or options.get("initialThemeSetting"),
            )
        self.options = options or InteractiveModeOptions()
        self.runtime_host = _as_runtime(runtime_host)
        settings = self.settings_manager
        tui_mode = self.options.tui_mode or settings.get_tui_mode()
        self.options.tui_mode = tui_mode
        self.auto_trust_on_reload_cwd = self.options.auto_trust_on_reload_cwd
        self.version = VERSION
        self.renderer = create_interactive_tui(
            InteractiveTuiOptions(
                tui_mode=tui_mode,
                show_hardware_cursor=settings.get_show_hardware_cursor(),
                log_directory=get_agent_dir(),
                on_right_click_paste=self._on_right_click_paste,
            )
        )
        self.ui = create_interactive_tui_reference(lambda: self.renderer)
        if hasattr(self.ui, "set_clear_on_shrink"):
            self.ui.set_clear_on_shrink(settings.get_clear_on_shrink())

        self.header_container = Container()
        self.loaded_resources_container = Container()
        self.chat_container = Container()
        self.document_container = Container()
        self.document_container.add_child(self.header_container)
        self.document_container.add_child(self.loaded_resources_container)
        self.document_container.add_child(self.chat_container)
        self.pending_messages_container = Container()
        self.status_container = Container()
        self.widget_container_above = Container()
        self.widget_container_below = Container()
        self.footer_container = Container()
        self.editor_container = Container()

        self.keybindings = KeybindingsManager.create()
        self.default_editor = CustomEditor(
            self.ui,
            EditorTheme(),
            self.keybindings,
            EditorOptions(
                padding_x=settings.get_editor_padding_x(),
                autocomplete_max_visible=settings.get_autocomplete_max_visible(),
            ),
        )
        self.editor = self.default_editor
        self.editor_container.add_child(self.editor)
        self.footer_data_provider = FooterDataProvider(self.session_manager.get_cwd())
        self.footer = FooterComponent(self.session, self.footer_data_provider)
        if hasattr(self.footer, "set_auto_compact_enabled"):
            self.footer.set_auto_compact_enabled(self.session.auto_compaction_enabled)
        self.footer_container.add_child(self.footer)

        self.hide_thinking_block = settings.get_hide_thinking_block()
        self.output_pad = settings.get_output_pad()
        loader = getattr(self.session, "resource_loader", None)
        if loader and hasattr(loader, "get_themes"):
            themes = loader.get_themes()
            set_registered_themes(getattr(themes, "themes", themes) if themes else [])
        self.theme_controller = ThemeController(on_change=lambda _theme: self.update_editor_border_color())

        self.is_initialized = False
        self.shutdown_requested = False
        self.startup_notices_shown = False
        self.changelog_markdown: str | None = None
        self.pending_user_inputs: list[str] = []
        self._input_ready = asyncio.Event()
        self.on_input_callback: Callable[[str], None] | None = None
        self.active_status_indicator: StatusIndicator | None = None
        self.idle_status = IdleStatus()
        self.working_visible = True
        self.working_message: str | None = None
        self.default_working_message = "Working..."
        self.tool_output_expanded = False
        self.is_bash_mode = False
        self.last_escape_time = 0.0
        self.last_sigint_time = 0.0
        self.skill_commands: dict[str, str] = {}
        self.pending_tools: dict[str, Any] = {}
        self.streaming_component: AssistantMessageComponent | None = None
        self.unsubscribe: Callable[[], None] | None = None
        self.signal_cleanup_handlers: list[Callable[[], None]] = []
        self.main_screen_render_state = None
        self.transcript_scroll_view: ScrollView | None = None
        self.fullscreen_layout_root: Any | None = None
        self.autocomplete_provider = None
        self.active_selector_dispose: Callable[[], None] | None = None

    @property
    def session(self) -> AgentSession:
        return self.runtime_host.session

    @property
    def session_manager(self) -> SessionManager:
        return self.session.session_manager

    @property
    def settings_manager(self) -> Any:
        return self.session.settings_manager

    def _on_right_click_paste(self) -> None:
        asyncio.create_task(self.handle_right_click_paste())

    def get_autocomplete_source_tag(self, source_info: Any | None) -> str | None:
        if not source_info:
            return None
        scope = getattr(source_info, "scope", None) or (source_info.get("scope") if isinstance(source_info, dict) else None)
        source = getattr(source_info, "source", None) or (source_info.get("source") if isinstance(source_info, dict) else "")
        source = (source or "").strip()
        prefix = "u" if scope == "user" else "p" if scope == "project" else "t"
        if source in ("auto", "local", "cli", ""):
            return prefix
        if source.startswith("npm:"):
            return f"{prefix}:{source}"
        git_source = parse_git_url(source) if source else None
        if git_source:
            ref = f"@{getattr(git_source, 'ref', '')}" if getattr(git_source, "ref", None) else ""
            host = getattr(git_source, "host", "")
            path = getattr(git_source, "path", "")
            return f"{prefix}:git:{host}/{path}{ref}"
        return prefix

    def create_base_autocomplete_provider(self) -> CombinedAutocompleteProvider:
        slash_commands = [
            SlashCommand(
                name=command.name,
                description=command.description,
                argument_hint=command.argument_hint,
            )
            for command in BUILTIN_SLASH_COMMANDS
        ]
        return CombinedAutocompleteProvider(commands=slash_commands, base_path=self.session_manager.get_cwd())

    def setup_autocomplete_provider(self) -> None:
        provider = self.create_base_autocomplete_provider()
        self.autocomplete_provider = provider
        self.default_editor.set_autocomplete_provider(provider)

    def mount_interactive_tui(self, tui: TuiMainScreen | TuiAltScreen, components: list[Any]) -> None:
        for component in components:
            tui.add_child(component)
        if is_viewport_tui(tui) and self.fullscreen_layout_root is not None and hasattr(tui, "set_layout_root"):
            tui.set_layout_root(self.fullscreen_layout_root)

    def switch_tui_mode(self, mode: str, restore_progress: bool = True, start_renderer: bool = True) -> bool:
        previous = self.renderer
        if getattr(previous, "mode", None) == mode:
            return True
        if getattr(previous, "has_overlay_entries", False):
            return False
        components = list(getattr(previous, "children", []))
        focus = previous.get_focused_component() if hasattr(previous, "get_focused_component") else None
        terminal = previous.terminal
        show_cursor = previous.get_show_hardware_cursor() if hasattr(previous, "get_show_hardware_cursor") else False
        if isinstance(previous, TuiMainScreen):
            self.main_screen_render_state = previous.capture_render_state()
        previous.stop({"preserve_screen": True} if hasattr(previous, "stop") else None)
        if hasattr(previous, "set_focus"):
            previous.set_focus(None)
        if hasattr(previous, "clear"):
            previous.clear()
        next_ui = create_interactive_tui(
            InteractiveTuiOptions(
                tui_mode=mode,
                show_hardware_cursor=show_cursor,
                log_directory=get_agent_dir(),
                terminal=terminal,
                on_right_click_paste=self._on_right_click_paste,
            )
        )
        if isinstance(next_ui, TuiMainScreen) and self.main_screen_render_state:
            next_ui.restore_render_state(self.main_screen_render_state)
        self.renderer = next_ui
        self.options.tui_mode = mode
        self.mount_interactive_tui(next_ui, components)
        next_ui.invalidate()
        if hasattr(next_ui, "set_focus"):
            next_ui.set_focus(focus)
        if start_renderer:
            next_ui.start()
        return True

    def update_terminal_title(self) -> None:
        cwd_basename = os.path.basename(self.session_manager.get_cwd())
        session_name = self.session_manager.get_session_name()
        title = f"{APP_TITLE} - {session_name} - {cwd_basename}" if session_name else f"{APP_TITLE} - {cwd_basename}"
        if hasattr(self.ui.terminal, "set_title"):
            self.ui.terminal.set_title(title)

    def get_changelog_for_display(self) -> str | None:
        messages = getattr(getattr(self.session, "state", None), "messages", None) or self.session.messages
        if messages:
            return None
        last_version = self.settings_manager.get_last_changelog_version()
        changelog_path = Path(__file__).resolve().parents[5] / "CHANGELOG.md"
        if not changelog_path.exists():
            return None
        text = changelog_path.read_text(encoding="utf-8")
        entries = get_new_entries(parse_changelog(text), last_version) if last_version else parse_changelog(text)
        if not entries:
            return None
        return "\n\n".join(f"## {entry.version}\n{entry.content}" for entry in entries[:3])

    def show_startup_notices_if_needed(self) -> None:
        if self.startup_notices_shown or not self.changelog_markdown:
            return
        self.startup_notices_shown = True
        self.chat_container.add_child(DynamicBorder())
        if self.settings_manager.get_collapse_changelog():
            match = re.search(r"##\s+\[?(\d+\.\d+\.\d+)\]?", self.changelog_markdown)
            latest = match.group(1) if match else self.version
            self.chat_container.add_child(Text(f"Updated to v{latest}. Use /changelog to view full changelog.", 1, 0))
        else:
            self.chat_container.add_child(Text(get_theme().bold("What's New"), 1, 0))
            self.chat_container.add_child(Text(self.changelog_markdown.strip(), 1, 0))
        self.chat_container.add_child(DynamicBorder())

    async def init(self) -> None:
        if self.is_initialized:
            return
        self.register_signal_handlers()
        self.changelog_markdown = self.get_changelog_for_display()
        self.setup_autocomplete_provider()
        self.setup_key_handlers()
        self.setup_editor_submit_handler()
        self.subscribe_to_agent()
        self.transcript_scroll_view = ScrollView(self.document_container, {"follow": "end", "primary": True})
        dock = VStack(
            [
                self.transcript_scroll_view,
                self.widget_container_above,
                self.editor_container,
                self.widget_container_below,
                self.pending_messages_container,
                self.status_container,
                self.footer_container,
            ]
        )
        self.fullscreen_layout_root = dock
        self.mount_interactive_tui(
            self.renderer,
            [
                self.document_container,
                self.widget_container_above,
                self.editor_container,
                self.widget_container_below,
                self.pending_messages_container,
                self.status_container,
                self.footer_container,
            ],
        )
        if is_viewport_tui(self.renderer):
            self.renderer.set_layout_root(dock)
        self.ui.set_focus(self.editor)
        self.update_terminal_title()
        self.show_startup_notices_if_needed()
        self.render_initial_messages()
        self.is_initialized = True

    async def run(self) -> None:
        await self.init()
        if not os.environ.get("PI_OFFLINE") and self.session.model_runtime:
            try:
                await refresh_model_catalogs(self.session.model_runtime)
            except Exception:
                pass
        for diagnostic in self.options.startup_diagnostics or []:
            kind = getattr(diagnostic, "type", None) or (diagnostic.get("type") if isinstance(diagnostic, dict) else "")
            message = getattr(diagnostic, "message", None) or (diagnostic.get("message") if isinstance(diagnostic, dict) else str(diagnostic))
            if kind == "error":
                self.show_error(message)
            elif kind == "warning":
                self.show_warning(message)
            else:
                self.show_status(message)
        if self.options.migrated_providers:
            self.show_warning(f"Migrated credentials to auth.json: {', '.join(self.options.migrated_providers)}")
        if self.options.model_fallback_message:
            self.show_warning(self.options.model_fallback_message)
        if self.options.initial_message:
            try:
                await self.session.prompt(self.options.initial_message, images=self.options.initial_images)
            except Exception as exc:
                self.show_error(str(exc))
        for message in self.options.initial_messages or []:
            try:
                await self.session.prompt(message)
            except Exception as exc:
                self.show_error(str(exc))
        self.ui.start()
        while not self.shutdown_requested:
            user_input = await self.get_user_input()
            if user_input is None:
                break
            try:
                await self.session.prompt(user_input)
            except Exception as exc:
                self.show_error(str(exc))

    async def get_user_input(self) -> str | None:
        if self.pending_user_inputs:
            return self.pending_user_inputs.pop(0)
        self._input_ready.clear()
        await self._input_ready.wait()
        if self.shutdown_requested:
            return None
        if self.pending_user_inputs:
            return self.pending_user_inputs.pop(0)
        return None

    def render_initial_messages(self) -> None:
        for message in self.session.messages:
            self.add_message_to_chat(message)

    def add_message_to_chat(self, message: Any, options: dict[str, Any] | None = None) -> None:
        role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None)
        if role == "user":
            self.chat_container.add_child(UserMessageComponent(message=message))
        elif role == "assistant":
            self.chat_container.add_child(AssistantMessageComponent(message=message))
        if hasattr(self.ui, "request_render"):
            self.ui.request_render()

    def add_custom_entry_to_chat(self, entry: Any) -> None:
        self.chat_container.add_child(Text(str(entry), 1, 0))

    def setup_key_handlers(self) -> None:
        editor = self.default_editor
        editor.on_escape = self._handle_escape
        editor.on_ctrl_d = self.handle_ctrl_d
        editor.on_action("app.clear", self.handle_ctrl_c)
        editor.on_action("app.suspend", self.handle_ctrl_z)
        editor.on_action("app.thinking.cycle", self.cycle_thinking_level)
        editor.on_action("app.model.cycleForward", lambda: self.cycle_model("forward"))
        editor.on_action("app.model.cycleBackward", lambda: self.cycle_model("backward"))
        editor.on_action("app.model.select", self.show_model_selector)
        editor.on_action("app.tools.expand", self.toggle_tool_output_expansion)
        editor.on_action("app.thinking.toggle", self.toggle_thinking_block_visibility)
        editor.on_action("app.session.new", self.handle_clear_command)
        editor.on_action("app.session.tree", self.show_tree_selector)
        editor.on_action("app.session.fork", self.show_user_message_selector)
        editor.on_action("app.session.resume", self.show_session_selector)
        editor.on_action("app.message.copy", lambda: asyncio.create_task(self.handle_copy_command({"flashConfirmation": True})))
        editor.on_action("app.message.dequeue", self.handle_dequeue)
        editor.on_paste_image = lambda: asyncio.create_task(self.handle_clipboard_paste())
        editor.on_change = self._on_editor_change

    def _on_editor_change(self, text: str) -> None:
        was_bash = self.is_bash_mode
        self.is_bash_mode = text.lstrip().startswith("!")
        if was_bash != self.is_bash_mode:
            self.update_editor_border_color()

    def _handle_escape(self) -> None:
        if self.session.is_streaming:
            self.restore_queued_messages_to_editor(abort=True)
        elif getattr(self.session, "is_bash_running", False):
            if hasattr(self.session, "abort_bash"):
                self.session.abort_bash()
        elif self.is_bash_mode:
            self.editor.set_text("")
            self.is_bash_mode = False
            self.update_editor_border_color()
        elif not self.editor.get_text().strip():
            action = self.settings_manager.get_double_escape_action()
            if action != "none":
                now = asyncio.get_event_loop().time() * 1000
                if now - self.last_escape_time < 500:
                    if action == "tree":
                        self.show_tree_selector()
                    else:
                        self.show_user_message_selector()
                    self.last_escape_time = 0
                else:
                    self.last_escape_time = now

    def setup_editor_submit_handler(self) -> None:
        async def on_submit(text: str) -> None:
            text = text.strip()
            if not text:
                return
            if await self._handle_slash_command(text):
                return
            self.pending_user_inputs.append(text)
            self._input_ready.set()

        def sync_submit(text: str) -> None:
            asyncio.create_task(on_submit(text))

        self.default_editor.on_submit = sync_submit

    async def _handle_slash_command(self, text: str) -> bool:
        if not text.startswith("/"):
            return False
        command, _, rest = text[1:].partition(" ")
        rest = rest.strip()
        handlers = {
            "quit": self.stop,
            "exit": self.stop,
            "settings": self.show_settings_selector,
            "model": lambda: self.handle_model_command(rest),
            "tree": self.show_tree_selector,
            "thinking": lambda: self.handle_thinking_command(rest),
            "scoped-models": self.show_models_selector,
            "session": self.handle_session_command,
            "changelog": self.handle_changelog_command,
            "hotkeys": self.handle_hotkeys_command,
            "fork": self.show_user_message_selector,
            "clone": lambda: asyncio.create_task(self.session.clone()),
            "trust": self.show_trust_selector,
            "login": lambda: self.show_login_provider_selector(rest or None),
            "new": self.handle_clear_command,
            "compact": lambda: asyncio.create_task(self.session.compact()),
            "resume": self.show_session_selector,
            "import": lambda: asyncio.create_task(self._handle_import(rest)),
            "name": lambda: self.handle_name_command(rest),
            "export": lambda: asyncio.create_task(self.handle_export_command(text)),
            "share": lambda: asyncio.create_task(self.handle_share_command()),
            "copy": lambda: asyncio.create_task(self.handle_copy_command()),
            "reload": lambda: asyncio.create_task(self.handle_reload_command()),
            "logout": lambda: self.handle_logout_command(rest),
            "debug": self.handle_debug_command,
        }
        handler = handlers.get(command)
        if handler is None:
            return False
        result = handler()
        if asyncio.iscoroutine(result):
            await result
        return True

    async def _switch_tree_branch(self, entry_id: str) -> None:
        try:
            await self.session.navigate_tree(entry_id)
            if hasattr(self.chat_container, "clear"):
                self.chat_container.clear()
            self.render_initial_messages()
            self.show_status(f"Switched to {entry_id}")
        except Exception as exc:
            self.show_error(f"Failed to switch branch: {exc}")

    async def _handle_import(self, path: str) -> None:
        if not path:
            self.show_error("Usage: /import <path.jsonl>")
            return
        if hasattr(self.session, "import_from_jsonl"):
            await self.session.import_from_jsonl(path)
            self.show_status(f"Imported {path}")

    def _path_command_argument(self, text: str, command: str) -> str | None:
        if text == command:
            return None
        prefix = f"{command} "
        if not text.startswith(prefix):
            return None
        args = text[len(prefix):].lstrip()
        if not args:
            return None
        quote = args[0]
        if quote in {'"', "'"}:
            end = args.find(quote, 1)
            return None if end < 0 else args[1:end]
        whitespace = args.find(" ")
        return args if whitespace < 0 else args[:whitespace]

    async def handle_export_command(self, text: str) -> None:
        output_path = self._path_command_argument(text, "/export")
        try:
            if output_path and output_path.endswith(".jsonl"):
                from pi_coding_agent.core.session_export import export_session_to_jsonl

                file_path = export_session_to_jsonl(self.session_manager, output_path)
            elif hasattr(self.session, "export_to_html"):
                file_path = await self.session.export_to_html(output_path)
            else:
                self.show_error("Session export is not available")
                return
            self.show_status(f"Session exported to: {file_path}")
        except Exception as exc:
            self.show_error(f"Failed to export session: {exc}")

    async def handle_share_command(self) -> None:
        self.show_error("Session sharing is not available in the Python port yet.")

    async def handle_copy_command(self, options: dict[str, Any] | None = None) -> None:
        text = self.session.get_last_assistant_text() if hasattr(self.session, "get_last_assistant_text") else None
        if not text:
            self.show_error("No agent messages to copy yet.")
            return
        try:
            copy_to_clipboard(text)
            if (options or {}).get("flashConfirmation") and hasattr(self.ui, "flash"):
                self.ui.flash("Copied!")
            else:
                self.show_status("Copied last agent message to clipboard")
        except Exception as exc:
            self.show_error(str(exc))

    async def handle_reload_command(self) -> None:
        if getattr(self.session, "is_streaming", False):
            self.show_warning("Wait for the current response to finish before reloading.")
            return
        if getattr(self.session, "is_compacting", False):
            self.show_warning("Wait for compaction to finish before reloading.")
            return
        try:
            if hasattr(self.session, "reload"):
                await self.session.reload()
            if hasattr(self.keybindings, "reload"):
                self.keybindings.reload()
            self.show_status("Reloaded keybindings, extensions, skills, prompts, themes, and context files")
        except Exception as exc:
            self.show_error(f"Reload failed: {exc}")

    def handle_logout_command(self, provider: str = "") -> None:
        storage = getattr(self.session, "_auth_storage", None)
        if storage is None:
            self.show_error("No stored credentials to remove.")
            return
        providers = [provider] if provider else list(getattr(storage, "list_stored_providers", lambda: [])())
        if not providers:
            self.show_error(
                "No stored credentials to remove. /logout only removes credentials saved by /login; "
                "environment variables and models.json config are unchanged."
            )
            return
        for name in providers:
            storage.logout(name)
        self.show_status(f"Logged out: {', '.join(providers)}")

    def handle_debug_command(self) -> None:
        stats = self.session.get_session_stats() if hasattr(self.session, "get_session_stats") else {}
        self.show_status(f"debug session={getattr(self.session, 'session_id', '?')} stats={stats}")

    def handle_model_command(self, search: str = "") -> None:
        if search and hasattr(self.session, "set_model"):
            try:
                self.session.set_model(search)
                self.show_status(f"Model: {search}")
                return
            except Exception as exc:
                self.show_warning(str(exc))
        self.show_model_selector(search or None)

    def handle_thinking_command(self, search_term: str = "") -> None:
        if search_term:
            available = self.session.get_available_thinking_levels()
            match = next((level for level in available if str(level).lower() == search_term.lower()), None)
            if match:
                self.select_thinking_level(match, persist=False)
                return
            self.show_error(f'Unknown thinking level "{search_term}". Available: {", ".join(map(str, available))}.')
            return
        self.show_thinking_selector()

    def select_thinking_level(self, level: Any, persist: bool) -> None:
        self.session.set_thinking_level(level, persist=persist)
        self.show_status(f"Thinking level: {level}")

    def cycle_thinking_level(self) -> None:
        if hasattr(self.session, "cycle_thinking_level"):
            level = self.session.cycle_thinking_level()
            if level:
                self.show_status(f"Thinking level: {level}")

    def cycle_model(self, direction: str = "forward") -> None:
        if hasattr(self.session, "cycle_model"):
            model = self.session.cycle_model(direction)
            if model:
                self.show_status(f"Model: {getattr(model, 'id', model)}")

    def handle_session_command(self) -> None:
        stats = self.session.get_session_stats() if hasattr(self.session, "get_session_stats") else {}
        self.show_status(
            f"Session {stats.get('sessionId', self.session.session_id)}  "
            f"tokens={stats.get('tokens', {}).get('total', 0)}"
        )

    def handle_changelog_command(self) -> None:
        markdown = self.changelog_markdown or self.get_changelog_for_display() or "No changelog entries."
        self.chat_container.add_child(Text(markdown, 1, 0))
        self.ui.request_render()

    def handle_hotkeys_command(self) -> None:
        lines = [f"  {action}: {', '.join(self.keybindings.get_keys(action))}" for action in (
            "app.interrupt", "app.clear", "app.exit", "app.model.cycleForward", "app.model.select", "app.thinking.cycle"
        )]
        self.chat_container.add_child(Text("Hotkeys\n" + "\n".join(lines), 1, 0))
        self.ui.request_render()

    def handle_name_command(self, name: str) -> None:
        if name and hasattr(self.session, "set_session_name"):
            self.session.set_session_name(name)
            self.update_terminal_title()
            self.show_status(f"Session name: {name}")

    def handle_clear_command(self) -> None:
        asyncio.create_task(self.session.new_session())
        self.chat_container.clear()
        self.show_status("New session")

    def handle_ctrl_c(self) -> None:
        if self.session.is_streaming and hasattr(self.session, "abort"):
            self.session.abort()
            return
        now = asyncio.get_event_loop().time()
        if now - self.last_sigint_time < 0.8:
            self.stop()
            return
        self.last_sigint_time = now
        self.show_status("Press Ctrl+C again to exit")

    def handle_ctrl_d(self) -> None:
        if not self.editor.get_text():
            self.stop()

    def handle_ctrl_z(self) -> None:
        if hasattr(os, "kill") and hasattr(signal, "SIGTSTP"):
            os.kill(os.getpid(), signal.SIGTSTP)

    def handle_dequeue(self) -> None:
        queued = self.restore_queued_messages_to_editor()
        if queued:
            self.show_status(f"Restored {queued} queued message(s)")

    def restore_queued_messages_to_editor(self, abort: bool = False, current_text: str | None = None) -> int:
        if abort and hasattr(self.session, "abort"):
            self.session.abort()
        messages = []
        if hasattr(self.session, "peek_steering_messages"):
            messages.extend(self.session.peek_steering_messages())
        if hasattr(self.session, "peek_follow_up_messages"):
            messages.extend(self.session.peek_follow_up_messages())
        if not messages:
            return 0
        text = "\n".join(str(item) for item in messages)
        if current_text:
            text = f"{current_text}\n{text}"
        self.editor.set_text(text)
        return len(messages)

    def update_editor_border_color(self) -> None:
        color = "\x1b[32m" if self.is_bash_mode else "\x1b[2m"
        if hasattr(self.editor, "_theme"):
            self.editor._theme.border_color = lambda s, c=color: f"{c}{s}\x1b[22m\x1b[39m"
        self.ui.request_render()

    def toggle_tool_output_expansion(self) -> None:
        self.tool_output_expanded = not self.tool_output_expanded
        self.show_status("Tools expanded" if self.tool_output_expanded else "Tools collapsed")

    def toggle_thinking_block_visibility(self) -> None:
        self.hide_thinking_block = not self.hide_thinking_block
        self.show_status("Thinking hidden" if self.hide_thinking_block else "Thinking visible")

    def clear_editor(self) -> None:
        self.editor.set_text("")

    def show_error(self, error_message: str) -> None:
        self.chat_container.add_child(Text(f"\x1b[31m{error_message}\x1b[39m", 1, 0))
        self.ui.request_render()

    def show_warning(self, warning_message: str) -> None:
        self.chat_container.add_child(Text(f"\x1b[33m{warning_message}\x1b[39m", 1, 0))
        self.ui.request_render()

    def show_status(self, message: str) -> None:
        self.status_container.clear()
        self.status_container.add_child(Text(f"\x1b[2m{message}\x1b[22m", 1, 0))
        self.ui.request_render()

    def show_new_version_notification(self, release: Any) -> None:
        version = getattr(release, "version", None) or getattr(release, "tag_name", str(release))
        self.show_status(f"New version available: {version}")

    def show_package_update_notification(self, packages: list[str]) -> None:
        self.show_status(f"Package updates: {', '.join(packages)}")

    def dispose_active_selector(self) -> None:
        if self.active_selector_dispose:
            self.active_selector_dispose()
            self.active_selector_dispose = None
        while getattr(self.ui, "has_overlay_entries", False):
            self.ui.hide_overlay()

    def show_selector(self, component: Any, on_cancel: Callable[[], None] | None = None) -> None:
        self.dispose_active_selector()
        if hasattr(self.ui, "show_overlay"):
            handle = self.ui.show_overlay(component)
            self.active_selector_dispose = lambda: handle.hide() if hasattr(handle, "hide") else None
        else:
            self.chat_container.add_child(component)
            self.active_selector_dispose = lambda: self.chat_container.remove_child(component)
        self.ui.request_render()

    def show_settings_selector(self) -> None:
        settings = self.settings_manager
        config = {
            "theme": settings.get_theme() if hasattr(settings, "get_theme") else "dark",
            "tuiMode": settings.get_tui_mode(),
            "showHardwareCursor": settings.get_show_hardware_cursor(),
            "hideThinkingBlock": settings.get_hide_thinking_block(),
            "steeringMode": settings.get_steering_mode(),
            "followUpMode": settings.get_follow_up_mode(),
            "thinkingLevel": getattr(self.session, "thinking_level", "off"),
            "autoCompact": self.session.auto_compaction_enabled,
            "doubleEscapeAction": settings.get_double_escape_action(),
            "treeFilterMode": settings.get_tree_filter_mode(),
        }
        self.show_selector(SettingsSelectorComponent(config=config, on_cancel=self.dispose_active_selector))

    def show_thinking_selector(self) -> None:
        levels = self.session.get_available_thinking_levels() or list(THINKING_LEVEL_OPTIONS)

        def on_select(level: Any) -> None:
            self.select_thinking_level(level, persist=True)
            self.dispose_active_selector()

        self.show_selector(ThinkingSelectorComponent(levels=levels, on_select=on_select, on_cancel=self.dispose_active_selector))

    def show_trust_selector(self) -> None:
        def on_select(decision: Any) -> None:
            store = ProjectTrustStore()
            store.set(self.session.cwd, decision)
            self.show_status(f"Trust: {decision}")
            self.dispose_active_selector()

        self.show_selector(TrustSelectorComponent(on_select=on_select, on_cancel=self.dispose_active_selector))

    def show_model_selector(self, initial_search_input: str | None = None) -> None:
        self.show_selector(
            ModelSelectorComponent(
                query=initial_search_input or "",
                on_select=lambda model: (self.session.set_model(model) if hasattr(self.session, "set_model") else None, self.dispose_active_selector()),
                on_cancel=self.dispose_active_selector,
            )
        )

    def show_models_selector(self) -> None:
        self.show_selector(ScopedModelsSelectorComponent(on_cancel=self.dispose_active_selector))

    def show_user_message_selector(self) -> None:
        self.show_selector(UserMessageSelectorComponent(on_cancel=self.dispose_active_selector))

    def show_tree_selector(self, initial_selected_id: str | None = None) -> None:
        tree = self.session_manager.get_tree() if hasattr(self.session_manager, "get_tree") else []
        leaf_id = None
        if hasattr(self.session_manager, "get_leaf_id"):
            leaf_id = self.session_manager.get_leaf_id()

        def on_select(entry_id: str) -> None:
            if hasattr(self.session, "navigate_tree"):
                asyncio.create_task(self._switch_tree_branch(entry_id))
            elif hasattr(self.session_manager, "set_leaf_id"):
                self.session_manager.set_leaf_id(entry_id)
            self.dispose_active_selector()

        self.show_selector(
            TreeSelectorComponent(
                tree=tree,
                current_leaf_id=leaf_id,
                initial_selected_id=initial_selected_id,
                initial_filter_mode=self.settings_manager.get_tree_filter_mode(),
                on_select=on_select,
                on_cancel=self.dispose_active_selector,
                on_copy=lambda text: copy_to_clipboard(text or ""),
            )
        )

    def show_session_selector(self) -> None:
        sessions = self.session_manager.list_sessions() if hasattr(self.session_manager, "list_sessions") else []

        def on_select(session: Any) -> None:
            path = getattr(session, "file_path", None) or getattr(session, "path", None)
            if path and hasattr(self.session, "switch_session"):
                asyncio.create_task(self.session.switch_session(path))
            self.dispose_active_selector()

        self.show_selector(
            SessionSelectorComponent(
                sessions=sessions,
                on_select=on_select,
                on_cancel=self.dispose_active_selector,
                cwd=self.session.cwd,
            )
        )

    def show_login_provider_selector(self, initial_search_input: str | None = None) -> None:
        self.show_selector(
            OAuthSelectorComponent(query=initial_search_input or "", on_cancel=self.dispose_active_selector)
        )

    def show_login_auth_type_selector(self, provider_options: list[Any] | None = None) -> None:
        self.show_selector(LoginDialogComponent(on_cancel=self.dispose_active_selector))

    def subscribe_to_agent(self) -> None:
        def on_event(event: Any) -> None:
            event_type = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else "")
            if event_type in ("agent_start", "turn_start"):
                self.set_working_visible(True)
            elif event_type in ("agent_end", "turn_end"):
                self.set_working_visible(False)
            elif event_type == "message_end":
                message = getattr(event, "message", None) or (event.get("message") if isinstance(event, dict) else None)
                if message:
                    self.add_message_to_chat(message)
            elif event_type == "text_delta":
                text = getattr(event, "text", None) or (event.get("text") if isinstance(event, dict) else "")
                if text and self.streaming_component is None:
                    self.streaming_component = AssistantMessageComponent(message={"role": "assistant", "content": [{"type": "text", "text": text}]})
                    self.chat_container.add_child(self.streaming_component)
            self.ui.request_render()

        if hasattr(self.session, "subscribe"):
            self.unsubscribe = self.session.subscribe(on_event)

    def set_working_visible(self, visible: bool) -> None:
        self.working_visible = visible
        if visible:
            self.show_status_indicator(WorkingStatusIndicator(self.ui, self.working_message or self.default_working_message))
        else:
            self.clear_status_indicator("working")

    def show_status_indicator(self, indicator: StatusIndicator) -> None:
        self.clear_status_indicator()
        self.active_status_indicator = indicator
        self.status_container.clear()
        self.status_container.add_child(indicator)
        self.ui.request_render()

    def clear_status_indicator(self, kind: str | None = None) -> None:
        current = self.active_status_indicator
        if current is None:
            return
        if kind and getattr(current, "kind", None) != kind:
            return
        if hasattr(current, "dispose"):
            current.dispose()
        self.active_status_indicator = None
        self.status_container.clear()
        self.status_container.add_child(self.idle_status)
        self.ui.request_render()

    async def handle_right_click_paste(self) -> None:
        target = self.renderer.get_focused_component() if hasattr(self.renderer, "get_focused_component") else None
        if target is None or not hasattr(target, "handle_input"):
            return
        try:
            text = read_clipboard_text()
            if text:
                target.handle_input(f"\x1b[200~{text}\x1b[201~")
                self.ui.request_render()
        except Exception:
            return

    async def handle_clipboard_paste(self) -> None:
        try:
            image = await read_clipboard_image()
            if image:
                ext = extension_for_image_mime_type(image.mime_type) or "png"
                path = os.path.join(os.environ.get("TMPDIR") or "/tmp", f"pi-clipboard.{ext}")
                Path(path).write_bytes(image.data)
                if hasattr(self.editor, "insert_text_at_cursor"):
                    self.editor.insert_text_at_cursor(path)
                return
            text = read_clipboard_text()
            if text and hasattr(self.editor, "insert_text_at_cursor"):
                self.editor.insert_text_at_cursor(text)
        except Exception:
            return

    def register_signal_handlers(self) -> None:
        def handle_term(_signum: int, _frame: Any) -> None:
            self.stop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                previous = signal.getsignal(sig)
                signal.signal(sig, handle_term)
                self.signal_cleanup_handlers.append(lambda s=sig, p=previous: signal.signal(s, p))
            except Exception:
                pass

    def unregister_signal_handlers(self) -> None:
        for cleanup in self.signal_cleanup_handlers:
            try:
                cleanup()
            except Exception:
                pass
        self.signal_cleanup_handlers.clear()

    def stop(self, fullscreen_exit_output: str | None = None) -> None:
        self.shutdown_requested = True
        self._input_ready.set()
        if self.unsubscribe:
            self.unsubscribe()
            self.unsubscribe = None
        output = fullscreen_exit_output or self.settings_manager.get_fullscreen_exit_output()
        if getattr(self.renderer, "mode", None) == "fullscreen" and output == "transcript":
            self.switch_tui_mode("regular", restore_progress=False, start_renderer=False)
        if hasattr(self.ui, "stop"):
            self.ui.stop({"preserve_screen": getattr(self.renderer, "mode", None) == "fullscreen"})
        self.unregister_signal_handlers()


async def run_interactive_mode(session: AgentSession | AgentSessionRuntime, initial_messages: list[str] | None = None) -> None:
    """Public entry used by the CLI. Falls back to readline when not a TTY."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        from .mode import _run_readline_fallback

        raw = session.session if isinstance(session, AgentSessionRuntime) else session
        await _run_readline_fallback(raw, initial_messages)
        return
    try:
        mode = InteractiveMode(session, InteractiveModeOptions(initial_messages=initial_messages))
        await mode.run()
    except Exception:
        from .mode import _run_readline_fallback

        raw = session.session if isinstance(session, AgentSessionRuntime) else session
        await _run_readline_fallback(raw, initial_messages)
