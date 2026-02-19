"""
Extension system types.

Extensions can:
- Subscribe to agent lifecycle events
- Register LLM-callable tools
- Register commands and CLI flags

Mirrors core/extensions/types.ts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Protocol


# ============================================================================
# UI Context
# ============================================================================


class ExtensionUIContext(Protocol):
    """UI methods for user interaction in extensions."""

    async def select(self, title: str, options: list[str], opts: dict | None = None) -> str | None: ...
    async def confirm(self, title: str, message: str, opts: dict | None = None) -> bool: ...
    async def input(self, title: str, placeholder: str | None = None, opts: dict | None = None) -> str | None: ...
    def notify(self, message: str, type: str = "info") -> None: ...
    def set_status(self, key: str, text: str | None) -> None: ...
    def set_working_message(self, message: str | None = None) -> None: ...
    @property
    def theme(self) -> Any: ...


# ============================================================================
# Context Usage
# ============================================================================


@dataclass
class ContextUsage:
    tokens: int | None
    context_window: int
    percent: float | None


# ============================================================================
# Extension Context
# ============================================================================


@dataclass
class ExtensionContext:
    """Context passed to extension event handlers."""

    ui: ExtensionUIContext
    has_ui: bool
    cwd: str
    session_manager: Any
    model_registry: Any
    is_idle: Callable[[], bool] = field(default=lambda: True)
    abort: Callable[[], None] = field(default=lambda: None)
    has_pending_messages: Callable[[], bool] = field(default=lambda: False)
    shutdown: Callable[[], None] = field(default=lambda: None)
    get_context_usage: Callable[[], ContextUsage | None] = field(default=lambda: None)
    compact: Callable[..., None] = field(default=lambda **kw: None)
    get_system_prompt: Callable[[], str] = field(default=lambda: "")
    model: Any = None


# ============================================================================
# Tool Definition
# ============================================================================


@dataclass
class ToolDefinition:
    """Tool definition for register_tool()."""

    name: str
    label: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[..., Any] = field(default=lambda *a, **kw: None)
    render_call: Callable | None = None
    render_result: Callable | None = None


@dataclass
class ToolRenderResultOptions:
    expanded: bool
    is_partial: bool


@dataclass
class ToolInfo:
    name: str
    description: str
    parameters: dict[str, Any]


# ============================================================================
# Command Registration
# ============================================================================


@dataclass
class RegisteredCommand:
    name: str
    description: str | None = None
    get_argument_completions: Callable[[str], list | None] | None = None
    handler: Callable[..., Any] = field(default=lambda args, ctx: None)


@dataclass
class ExtensionFlag:
    name: str
    description: str | None
    type: str  # "boolean" | "string"
    default: bool | str | None
    extension_path: str


@dataclass
class ExtensionShortcut:
    shortcut: str
    description: str | None
    handler: Callable[..., Any]
    extension_path: str


# ============================================================================
# Events
# ============================================================================


@dataclass
class SessionStartEvent:
    type: str = "session_start"


@dataclass
class SessionBeforeSwitchEvent:
    reason: str
    target_session_file: str | None = None
    type: str = "session_before_switch"


@dataclass
class SessionSwitchEvent:
    reason: str
    previous_session_file: str | None = None
    type: str = "session_switch"


@dataclass
class SessionBeforeForkEvent:
    entry_id: str
    type: str = "session_before_fork"


@dataclass
class SessionForkEvent:
    previous_session_file: str | None = None
    type: str = "session_fork"


@dataclass
class SessionBeforeCompactEvent:
    preparation: Any
    branch_entries: list[Any] = field(default_factory=list)
    custom_instructions: str | None = None
    type: str = "session_before_compact"


@dataclass
class SessionCompactEvent:
    compaction_entry: Any
    from_extension: bool = False
    type: str = "session_compact"


@dataclass
class SessionShutdownEvent:
    type: str = "session_shutdown"


@dataclass
class SessionBeforeTreeEvent:
    preparation: Any
    type: str = "session_before_tree"


@dataclass
class SessionTreeEvent:
    new_leaf_id: str | None
    old_leaf_id: str | None
    summary_entry: Any = None
    from_extension: bool = False
    type: str = "session_tree"


@dataclass
class ContextEvent:
    messages: list[Any] = field(default_factory=list)
    type: str = "context"


@dataclass
class BeforeAgentStartEvent:
    prompt: str
    system_prompt: str
    images: list[Any] = field(default_factory=list)
    type: str = "before_agent_start"


@dataclass
class AgentStartEvent:
    type: str = "agent_start"


@dataclass
class AgentEndEvent:
    messages: list[Any] = field(default_factory=list)
    type: str = "agent_end"


@dataclass
class TurnStartEvent:
    turn_index: int
    timestamp: int
    type: str = "turn_start"


@dataclass
class TurnEndEvent:
    turn_index: int
    message: Any
    tool_results: list[Any] = field(default_factory=list)
    type: str = "turn_end"


@dataclass
class MessageStartEvent:
    message: Any
    type: str = "message_start"


@dataclass
class MessageUpdateEvent:
    message: Any
    assistant_message_event: Any
    type: str = "message_update"


@dataclass
class MessageEndEvent:
    message: Any
    type: str = "message_end"


@dataclass
class ToolExecutionStartEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    type: str = "tool_execution_start"


@dataclass
class ToolExecutionUpdateEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any
    type: str = "tool_execution_update"


@dataclass
class ToolExecutionEndEvent:
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool
    type: str = "tool_execution_end"


@dataclass
class ModelSelectEvent:
    model: Any
    previous_model: Any | None
    source: str  # "set" | "cycle" | "restore"
    type: str = "model_select"


@dataclass
class UserBashEvent:
    command: str
    exclude_from_context: bool
    cwd: str
    type: str = "user_bash"


@dataclass
class InputEvent:
    text: str
    source: str  # "interactive" | "rpc" | "extension"
    images: list[Any] = field(default_factory=list)
    type: str = "input"


@dataclass
class ResourcesDiscoverEvent:
    cwd: str
    reason: str  # "startup" | "reload"
    type: str = "resources_discover"


@dataclass
class ToolCallEvent:
    tool_call_id: str
    tool_name: str
    input: dict[str, Any]
    type: str = "tool_call"


@dataclass
class ToolResultEvent:
    tool_call_id: str
    tool_name: str
    input: dict[str, Any]
    content: list[Any]
    is_error: bool
    details: Any = None
    type: str = "tool_result"


# ============================================================================
# Event Results
# ============================================================================


@dataclass
class ContextEventResult:
    messages: list[Any] | None = None


@dataclass
class ToolCallEventResult:
    block: bool = False
    reason: str | None = None


@dataclass
class ToolResultEventResult:
    content: list[Any] | None = None
    details: Any = None
    is_error: bool | None = None


@dataclass
class BeforeAgentStartEventResult:
    message: Any = None
    system_prompt: str | None = None


@dataclass
class SessionBeforeSwitchResult:
    cancel: bool = False


@dataclass
class SessionBeforeForkResult:
    cancel: bool = False
    skip_conversation_restore: bool = False


@dataclass
class SessionBeforeCompactResult:
    cancel: bool = False
    compaction: Any = None


@dataclass
class SessionBeforeTreeResult:
    cancel: bool = False
    summary: dict[str, Any] | None = None
    custom_instructions: str | None = None
    replace_instructions: bool = False
    label: str | None = None


@dataclass
class ResourcesDiscoverResult:
    skill_paths: list[str] = field(default_factory=list)
    prompt_paths: list[str] = field(default_factory=list)
    theme_paths: list[str] = field(default_factory=list)


@dataclass
class UserBashEventResult:
    operations: Any | None = None
    result: Any | None = None


InputEventResult = dict[str, Any]


# ============================================================================
# Runtime
# ============================================================================


@dataclass
class ExtensionRuntimeState:
    flag_values: dict[str, bool | str] = field(default_factory=dict)
    pending_provider_registrations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExtensionActions:
    """Actions available to extensions — mirrors ExtensionActions in TypeScript."""

    send_message: Callable[..., None] = field(default=lambda *a, **kw: None)
    send_user_message: Callable[..., None] = field(default=lambda *a, **kw: None)
    append_entry: Callable[..., None] = field(default=lambda *a, **kw: None)
    set_session_name: Callable[[str], None] = field(default=lambda n: None)
    get_session_name: Callable[[], str | None] = field(default=lambda: None)
    set_label: Callable[..., None] = field(default=lambda *a: None)
    get_active_tools: Callable[[], list[str]] = field(default=list)
    get_all_tools: Callable[[], list[Any]] = field(default=list)
    set_active_tools: Callable[[list[str]], None] = field(default=lambda t: None)
    get_commands: Callable[[], list[Any]] = field(default=list)
    set_model: Callable[..., Any] = field(default=lambda m: None)
    get_thinking_level: Callable[[], str] = field(default=lambda: "low")
    set_thinking_level: Callable[[str], None] = field(default=lambda level: None)


@dataclass
class ExtensionRuntime(ExtensionRuntimeState):
    send_message: Callable[..., None] = field(default=lambda *a, **kw: None)
    send_user_message: Callable[..., None] = field(default=lambda *a, **kw: None)
    append_entry: Callable[..., None] = field(default=lambda *a, **kw: None)
    set_session_name: Callable[[str], None] = field(default=lambda n: None)
    get_session_name: Callable[[], str | None] = field(default=lambda: None)
    set_label: Callable[..., None] = field(default=lambda *a: None)
    get_active_tools: Callable[[], list[str]] = field(default=list)
    get_all_tools: Callable[[], list[ToolInfo]] = field(default=list)
    set_active_tools: Callable[[list[str]], None] = field(default=lambda t: None)
    get_commands: Callable[[], list[Any]] = field(default=list)
    set_model: Callable[..., Any] = field(default=lambda m: None)
    get_thinking_level: Callable[[], str] = field(default=lambda: "low")
    set_thinking_level: Callable[[str], None] = field(default=lambda l: None)


# ============================================================================
# Loaded Extension
# ============================================================================


@dataclass
class Extension:
    """Loaded extension with all registered handlers and tools."""

    path: str
    resolved_path: str
    handlers: dict[str, list[Callable]] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=dict)
    message_renderers: dict[str, Any] = field(default_factory=dict)
    commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    flags: dict[str, ExtensionFlag] = field(default_factory=dict)
    shortcuts: dict[str, ExtensionShortcut] = field(default_factory=dict)


@dataclass
class LoadExtensionsResult:
    extensions: list[Extension] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    runtime: ExtensionRuntime = field(default_factory=ExtensionRuntime)


@dataclass
class ExtensionError:
    extension_path: str
    event: str
    error: str
    stack: str | None = None


# ============================================================================
# Provider Config
# ============================================================================


@dataclass
class ProviderModelConfig:
    id: str
    name: str
    reasoning: bool
    input: list[str]
    cost: dict[str, float]
    context_window: int
    max_tokens: int
    api: str | None = None
    headers: dict[str, str] | None = None
    compat: dict[str, Any] | None = None


@dataclass
class ProviderConfig:
    base_url: str | None = None
    api_key: str | None = None
    api: str | None = None
    stream_simple: Callable | None = None
    headers: dict[str, str] | None = None
    auth_header: bool = False
    models: list[ProviderModelConfig] | None = None
    oauth: dict[str, Any] | None = None


ExtensionFactory = Callable[["ExtensionAPI"], None]


# ============================================================================
# Extension API (protocol)
# ============================================================================


class ExtensionAPI(Protocol):
    """API passed to extension factory functions."""

    def on(self, event: str, handler: Callable) -> None: ...
    def register_tool(self, tool: ToolDefinition) -> None: ...
    def register_command(self, name: str, options: dict) -> None: ...
    def register_shortcut(self, shortcut: str, options: dict) -> None: ...
    def register_flag(self, name: str, options: dict) -> None: ...
    def get_flag(self, name: str) -> bool | str | None: ...
    def register_message_renderer(self, custom_type: str, renderer: Callable) -> None: ...
    def send_message(self, message: Any, options: dict | None = None) -> None: ...
    def send_user_message(self, content: Any, options: dict | None = None) -> None: ...
    def append_entry(self, custom_type: str, data: Any = None) -> None: ...
    def set_session_name(self, name: str) -> None: ...
    def get_session_name(self) -> str | None: ...
    def set_label(self, entry_id: str, label: str | None) -> None: ...
    def register_provider(self, name: str, config: ProviderConfig) -> None: ...
    @property
    def events(self) -> Any: ...
