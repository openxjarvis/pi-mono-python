"""
RPC protocol types for headless operation.

Commands are sent as JSON lines on stdin.
Responses and events are emitted as JSON lines on stdout.

Mirrors packages/coding-agent/src/modes/rpc/rpc-types.ts
"""
from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel


# ============================================================================
# RPC Commands (stdin)
# ============================================================================

class RpcCommandPrompt(BaseModel):
    type: Literal["prompt"]
    id: str | None = None
    message: str
    images: list[dict[str, Any]] | None = None
    streamingBehavior: Literal["steer", "followUp"] | None = None


class RpcCommandSteer(BaseModel):
    type: Literal["steer"]
    id: str | None = None
    message: str
    images: list[dict[str, Any]] | None = None


class RpcCommandFollowUp(BaseModel):
    type: Literal["follow_up"]
    id: str | None = None
    message: str
    images: list[dict[str, Any]] | None = None


class RpcCommandAbort(BaseModel):
    type: Literal["abort"]
    id: str | None = None


class RpcCommandNewSession(BaseModel):
    type: Literal["new_session"]
    id: str | None = None
    parentSession: str | None = None


class RpcCommandGetState(BaseModel):
    type: Literal["get_state"]
    id: str | None = None


class RpcCommandSetModel(BaseModel):
    type: Literal["set_model"]
    id: str | None = None
    provider: str
    modelId: str


class RpcCommandCycleModel(BaseModel):
    type: Literal["cycle_model"]
    id: str | None = None


class RpcCommandGetAvailableModels(BaseModel):
    type: Literal["get_available_models"]
    id: str | None = None


class RpcCommandSetThinkingLevel(BaseModel):
    type: Literal["set_thinking_level"]
    id: str | None = None
    level: str


class RpcCommandCycleThinkingLevel(BaseModel):
    type: Literal["cycle_thinking_level"]
    id: str | None = None


class RpcCommandSetSteeringMode(BaseModel):
    type: Literal["set_steering_mode"]
    id: str | None = None
    mode: Literal["all", "one-at-a-time"]


class RpcCommandSetFollowUpMode(BaseModel):
    type: Literal["set_follow_up_mode"]
    id: str | None = None
    mode: Literal["all", "one-at-a-time"]


class RpcCommandCompact(BaseModel):
    type: Literal["compact"]
    id: str | None = None
    customInstructions: str | None = None


class RpcCommandSetAutoCompaction(BaseModel):
    type: Literal["set_auto_compaction"]
    id: str | None = None
    enabled: bool


class RpcCommandSetAutoRetry(BaseModel):
    type: Literal["set_auto_retry"]
    id: str | None = None
    enabled: bool


class RpcCommandAbortRetry(BaseModel):
    type: Literal["abort_retry"]
    id: str | None = None


class RpcCommandBash(BaseModel):
    type: Literal["bash"]
    id: str | None = None
    command: str


class RpcCommandAbortBash(BaseModel):
    type: Literal["abort_bash"]
    id: str | None = None


class RpcCommandGetSessionStats(BaseModel):
    type: Literal["get_session_stats"]
    id: str | None = None


class RpcCommandExportHtml(BaseModel):
    type: Literal["export_html"]
    id: str | None = None
    outputPath: str | None = None


class RpcCommandSwitchSession(BaseModel):
    type: Literal["switch_session"]
    id: str | None = None
    sessionPath: str


class RpcCommandFork(BaseModel):
    type: Literal["fork"]
    id: str | None = None
    entryId: str


class RpcCommandGetForkMessages(BaseModel):
    type: Literal["get_fork_messages"]
    id: str | None = None


class RpcCommandGetLastAssistantText(BaseModel):
    type: Literal["get_last_assistant_text"]
    id: str | None = None


class RpcCommandSetSessionName(BaseModel):
    type: Literal["set_session_name"]
    id: str | None = None
    name: str


class RpcCommandGetMessages(BaseModel):
    type: Literal["get_messages"]
    id: str | None = None


class RpcCommandGetCommands(BaseModel):
    type: Literal["get_commands"]
    id: str | None = None


RpcCommand = Union[
    RpcCommandPrompt,
    RpcCommandSteer,
    RpcCommandFollowUp,
    RpcCommandAbort,
    RpcCommandNewSession,
    RpcCommandGetState,
    RpcCommandSetModel,
    RpcCommandCycleModel,
    RpcCommandGetAvailableModels,
    RpcCommandSetThinkingLevel,
    RpcCommandCycleThinkingLevel,
    RpcCommandSetSteeringMode,
    RpcCommandSetFollowUpMode,
    RpcCommandCompact,
    RpcCommandSetAutoCompaction,
    RpcCommandSetAutoRetry,
    RpcCommandAbortRetry,
    RpcCommandBash,
    RpcCommandAbortBash,
    RpcCommandGetSessionStats,
    RpcCommandExportHtml,
    RpcCommandSwitchSession,
    RpcCommandFork,
    RpcCommandGetForkMessages,
    RpcCommandGetLastAssistantText,
    RpcCommandSetSessionName,
    RpcCommandGetMessages,
    RpcCommandGetCommands,
]


# ============================================================================
# RPC Slash Command
# ============================================================================

class RpcSlashCommand(BaseModel):
    name: str
    description: str | None = None
    source: Literal["extension", "prompt", "skill"]
    location: Literal["user", "project", "path"] | None = None
    path: str | None = None


# ============================================================================
# RPC Session State
# ============================================================================

class RpcSessionState(BaseModel):
    model: dict[str, Any] | None = None
    thinkingLevel: str
    isStreaming: bool
    isCompacting: bool
    steeringMode: Literal["all", "one-at-a-time"]
    followUpMode: Literal["all", "one-at-a-time"]
    sessionFile: str | None = None
    sessionId: str
    sessionName: str | None = None
    autoCompactionEnabled: bool
    messageCount: int
    pendingMessageCount: int


# ============================================================================
# RPC Responses (stdout)
# ============================================================================

class RpcResponseBase(BaseModel):
    type: Literal["response"] = "response"
    id: str | None = None
    command: str
    success: bool


class RpcResponseSuccess(RpcResponseBase):
    success: Literal[True] = True
    data: Any = None


class RpcResponseError(RpcResponseBase):
    success: Literal[False] = False
    error: str


RpcResponse = Union[RpcResponseSuccess, RpcResponseError]


# ============================================================================
# Extension UI Events (stdout)
# ============================================================================

class RpcExtensionUIRequestSelect(BaseModel):
    type: Literal["extension_ui_request"] = "extension_ui_request"
    id: str
    method: Literal["select"]
    title: str
    options: list[str]
    timeout: int | None = None


class RpcExtensionUIRequestConfirm(BaseModel):
    type: Literal["extension_ui_request"] = "extension_ui_request"
    id: str
    method: Literal["confirm"]
    title: str
    message: str
    timeout: int | None = None


class RpcExtensionUIRequestInput(BaseModel):
    type: Literal["extension_ui_request"] = "extension_ui_request"
    id: str
    method: Literal["input"]
    title: str
    placeholder: str | None = None
    timeout: int | None = None


class RpcExtensionUIRequestEditor(BaseModel):
    type: Literal["extension_ui_request"] = "extension_ui_request"
    id: str
    method: Literal["editor"]
    title: str
    prefill: str | None = None


class RpcExtensionUIRequestNotify(BaseModel):
    type: Literal["extension_ui_request"] = "extension_ui_request"
    id: str
    method: Literal["notify"]
    message: str
    notifyType: Literal["info", "warning", "error"] | None = None


class RpcExtensionUIRequestSetStatus(BaseModel):
    type: Literal["extension_ui_request"] = "extension_ui_request"
    id: str
    method: Literal["setStatus"]
    statusKey: str
    statusText: str | None = None


class RpcExtensionUIRequestSetWidget(BaseModel):
    type: Literal["extension_ui_request"] = "extension_ui_request"
    id: str
    method: Literal["setWidget"]
    widgetKey: str
    widgetLines: list[str] | None = None
    widgetPlacement: Literal["aboveEditor", "belowEditor"] | None = None


class RpcExtensionUIRequestSetTitle(BaseModel):
    type: Literal["extension_ui_request"] = "extension_ui_request"
    id: str
    method: Literal["setTitle"]
    title: str


class RpcExtensionUIRequestSetEditorText(BaseModel):
    type: Literal["extension_ui_request"] = "extension_ui_request"
    id: str
    method: Literal["set_editor_text"]
    text: str


RpcExtensionUIRequest = Union[
    RpcExtensionUIRequestSelect,
    RpcExtensionUIRequestConfirm,
    RpcExtensionUIRequestInput,
    RpcExtensionUIRequestEditor,
    RpcExtensionUIRequestNotify,
    RpcExtensionUIRequestSetStatus,
    RpcExtensionUIRequestSetWidget,
    RpcExtensionUIRequestSetTitle,
    RpcExtensionUIRequestSetEditorText,
]


# ============================================================================
# Extension UI Commands (stdin)
# ============================================================================

class RpcExtensionUIResponseValue(BaseModel):
    type: Literal["extension_ui_response"] = "extension_ui_response"
    id: str
    value: str


class RpcExtensionUIResponseConfirmed(BaseModel):
    type: Literal["extension_ui_response"] = "extension_ui_response"
    id: str
    confirmed: bool


class RpcExtensionUIResponseCancelled(BaseModel):
    type: Literal["extension_ui_response"] = "extension_ui_response"
    id: str
    cancelled: Literal[True]


RpcExtensionUIResponse = Union[
    RpcExtensionUIResponseValue,
    RpcExtensionUIResponseConfirmed,
    RpcExtensionUIResponseCancelled,
]
