"""
Core type definitions — mirrors packages/ai/src/types.ts
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Callable, Literal, Union

from pydantic import BaseModel, Field

# ─── Provider / API identifiers ──────────────────────────────────────────────

KnownApi = Literal[
    "openai-completions",
    "mistral-conversations",
    "openai-responses",
    "azure-openai-responses",
    "openai-codex-responses",
    "anthropic-messages",
    "bedrock-converse-stream",
    "google-generative-ai",
    "google-gemini-cli",
    "google-vertex",
    "pi-messages",
]
Api = str  # KnownApi or arbitrary string

KnownProvider = Literal[
    "amazon-bedrock",
    "ant-ling",
    "anthropic",
    "google",
    "google-gemini-cli",
    "google-antigravity",
    "google-vertex",
    "openai",
    "azure-openai-responses",
    "openai-codex",
    "radius",
    "nvidia",
    "deepseek",
    "github-copilot",
    "xai",
    "groq",
    "cerebras",
    "openrouter",
    "vercel-ai-gateway",
    "zai",
    "zai-coding-cn",
    "mistral",
    "minimax",
    "minimax-cn",
    "moonshotai",
    "moonshotai-cn",
    "huggingface",
    "fireworks",
    "together",
    "baseten",
    "opencode",
    "opencode-go",
    "kimi-coding",
    "cloudflare-workers-ai",
    "cloudflare-ai-gateway",
    "qwen-token-plan",
    "qwen-token-plan-cn",
    "qwen-token-plan-individual",
    "xiaomi",
    "xiaomi-token-plan-cn",
    "xiaomi-token-plan-ams",
    "xiaomi-token-plan-sgp",
]
Provider = str  # KnownProvider or arbitrary string
ProviderId = Provider

ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
ModelThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]
ToolChoice = Literal["auto", "none"]
CacheRetention = Literal["none", "short", "long"]
Transport = Literal["sse", "websocket", "websocket-cached", "auto"]
StopReason = Literal["pending", "stop", "length", "toolUse", "error", "aborted", "deferred"]
SessionAffinityFormat = Literal["openai", "openai-nosession", "openrouter"]
ThinkingTokenBudgetField = Literal["thinking_token_budget", "thinking_budget", "thinking_budget_tokens"]


# ─── Compat types ─────────────────────────────────────────────────────────────

@dataclass
class OpenAICompletionsCompat:
    """Compatibility settings for OpenAI Completions API."""
    supports_store: bool = False
    supports_developer_role: bool = False
    supports_reasoning_effort: bool = False
    reasoning_effort_map: dict[str, str] | None = None
    supports_usage_in_streaming: bool = True
    supports_finish_reason: bool = True
    max_tokens_field: Literal["max_completion_tokens", "max_tokens"] | None = None
    requires_tool_result_name: bool = False
    requires_assistant_after_tool_result: bool = False
    requires_thinking_as_text: bool = False
    requires_reasoning_content_on_assistant_messages: bool = False
    thinking_format: Literal[
        "openai",
        "openrouter",
        "deepseek",
        "together",
        "baseten",
        "zai",
        "qwen",
        "chat-template",
        "qwen-chat-template",
        "string-thinking",
        "ant-ling",
    ] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    chat_template_args: dict[str, Any] | None = None
    open_router_routing: "OpenRouterRouting | None" = None
    vercel_gateway_routing: "VercelGatewayRouting | None" = None
    zai_tool_stream: bool = False
    thinking_token_budget_field: ThinkingTokenBudgetField | None = None
    supports_thinking_token_budget: bool = False
    supports_openai_grammar_tools: bool = False
    supports_strict_mode: bool = True
    cache_control_format: Literal["anthropic"] | None = None
    send_session_affinity_headers: bool = False
    deferred_tools_mode: Literal["kimi"] | None = None
    session_affinity_format: SessionAffinityFormat | None = None
    supports_long_cache_retention: bool = True


@dataclass
class OpenAIResponsesCompat:
    """Compatibility settings for OpenAI Responses API."""
    supports_developer_role: bool = True
    session_affinity_format: SessionAffinityFormat | None = None
    supports_long_cache_retention: bool = True
    supports_strict_mode: bool | None = None
    supports_openai_grammar_tools: bool = False
    supports_additional_tools: bool = False
    supports_tool_search: bool = False
    supports_explicit_prompt_cache_mode: bool = False


@dataclass
class AnthropicMessagesCompat:
    """Compatibility settings for Anthropic Messages-compatible APIs."""
    supports_eager_tool_input_streaming: bool = True
    supports_long_cache_retention: bool = True
    send_session_affinity_headers: bool = False
    supports_cache_control_on_tools: bool = True
    supports_temperature: bool = True
    force_adaptive_thinking: bool = False
    allow_empty_signature: bool = False
    supports_strict_tools: bool = False
    allowed_fallback_models: list[dict[str, Any]] | None = None
    supports_tool_references: bool | None = None


@dataclass
class BedrockCompat:
    """Compatibility settings for Amazon Bedrock models."""
    supports_strict_mode: bool = False


@dataclass
class OpenRouterRouting:
    """Routing configuration for OpenRouter."""
    allow_fallbacks: bool | None = None
    require_parameters: bool | None = None
    data_collection: Literal["deny", "allow"] | None = None
    zdr: bool | None = None
    enforce_distillable_text: bool | None = None
    only: list[str] | None = None
    order: list[str] | None = None
    ignore: list[str] | None = None
    quantizations: list[str] | None = None
    sort: Any | None = None
    max_price: dict[str, Any] | None = None
    preferred_min_throughput: Any | None = None
    preferred_max_latency: Any | None = None


@dataclass
class VercelGatewayRouting:
    """Routing configuration for Vercel Gateway."""
    only: list[str] | None = None
    order: list[str] | None = None


# ─── Thinking budgets ─────────────────────────────────────────────────────────

class ThinkingBudgets(BaseModel):
    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


# ─── Stream options ───────────────────────────────────────────────────────────

class StreamOptions(BaseModel):
    temperature: float | None = None
    max_tokens: int | None = None
    signal: Any | None = None
    api_key: str | None = None
    fetch: Any | None = None
    env: dict[str, str] | None = None
    transport: Transport | None = None
    cache_retention: CacheRetention | None = "short"
    session_id: str | None = None
    websocket_connect_timeout_ms: int | None = None
    on_payload: (
        Callable[[Any, "Model"], Any | None] |
        Callable[[Any, "Model"], Awaitable[Any | None]] |
        None
    ) = None
    on_response: Callable[[Any, "Model"], Any | None] | None = None
    headers: dict[str, str | None] | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = 60000
    metadata: dict[str, Any] | None = None
    sampling_params: dict[str, Any] | None = None

    model_config = {"arbitrary_types_allowed": True}

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style .get() for backwards compatibility with provider code.

        Provider implementations use opts.get("field") to safely read options
        without raising AttributeError. This mirrors TypeScript's optional
        chaining: ``options?.field``.
        """
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Dict-style [] access for backwards compatibility.

        Used in provider code like: ``opts["on_payload"](params, model)``.
        """
        value = getattr(self, key, None)
        if value is None:
            raise KeyError(key)
        return value


class SimpleStreamOptions(StreamOptions):
    """Unified options with reasoning — passed to stream_simple() / complete_simple()."""
    tool_choice: ToolChoice | None = None
    reasoning: ThinkingLevel | None = None
    deferred: bool | dict[str, Any] | None = None
    thinking_budgets: ThinkingBudgets | None = None


# ─── Content blocks ───────────────────────────────────────────────────────────

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str
    text_signature: str | None = None


class TextSignatureV1(BaseModel):
    """Structured text signature for OpenAI Responses API (v1 format)."""
    v: Literal[1] = 1
    id: str
    phase: Literal["commentary", "final_answer"] | None = None


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: str | None = None
    redacted: bool | None = None  # True for Anthropic redacted_thinking blocks


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    data: str  # base64 encoded
    mime_type: str  # e.g. "image/jpeg"


class ToolCall(BaseModel):
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    thought_signature: str | None = None  # Google-specific
    namespace: str | None = None  # OpenAI Responses namespaced tools


# ─── Usage / cost ─────────────────────────────────────────────────────────────

class UsageCost(BaseModel):
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


class Usage(BaseModel):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int | None = None  # subset of cache_write with 1h retention (Anthropic)
    reasoning: int | None = None  # subset of output when the provider reports it
    total_tokens: int = 0
    cost: UsageCost = Field(default_factory=UsageCost)


# ─── Messages ─────────────────────────────────────────────────────────────────

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[TextContent | ImageContent]
    timestamp: int  # Unix ms


class DeferredHandle(BaseModel):
    provider: str
    model_id: str
    api: str
    id: str
    expires_at: int | None = None
    poll_after_ms: int | None = None
    data: Any | None = None


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[TextContent | ThinkingContent | ToolCall]
    api: Api
    provider: Provider
    model: str
    response_model: str | None = None
    response_id: str | None = None
    diagnostics: list[dict[str, Any]] | None = None
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    deferred: DeferredHandle | None = None
    error_message: str | None = None
    raw_stop_reason: str | None = None
    end_turn: bool | None = None
    timestamp: int  # Unix ms


class ToolResultMessage(BaseModel):
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[TextContent | ImageContent]
    details: Any | None = None
    usage: Usage | None = None
    added_tool_names: list[str] | None = None
    is_error: bool = False
    timestamp: int  # Unix ms


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


# ─── Tool ─────────────────────────────────────────────────────────────────────

class ConstrainedSamplingConfig(BaseModel):
    type: Literal["json_schema", "grammar"]
    strict: Literal["prefer", "require"] | None = None
    variants: dict[str, str] | None = None


class Tool(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object
    constrained_sampling: ConstrainedSamplingConfig | Literal[False] | None = None


# ─── Context ──────────────────────────────────────────────────────────────────

class Context(BaseModel):
    system_prompt: str | None = None
    messages: list[Message] = Field(default_factory=list)
    tools: list[Tool] | None = None


# ─── Model ────────────────────────────────────────────────────────────────────

class ModelCostTier(BaseModel):
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    input_tokens_above: int = 0


class ModelCost(BaseModel):
    input: float = 0.0   # $/million tokens
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    tiers: list[ModelCostTier] | None = None


class Model(BaseModel):
    id: str
    name: str
    api: Api
    provider: Provider
    base_url: str
    reasoning: bool = False
    thinking_level_map: dict[str, str | None] | None = None
    input: list[Literal["text", "image"]] = Field(default_factory=lambda: ["text"])
    cost: ModelCost = Field(default_factory=ModelCost)
    context_window: int = 128000
    max_tokens: int = 8192
    sampling_params: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
    compat: (
        OpenAICompletionsCompat |
        OpenAIResponsesCompat |
        AnthropicMessagesCompat |
        BedrockCompat |
        OpenRouterRouting |
        VercelGatewayRouting |
        dict[str, Any] |
        None
    ) = None


# ─── Streaming events ─────────────────────────────────────────────────────────

class EventStart(BaseModel):
    type: Literal["start"] = "start"
    partial: AssistantMessage


class EventTextStart(BaseModel):
    type: Literal["text_start"] = "text_start"
    content_index: int
    partial: AssistantMessage


class EventTextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class EventTextEnd(BaseModel):
    type: Literal["text_end"] = "text_end"
    content_index: int
    content: str
    partial: AssistantMessage


class EventThinkingStart(BaseModel):
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int
    partial: AssistantMessage


class EventThinkingDelta(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class EventThinkingEnd(BaseModel):
    type: Literal["thinking_end"] = "thinking_end"
    content_index: int
    content: str
    partial: AssistantMessage


class EventToolCallStart(BaseModel):
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int
    partial: AssistantMessage


class EventToolCallDelta(BaseModel):
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class EventToolCallEnd(BaseModel):
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage


class EventDone(BaseModel):
    type: Literal["done"] = "done"
    reason: Literal["stop", "length", "toolUse", "deferred"]
    message: AssistantMessage


class EventError(BaseModel):
    type: Literal["error"] = "error"
    reason: Literal["aborted", "error"]
    error: AssistantMessage


AssistantMessageEvent = Union[
    EventStart,
    EventTextStart,
    EventTextDelta,
    EventTextEnd,
    EventThinkingStart,
    EventThinkingDelta,
    EventThinkingEnd,
    EventToolCallStart,
    EventToolCallDelta,
    EventToolCallEnd,
    EventDone,
    EventError,
]

# Async generator of AssistantMessageEvent
AssistantMessageEventStream = AsyncGenerator[AssistantMessageEvent, None]

# StreamFunction type alias
StreamFunction = Callable[
    ["Model", "Context", "SimpleStreamOptions | None"],
    "AssistantMessageEventStream",
]
