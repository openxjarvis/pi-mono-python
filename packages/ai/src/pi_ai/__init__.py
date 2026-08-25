"""
pi_ai — Unified LLM API
Python mirror of @mariozechner/pi-ai
"""

# Core types
from .types import (
    Api,
    AssistantMessage,
    AssistantMessageEvent,
    AssistantMessageEventStream,
    CacheRetention,
    Context,
    EventDone,
    EventError,
    EventStart,
    EventTextDelta,
    EventTextEnd,
    EventTextStart,
    EventThinkingDelta,
    EventThinkingEnd,
    EventThinkingStart,
    EventToolCallDelta,
    EventToolCallEnd,
    EventToolCallStart,
    ImageContent,
    KnownApi,
    KnownProvider,
    Message,
    Model,
    ModelCost,
    AnthropicMessagesCompat,
    BedrockCompat,
    DeferredHandle,
    OpenAICompletionsCompat,
    OpenAIResponsesCompat,
    OpenRouterRouting,
    Provider,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingBudgets,
    ThinkingContent,
    ThinkingLevel,
    Tool,
    ToolCall,
    ToolResultMessage,
    Transport,
    Usage,
    UsageCost,
    UserMessage,
    VercelGatewayRouting,
)

# Model registry
from .models import (
    calculate_cost,
    clamp_thinking_level,
    get_model,
    get_models,
    get_providers,
    get_supported_thinking_levels,
    models_are_equal,
    supports_xhigh,
)

# API registry
from .api_registry import get_api_provider, get_api_providers, register_api_provider, unregister_api_providers, clear_api_providers

# Environment API keys
from .env_api_keys import find_env_keys, get_env_api_key

# Streaming functions
from .stream import complete, complete_simple, stream, stream_simple

# Utilities
from .utils.event_stream import AssistantMessageEventStream as AssistantMessageEventStreamClass, EventStream, create_assistant_message_event_stream
from .utils.json_parse import parse_partial_json, parse_streaming_json
from .utils.overflow import is_context_overflow, is_recoverable_length, get_overflow_patterns
from .utils.validation import validate_tool_arguments, validate_tool_call
from .utils.sanitize_unicode import sanitize_surrogates
from .utils.retry import RetryCallbacks, RetryPolicy, is_retryable_assistant_error, retry_assistant_call
from .utils.deferred_tools import split_deferred_tools
from .utils.estimate import estimate_context_tokens
from .utils.text import content_text
from .utils.uuid import uuidv7
from .auth import (
    InMemoryCredentialStore,
    ModelsError,
    default_provider_auth_context,
    env_api_key_auth,
    resolve_provider_auth,
)
from .models_runtime import Models, create_models, create_provider_from_id, has_api
from .images import GeneratedImage, ImageGenerationRequest, generate_images

__all__ = [
    # Types
    "Api", "KnownApi", "KnownProvider", "Provider",
    "ThinkingLevel", "ThinkingBudgets", "CacheRetention", "Transport", "StopReason",
    "StreamOptions", "SimpleStreamOptions",
    "TextContent", "ThinkingContent", "ImageContent", "ToolCall",
    "Usage", "UsageCost",
    "UserMessage", "AssistantMessage", "ToolResultMessage", "Message",
    "Tool", "Context", "Model", "ModelCost",
    "OpenAICompletionsCompat", "OpenAIResponsesCompat", "AnthropicMessagesCompat", "BedrockCompat",
    "OpenRouterRouting", "VercelGatewayRouting", "DeferredHandle",
    "AssistantMessageEvent", "AssistantMessageEventStream",
    "EventStart", "EventTextStart", "EventTextDelta", "EventTextEnd",
    "EventThinkingStart", "EventThinkingDelta", "EventThinkingEnd",
    "EventToolCallStart", "EventToolCallDelta", "EventToolCallEnd",
    "EventDone", "EventError",
    # Models
    "get_model", "get_providers", "get_models", "calculate_cost", "supports_xhigh", "models_are_equal",
    "get_supported_thinking_levels", "clamp_thinking_level",
    # Registry
    "register_api_provider", "get_api_provider", "get_api_providers", "unregister_api_providers", "clear_api_providers",
    # Keys
    "get_env_api_key", "find_env_keys",
    # Streaming
    "stream", "complete", "stream_simple", "complete_simple",
    # Utils
    "EventStream", "AssistantMessageEventStreamClass", "create_assistant_message_event_stream",
    "parse_partial_json", "parse_streaming_json",
    "validate_tool_arguments", "validate_tool_call",
    "is_context_overflow", "is_recoverable_length", "get_overflow_patterns",
    "sanitize_surrogates",
    "RetryCallbacks", "RetryPolicy", "is_retryable_assistant_error", "retry_assistant_call",
    "split_deferred_tools", "estimate_context_tokens", "content_text", "uuidv7",
    "InMemoryCredentialStore", "ModelsError", "default_provider_auth_context",
    "env_api_key_auth", "resolve_provider_auth",
    "Models", "create_models", "create_provider_from_id", "has_api",
    "GeneratedImage", "ImageGenerationRequest", "generate_images",
]
