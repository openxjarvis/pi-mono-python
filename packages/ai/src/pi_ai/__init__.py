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
    Provider,
    SimpleStreamOptions,
    StopReason,
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
)

# Model registry
from .models import calculate_cost, get_model, get_models, get_providers, supports_xhigh

# API registry
from .api_registry import get_api_provider, register_api_provider, unregister_api_providers

# Environment API keys
from .env_api_keys import get_env_api_key

# Streaming functions
from .stream import complete, complete_simple, stream, stream_simple

# Utilities
from .utils.event_stream import EventStream
from .utils.json_parse import parse_partial_json
from .utils.validation import validate_tool_arguments

__all__ = [
    # Types
    "Api", "KnownApi", "KnownProvider", "Provider",
    "ThinkingLevel", "ThinkingBudgets", "CacheRetention", "Transport", "StopReason",
    "StreamOptions", "SimpleStreamOptions",
    "TextContent", "ThinkingContent", "ImageContent", "ToolCall",
    "Usage", "UsageCost",
    "UserMessage", "AssistantMessage", "ToolResultMessage", "Message",
    "Tool", "Context", "Model", "ModelCost",
    "AssistantMessageEvent", "AssistantMessageEventStream",
    "EventStart", "EventTextStart", "EventTextDelta", "EventTextEnd",
    "EventThinkingStart", "EventThinkingDelta", "EventThinkingEnd",
    "EventToolCallStart", "EventToolCallDelta", "EventToolCallEnd",
    "EventDone", "EventError",
    # Models
    "get_model", "get_providers", "get_models", "calculate_cost", "supports_xhigh",
    # Registry
    "register_api_provider", "get_api_provider", "unregister_api_providers",
    # Keys
    "get_env_api_key",
    # Streaming
    "stream", "complete", "stream_simple", "complete_simple",
    # Utils
    "EventStream", "parse_partial_json", "validate_tool_arguments",
]
