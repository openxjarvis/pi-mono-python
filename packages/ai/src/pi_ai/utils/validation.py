"""
Tool argument validation — mirrors packages/ai/src/utils/validation.ts
"""
from __future__ import annotations

from typing import Any

from ..types import Tool, ToolCall


def validate_tool_arguments(tool: Tool, tool_call: ToolCall) -> dict[str, Any]:
    """
    Validate and return tool arguments against the tool's parameter schema.

    Mirrors the TypeScript validateToolArguments() that uses AJV.
    Uses basic JSON Schema validation here.

    Returns the validated arguments dict.
    Raises ValueError if validation fails.
    """
    args = tool_call.arguments
    schema = tool.parameters

    if not isinstance(args, dict):
        raise ValueError(f"Tool arguments must be an object, got {type(args).__name__}")

    required = schema.get("required", [])
    for field in required:
        if field not in args:
            raise ValueError(f"Missing required parameter '{field}' for tool '{tool.name}'")

    properties = schema.get("properties", {})
    for key, value in args.items():
        if properties and key not in properties:
            # Extra fields are generally OK (lenient validation)
            pass

    return args
