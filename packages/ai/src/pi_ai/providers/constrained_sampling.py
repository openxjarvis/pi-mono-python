"""
Constrained sampling helpers.
Mirrors packages/ai/src/api/constrained-sampling.ts
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from pi_ai.types import Tool

UNSUPPORTED_STRICT_SCHEMA_KEYS = (
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
)


class UnsupportedStrictJsonSchemaError(ValueError):
    pass


def _is_object(value: Any) -> bool:
    return isinstance(value, dict)


def _schema_allows_null(schema: Any) -> bool:
    if not _is_object(schema):
        return False
    schema_type = schema.get("type")
    if schema_type == "null" or (isinstance(schema_type, list) and "null" in schema_type):
        return True
    if schema.get("const") is None and "const" in schema:
        return True
    if isinstance(schema.get("enum"), list) and None in schema["enum"]:
        return True
    return isinstance(schema.get("anyOf"), list) and any(_schema_allows_null(v) for v in schema["anyOf"])


def _is_structured(schema: Any) -> bool:
    if not _is_object(schema):
        return False
    types = [schema["type"]] if isinstance(schema.get("type"), str) else list(schema.get("type") or [])
    return "object" in types or "array" in types or "properties" in schema or "items" in schema


def _make_node_strict(schema: Any) -> None:
    if not _is_object(schema):
        raise UnsupportedStrictJsonSchemaError("boolean schemas are unsupported")
    for key in UNSUPPORTED_STRICT_SCHEMA_KEYS:
        if key in schema:
            raise UnsupportedStrictJsonSchemaError(f"{key} schemas are unsupported")
    if "anyOf" in schema:
        if not schema["anyOf"]:
            raise UnsupportedStrictJsonSchemaError("anyOf must contain at least one schema")
        for variant in schema["anyOf"]:
            if _is_structured(variant):
                raise UnsupportedStrictJsonSchemaError("object and array unions are unsupported")
            _make_node_strict(variant)
    if "items" in schema:
        if isinstance(schema["items"], list):
            raise UnsupportedStrictJsonSchemaError("tuple schemas are unsupported")
        _make_node_strict(schema["items"])
    if schema.get("type") != "object":
        return
    properties = schema.get("properties") or {}
    names = list(properties)
    required = set(schema.get("required") or [])
    for key, prop in properties.items():
        _make_node_strict(prop)
        if key not in required and not _schema_allows_null(prop):
            properties[key] = {"anyOf": [prop, {"type": "null"}]}
    schema["required"] = names
    schema["additionalProperties"] = False


def make_strict_json_schema(schema: Any) -> dict[str, Any]:
    cloned = copy.deepcopy(schema)
    if not _is_object(cloned):
        raise UnsupportedStrictJsonSchemaError("root schema must have type object")
    _make_node_strict(cloned)
    if cloned.get("type") != "object":
        raise UnsupportedStrictJsonSchemaError("root schema must have type object")
    return cloned


def get_json_schema_tool_parameters(tool: Tool, strict: bool | None) -> Any:
    return make_strict_json_schema(tool.parameters) if strict else tool.parameters


@dataclass
class GrammarConstrainedSampling:
    format: str
    definition: str
    input_property: str


def resolve_json_schema_strict_sampling(tool: Tool, supports_strict_mode: bool) -> bool | None:
    config = getattr(tool, "constrained_sampling", None)
    if not config or getattr(config, "type", None) != "json_schema":
        return None
    if supports_strict_mode:
        try:
            make_strict_json_schema(tool.parameters)
            return True
        except UnsupportedStrictJsonSchemaError as error:
            if getattr(config, "strict", None) != "require":
                return None
            raise RuntimeError(
                f'Tool "{tool.name}" requires JSON-schema constrained sampling, but {error}.'
            ) from error
    if getattr(config, "strict", None) == "require":
        raise RuntimeError(
            f'Tool "{tool.name}" requires JSON-schema constrained sampling, but strict tools are unsupported.'
        )
    return None


def create_grammar_tool_input_properties(tools: list[Tool] | None, supports: bool) -> dict[str, str]:
    return {}
