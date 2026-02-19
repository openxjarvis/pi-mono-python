"""
Model resolution, scoping, and initial selection.

Handles fuzzy model matching, glob patterns, thinking level parsing,
and provider-based model lookups.

Mirrors core/model-resolver.ts
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pi_coding_agent.core.defaults import DEFAULT_THINKING_LEVEL

if TYPE_CHECKING:
    from pi_coding_agent.core.model_registry import ModelRegistry

# Default model IDs for each known provider
DEFAULT_MODEL_PER_PROVIDER: dict[str, str] = {
    "amazon-bedrock": "us.anthropic.claude-opus-4-6-v1",
    "anthropic": "claude-opus-4-6",
    "openai": "gpt-5.1-codex",
    "azure-openai-responses": "gpt-5.2",
    "openai-codex": "gpt-5.3-codex",
    "google": "gemini-2.5-pro",
    "google-gemini-cli": "gemini-2.5-pro",
    "google-antigravity": "gemini-3-pro-high",
    "google-vertex": "gemini-3-pro-preview",
    "github-copilot": "gpt-4o",
    "openrouter": "openai/gpt-5.1-codex",
    "vercel-ai-gateway": "anthropic/claude-opus-4-6",
    "xai": "grok-4-fast-non-reasoning",
    "groq": "openai/gpt-oss-120b",
    "cerebras": "zai-glm-4.6",
    "zai": "glm-4.6",
    "mistral": "devstral-medium-latest",
    "minimax": "MiniMax-M2.1",
    "minimax-cn": "MiniMax-M2.1",
    "huggingface": "moonshotai/Kimi-K2.5",
    "opencode": "claude-opus-4-6",
    "kimi-coding": "kimi-k2-thinking",
}

_VALID_THINKING_LEVELS = {"minimal", "low", "medium", "high", "xhigh", "off"}

import re as _re
_DATE_PATTERN = _re.compile(r"-\d{8}$")


def is_valid_thinking_level(level: str) -> bool:
    return level in _VALID_THINKING_LEVELS


def _is_alias(model_id: str) -> bool:
    """Return True if model ID looks like an alias (no date suffix)."""
    if model_id.endswith("-latest"):
        return True
    return not bool(_DATE_PATTERN.search(model_id))


def _try_match_model(pattern: str, available_models: list[Any]) -> Any | None:
    """Try to match a pattern to a model from the available models list."""
    slash_idx = pattern.find("/")
    if slash_idx != -1:
        provider_part = pattern[:slash_idx]
        model_id_part = pattern[slash_idx + 1:]
        for m in available_models:
            if (getattr(m, "provider", "").lower() == provider_part.lower()
                    and m.id.lower() == model_id_part.lower()):
                return m

    # Exact ID match (case-insensitive)
    for m in available_models:
        if m.id.lower() == pattern.lower():
            return m

    # Partial matching
    lower = pattern.lower()
    matches = [
        m for m in available_models
        if lower in m.id.lower() or lower in (getattr(m, "name", "") or "").lower()
    ]
    if not matches:
        return None

    aliases = [m for m in matches if _is_alias(m.id)]
    dated = [m for m in matches if not _is_alias(m.id)]

    if aliases:
        aliases.sort(key=lambda m: m.id, reverse=True)
        return aliases[0]
    else:
        dated.sort(key=lambda m: m.id, reverse=True)
        return dated[0]


@dataclass
class ParsedModelResult:
    model: Any | None
    thinking_level: str | None
    warning: str | None


def parse_model_pattern(
    pattern: str,
    available_models: list[Any],
    allow_invalid_thinking_level_fallback: bool = True,
) -> ParsedModelResult:
    """Parse a pattern to extract model and thinking level.

    Algorithm:
    1. Try to match full pattern as a model.
    2. If found, return it without a thinking level.
    3. If not found and has colons, split on last colon:
       - If suffix is valid thinking level, use it and recurse on prefix.
       - If suffix is invalid, warn and recurse on prefix with None.
    """
    exact = _try_match_model(pattern, available_models)
    if exact:
        return ParsedModelResult(model=exact, thinking_level=None, warning=None)

    last_colon = pattern.rfind(":")
    if last_colon == -1:
        return ParsedModelResult(model=None, thinking_level=None, warning=None)

    prefix = pattern[:last_colon]
    suffix = pattern[last_colon + 1:]

    if is_valid_thinking_level(suffix):
        result = parse_model_pattern(prefix, available_models, allow_invalid_thinking_level_fallback)
        if result.model:
            return ParsedModelResult(
                model=result.model,
                thinking_level=None if result.warning else suffix,
                warning=result.warning,
            )
        return result
    else:
        if not allow_invalid_thinking_level_fallback:
            return ParsedModelResult(model=None, thinking_level=None, warning=None)
        result = parse_model_pattern(prefix, available_models, allow_invalid_thinking_level_fallback)
        if result.model:
            return ParsedModelResult(
                model=result.model,
                thinking_level=None,
                warning=f'Invalid thinking level "{suffix}" in pattern "{pattern}". Using default instead.',
            )
        return result


@dataclass
class ScopedModel:
    model: Any
    thinking_level: str | None = None


async def resolve_model_scope(patterns: list[str], model_registry: "ModelRegistry") -> list[ScopedModel]:
    """Resolve model patterns (with optional :level suffix and globs) to ScopedModel list."""
    available_models = await model_registry.get_available()
    scoped: list[ScopedModel] = []

    def _already_added(m: Any) -> bool:
        return any(sm.model.id == m.id and sm.model.provider == m.provider for sm in scoped)

    for pattern in patterns:
        # Glob patterns
        if any(c in pattern for c in ("*", "?", "[")):
            colon_idx = pattern.rfind(":")
            glob_pattern = pattern
            thinking_level: str | None = None
            if colon_idx != -1:
                suffix = pattern[colon_idx + 1:]
                if is_valid_thinking_level(suffix):
                    thinking_level = suffix
                    glob_pattern = pattern[:colon_idx]

            matching = [
                m for m in available_models
                if fnmatch.fnmatch(f"{m.provider}/{m.id}".lower(), glob_pattern.lower())
                or fnmatch.fnmatch(m.id.lower(), glob_pattern.lower())
            ]
            if not matching:
                import sys
                print(f"Warning: No models match pattern \"{pattern}\"", file=sys.stderr)
                continue
            for m in matching:
                if not _already_added(m):
                    scoped.append(ScopedModel(model=m, thinking_level=thinking_level))
            continue

        result = parse_model_pattern(pattern, available_models)
        if result.warning:
            import sys
            print(f"Warning: {result.warning}", file=sys.stderr)
        if not result.model:
            import sys
            print(f"Warning: No models match pattern \"{pattern}\"", file=sys.stderr)
            continue
        if not _already_added(result.model):
            scoped.append(ScopedModel(model=result.model, thinking_level=result.thinking_level))

    return scoped


@dataclass
class ResolveCliModelResult:
    model: Any | None
    thinking_level: str | None
    warning: str | None
    error: str | None


def resolve_cli_model(
    cli_provider: str | None,
    cli_model: str | None,
    model_registry: "ModelRegistry",
) -> ResolveCliModelResult:
    """Resolve a single model from CLI flags with fuzzy matching."""
    if not cli_model:
        return ResolveCliModelResult(model=None, thinking_level=None, warning=None, error=None)

    all_models = model_registry.get_all()
    if not all_models:
        return ResolveCliModelResult(
            model=None, thinking_level=None, warning=None,
            error="No models available. Check your installation or add models to models.json.",
        )

    provider_map = {m.provider.lower(): m.provider for m in all_models}
    provider = provider_map.get(cli_provider.lower()) if cli_provider else None

    if cli_provider and not provider:
        return ResolveCliModelResult(
            model=None, thinking_level=None, warning=None,
            error=f'Unknown provider "{cli_provider}". Use --list-models to see available providers/models.',
        )

    if not provider:
        lower = cli_model.lower()
        exact = next(
            (m for m in all_models
             if m.id.lower() == lower or f"{m.provider}/{m.id}".lower() == lower),
            None,
        )
        if exact:
            return ResolveCliModelResult(model=exact, thinking_level=None, warning=None, error=None)

    pattern = cli_model
    if not provider:
        slash_idx = cli_model.find("/")
        if slash_idx != -1:
            maybe_provider = cli_model[:slash_idx]
            canonical = provider_map.get(maybe_provider.lower())
            if canonical:
                provider = canonical
                pattern = cli_model[slash_idx + 1:]
    else:
        prefix = f"{provider}/"
        if cli_model.lower().startswith(prefix.lower()):
            pattern = cli_model[len(prefix):]

    candidates = [m for m in all_models if m.provider == provider] if provider else all_models
    result = parse_model_pattern(pattern, candidates, allow_invalid_thinking_level_fallback=False)

    if not result.model:
        display = f"{provider}/{pattern}" if provider else cli_model
        return ResolveCliModelResult(
            model=None, thinking_level=None, warning=result.warning,
            error=f'Model "{display}" not found. Use --list-models to see available models.',
        )

    return ResolveCliModelResult(
        model=result.model,
        thinking_level=result.thinking_level,
        warning=result.warning,
        error=None,
    )
