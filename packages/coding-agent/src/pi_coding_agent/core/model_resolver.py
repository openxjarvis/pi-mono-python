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

# Default model IDs for each known provider (aligned with TS v0.84.3)
DEFAULT_MODEL_PER_PROVIDER: dict[str, str] = {
    "amazon-bedrock": "us.anthropic.claude-opus-4-6-v1",
    "ant-ling": "Ring-2.6-1T",
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5.5",
    "azure-openai-responses": "gpt-5.4",
    "openai-codex": "gpt-5.5",
    "radius": "auto",
    "nvidia": "nvidia/nemotron-3-super-120b-a12b",
    "deepseek": "deepseek-v4-pro",
    "google": "gemini-3.1-pro-preview",
    "google-vertex": "gemini-3.1-pro-preview",
    "google-gemini-cli": "gemini-2.5-pro",
    "google-antigravity": "gemini-3-pro-high",
    "github-copilot": "gpt-5.4",
    "openrouter": "moonshotai/kimi-k2.6",
    "vercel-ai-gateway": "zai/glm-5.1",
    "xai": "grok-4.6",
    "groq": "openai/gpt-oss-120b",
    "cerebras": "gpt-oss-120b",
    "zai": "glm-5.3",
    "zai-coding-cn": "glm-5.3",
    "mistral": "devstral-medium-latest",
    "minimax": "MiniMax-M2.7",
    "minimax-cn": "MiniMax-M2.7",
    "moonshotai": "kimi-k2.6",
    "moonshotai-cn": "kimi-k2.6",
    "huggingface": "moonshotai/Kimi-K2.6",
    "fireworks": "accounts/fireworks/models/kimi-k2p6",
    "together": "moonshotai/Kimi-K2.6",
    "baseten": "zai-org/GLM-5.2",
    "opencode": "kimi-k2.6",
    "opencode-go": "kimi-k2.6",
    "kimi-coding": "kimi-for-coding",
    "cloudflare-workers-ai": "@cf/moonshotai/kimi-k2.6",
    "cloudflare-ai-gateway": "workers-ai/@cf/moonshotai/kimi-k2.6",
    "qwen-token-plan": "qwen3.7-max",
    "qwen-token-plan-cn": "qwen3.7-max",
    "qwen-token-plan-individual": "qwen3.8-max",
    "xiaomi": "mimo-v2.5-pro",
    "xiaomi-token-plan-cn": "mimo-v2.5-pro",
    "xiaomi-token-plan-ams": "mimo-v2.5-pro",
    "xiaomi-token-plan-sgp": "mimo-v2.5-pro",
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


def find_exact_model_reference_match(model_reference: str, available_models: list[Any]) -> Any | None:
    """Find an exact model reference match.

    Supports a bare model id or a canonical provider/modelId reference.
    Ambiguous matches across providers are rejected.
    Mirrors findExactModelReferenceMatch() in TypeScript.
    """
    trimmed = model_reference.strip()
    if not trimmed:
        return None
    normalized = trimmed.lower()

    canonical = [
        m for m in available_models
        if f"{m.provider}/{m.id}".lower() == normalized
    ]
    if len(canonical) == 1:
        return canonical[0]
    if len(canonical) > 1:
        return None

    slash_idx = trimmed.find("/")
    if slash_idx != -1:
        provider = trimmed[:slash_idx].strip()
        model_id = trimmed[slash_idx + 1:].strip()
        if provider and model_id:
            provider_matches = [
                m for m in available_models
                if m.provider.lower() == provider.lower() and m.id.lower() == model_id.lower()
            ]
            if len(provider_matches) == 1:
                return provider_matches[0]
            if len(provider_matches) > 1:
                return None

    id_matches = [m for m in available_models if m.id.lower() == normalized]
    return id_matches[0] if len(id_matches) == 1 else None


def _try_match_model(pattern: str, available_models: list[Any]) -> Any | None:
    """Try to match a pattern to a model from the available models list."""
    exact = find_exact_model_reference_match(pattern, available_models)
    if exact:
        return exact

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

    def _has_auth(prov: str) -> bool:
        try:
            return bool(model_registry.get_api_key(prov))
        except Exception:
            return False

    if not provider:
        lower = cli_model.lower()
        exact_matches = [
            m for m in all_models
            if m.id.lower() == lower or f"{m.provider}/{m.id}".lower() == lower
        ]
        if len(exact_matches) == 1:
            return ResolveCliModelResult(model=exact_matches[0], thinking_level=None, warning=None, error=None)
        if len(exact_matches) > 1:
            authenticated = [m for m in exact_matches if _has_auth(m.provider)]
            if len(authenticated) == 1:
                return ResolveCliModelResult(model=authenticated[0], thinking_level=None, warning=None, error=None)
            matches = ", ".join(sorted(f"{m.provider}/{m.id}" for m in exact_matches))
            hint = (
                "No matching provider is authenticated."
                if not authenticated
                else "More than one matching provider is authenticated."
            )
            return ResolveCliModelResult(
                model=None, thinking_level=None, warning=None,
                error=f'Model "{cli_model}" is ambiguous across providers: {matches}. {hint} Use --provider or provider/model.',
            )

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

    if result.model:
        return ResolveCliModelResult(
            model=result.model,
            thinking_level=result.thinking_level,
            warning=result.warning,
            error=None,
        )

    display = f"{provider}/{pattern}" if provider else cli_model
    return ResolveCliModelResult(
        model=None, thinking_level=None, warning=result.warning,
        error=f'Model "{display}" not found. Use --list-models to see available models.',
    )


@dataclass
class InitialModelResult:
    model: Any | None
    thinking_level: str
    fallback_message: str | None


async def find_initial_model(
    *,
    scoped_models: list[ScopedModel] | None = None,
    is_continuing: bool = False,
    default_provider: str | None = None,
    default_model_id: str | None = None,
    default_thinking_level: str | None = None,
    model_thinking_levels: dict[str, str] | None = None,
    model_registry: "ModelRegistry",
) -> InitialModelResult:
    """Find the initial model using settings defaults, then provider defaults.

    Mirrors findInitialModel() in TypeScript (without CLI-exit side effects).
    """
    scoped_models = scoped_models or []
    thinking = default_thinking_level or DEFAULT_THINKING_LEVEL

    if scoped_models and not is_continuing:
        first = scoped_models[0]
        per_model = (model_thinking_levels or {}).get(f"{first.model.provider}/{first.model.id}")
        return InitialModelResult(
            model=first.model,
            thinking_level=first.thinking_level or per_model or thinking,
            fallback_message=None,
        )

    if default_provider and default_model_id:
        found = model_registry.find(default_provider, default_model_id)
        if found and model_registry.get_api_key(found.provider):
            per_model = (model_thinking_levels or {}).get(f"{default_provider}/{default_model_id}")
            return InitialModelResult(
                model=found,
                thinking_level=per_model or default_thinking_level or DEFAULT_THINKING_LEVEL,
                fallback_message=None,
            )

    available = await model_registry.get_available()
    if available:
        for provider, default_id in DEFAULT_MODEL_PER_PROVIDER.items():
            match = next((m for m in available if m.provider == provider and m.id == default_id), None)
            if match:
                return InitialModelResult(model=match, thinking_level=DEFAULT_THINKING_LEVEL, fallback_message=None)
        return InitialModelResult(model=available[0], thinking_level=DEFAULT_THINKING_LEVEL, fallback_message=None)

    return InitialModelResult(model=None, thinking_level=DEFAULT_THINKING_LEVEL, fallback_message=None)


async def restore_model_from_session(
    saved_provider: str,
    saved_model_id: str,
    current_model: Any | None,
    model_registry: "ModelRegistry",
) -> tuple[Any | None, str | None]:
    """Restore a model from session, falling back to available models."""
    restored = model_registry.find(saved_provider, saved_model_id)
    has_auth = bool(restored and model_registry.get_api_key(restored.provider))
    if restored and has_auth:
        return restored, None

    reason = "model no longer exists" if not restored else "no auth configured"
    if current_model:
        return current_model, (
            f"Could not restore model {saved_provider}/{saved_model_id} ({reason}). "
            f"Using {current_model.provider}/{current_model.id}."
        )

    available = await model_registry.get_available()
    if available:
        fallback = None
        for provider, default_id in DEFAULT_MODEL_PER_PROVIDER.items():
            match = next((m for m in available if m.provider == provider and m.id == default_id), None)
            if match:
                fallback = match
                break
        if fallback is None:
            fallback = available[0]
        return fallback, (
            f"Could not restore model {saved_provider}/{saved_model_id} ({reason}). "
            f"Using {fallback.provider}/{fallback.id}."
        )
    return None, None
