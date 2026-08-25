"""
Environment variable API key resolution — mirrors packages/ai/src/env-api-keys.ts
"""
from __future__ import annotations

import os
from pathlib import Path

ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_OAUTH_TOKEN_ENV = "ANTHROPIC_OAUTH_TOKEN"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

# Maps provider name → primary environment variable name
PROVIDER_ENV_VARS: dict[str, str] = {
    "ant-ling": "ANT_LING_API_KEY",
    "qwen-token-plan": "QWEN_TOKEN_PLAN_API_KEY",
    "qwen-token-plan-cn": "QWEN_TOKEN_PLAN_CN_API_KEY",
    "qwen-token-plan-individual": "QWEN_TOKEN_PLAN_API_KEY",
    "openai": "OPENAI_API_KEY",
    "azure-openai-responses": "AZURE_OPENAI_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GEMINI_API_KEY",
    "google-vertex": "GOOGLE_CLOUD_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "xai": "XAI_API_KEY",
    "radius": "RADIUS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "vercel-ai-gateway": "AI_GATEWAY_API_KEY",
    "zai": "ZAI_API_KEY",
    "zai-coding-cn": "ZAI_CODING_CN_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
    "moonshotai": "MOONSHOT_API_KEY",
    "moonshotai-cn": "MOONSHOT_API_KEY",
    "huggingface": "HF_TOKEN",
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "opencode": "OPENCODE_API_KEY",
    "opencode-go": "OPENCODE_API_KEY",
    "kimi-coding": "KIMI_API_KEY",
    "kimi": "KIMI_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "cloudflare-workers-ai": "CLOUDFLARE_API_KEY",
    "cloudflare-ai-gateway": "CLOUDFLARE_API_KEY",
    "xiaomi": "XIAOMI_API_KEY",
    "xiaomi-token-plan-cn": "XIAOMI_TOKEN_PLAN_CN_API_KEY",
    "xiaomi-token-plan-ams": "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
    "xiaomi-token-plan-sgp": "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
}


def _get_api_key_env_vars(provider: str) -> list[str] | None:
    if provider == "github-copilot":
        return ["COPILOT_GITHUB_TOKEN"]
    if provider == "anthropic":
        return [ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV]
    env_var = PROVIDER_ENV_VARS.get(provider)
    return [env_var] if env_var else None


def find_env_keys(provider: str, env: dict[str, str] | None = None) -> list[str] | None:
    """Find configured environment variables that can provide an API key.

    Reports actual API key variables only — excludes ambient credential sources
    such as AWS profiles and Google Application Default Credentials.
    """
    source = env if env is not None else os.environ
    env_vars = _get_api_key_env_vars(provider)
    if not env_vars:
        return None
    found = [v for v in env_vars if source.get(v)]
    return found or None


def get_env_api_key(provider: str, env: dict[str, str] | None = None) -> str | None:
    """Resolve an API key from environment variables for the given provider.

    Will not return API keys for providers that require OAuth tokens.
    """
    source = env if env is not None else os.environ

    # GitHub Copilot: COPILOT_GITHUB_TOKEN is the documented env var.
    # Also accept GH_TOKEN / GITHUB_TOKEN as a practical fallback.
    if provider == "github-copilot":
        return (
            source.get("COPILOT_GITHUB_TOKEN")
            or source.get("GH_TOKEN")
            or source.get("GITHUB_TOKEN")
        )

    env_keys = find_env_keys(provider, source)
    if env_keys:
        if provider == "anthropic":
            api_key_env = next((k for k in env_keys if k != ANTHROPIC_AUTH_TOKEN_ENV), None)
            if api_key_env:
                return source.get(api_key_env)
        else:
            return source.get(env_keys[0])

    # Vertex AI supports either an explicit API key or Application Default Credentials.
    if provider == "google-vertex":
        api_key = source.get("GOOGLE_CLOUD_API_KEY")
        if api_key:
            return api_key

        gac_path = source.get("GOOGLE_APPLICATION_CREDENTIALS")
        has_credentials = False
        if gac_path and Path(gac_path).exists():
            has_credentials = True
        else:
            default_adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
            has_credentials = default_adc.exists()

        has_project = bool(source.get("GOOGLE_CLOUD_PROJECT") or source.get("GCLOUD_PROJECT"))
        has_location = bool(source.get("GOOGLE_CLOUD_LOCATION"))
        if has_credentials and has_project and has_location:
            return "<authenticated>"
        return None

    if provider == "amazon-bedrock":
        if (
            source.get("AWS_PROFILE")
            or (source.get("AWS_ACCESS_KEY_ID") and source.get("AWS_SECRET_ACCESS_KEY"))
            or source.get("AWS_BEARER_TOKEN_BEDROCK")
            or source.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
            or source.get("AWS_CONTAINER_CREDENTIALS_FULL_URI")
            or source.get("AWS_WEB_IDENTITY_TOKEN_FILE")
        ):
            return "<authenticated>"
        return None

    return None
