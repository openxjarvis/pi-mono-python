"""
Environment variable API key resolution — mirrors packages/ai/src/env-api-keys.ts
"""
from __future__ import annotations

import os

# Maps provider name → environment variable name
PROVIDER_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "google-gemini-cli": "GOOGLE_API_KEY",
    "google-vertex": "GOOGLE_APPLICATION_CREDENTIALS",
    "amazon-bedrock": "AWS_ACCESS_KEY_ID",  # Bedrock uses AWS credentials
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "vercel-ai-gateway": "VERCEL_AI_GATEWAY_API_KEY",
    "azure-openai-responses": "AZURE_OPENAI_API_KEY",
    "github-copilot": "GITHUB_TOKEN",
    "cerebras": "CEREBRAS_API_KEY",
    "zai": "ZAI_API_KEY",
    "huggingface": "HF_TOKEN",
    "minimax": "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
    "opencode": "OPENCODE_API_KEY",
}


def get_env_api_key(provider: str) -> str | None:
    """Resolve an API key from environment variables for the given provider."""
    env_var = PROVIDER_ENV_VARS.get(provider)
    if env_var:
        key = os.environ.get(env_var)
        if key:
            return key
    # Allow GEMINI_API_KEY as alias for GOOGLE_API_KEY (for compatibility)
    if provider in ("google", "google-gemini-cli"):
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            return key
    # Fallback: try common naming patterns
    normalized = provider.upper().replace("-", "_")
    return os.environ.get(f"{normalized}_API_KEY")
