"""
Built-in provider factories from the generated catalog.
Mirrors packages/ai/src/providers/*.ts
"""
from __future__ import annotations

from pi_ai.env_api_keys import PROVIDER_ENV_VARS
from pi_ai.provider_factory import Provider, create_provider, models_for_provider

PROVIDER_META: dict[str, tuple[str, str | None]] = {
    "amazon-bedrock": ("Amazon Bedrock", None),
    "ant-ling": ("Ant Ling", "https://api.antling.com"),
    "anthropic": ("Anthropic", "https://api.anthropic.com"),
    "azure-openai-responses": ("Azure OpenAI Responses", None),
    "baseten": ("Baseten", "https://inference.baseten.co"),
    "cerebras": ("Cerebras", "https://api.cerebras.ai"),
    "cloudflare-ai-gateway": ("Cloudflare AI Gateway", None),
    "cloudflare-workers-ai": ("Cloudflare Workers AI", None),
    "deepseek": ("DeepSeek", "https://api.deepseek.com"),
    "fireworks": ("Fireworks", "https://api.fireworks.ai"),
    "github-copilot": ("GitHub Copilot", None),
    "google": ("Google", None),
    "google-vertex": ("Google Vertex", None),
    "groq": ("Groq", "https://api.groq.com"),
    "huggingface": ("Hugging Face", "https://router.huggingface.co"),
    "kimi-coding": ("Kimi Coding", "https://api.kimi.com"),
    "minimax": ("MiniMax", "https://api.minimax.io"),
    "minimax-cn": ("MiniMax China", "https://api.minimaxi.com"),
    "mistral": ("Mistral", "https://api.mistral.ai"),
    "moonshotai": ("Moonshot AI", "https://api.moonshot.ai"),
    "moonshotai-cn": ("Moonshot AI China", "https://api.moonshot.cn"),
    "nvidia": ("NVIDIA NIM", "https://integrate.api.nvidia.com"),
    "openai": ("OpenAI", "https://api.openai.com"),
    "openai-codex": ("OpenAI Codex", "https://chatgpt.com/backend-api"),
    "opencode": ("OpenCode", None),
    "opencode-go": ("OpenCode Go", None),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api"),
    "qwen-token-plan": ("Qwen Token Plan", None),
    "qwen-token-plan-cn": ("Qwen Token Plan China", None),
    "qwen-token-plan-individual": ("Qwen Token Plan Individual", None),
    "radius": ("Radius", None),
    "together": ("Together AI", "https://api.together.xyz"),
    "vercel-ai-gateway": ("Vercel AI Gateway", "https://ai-gateway.vercel.sh"),
    "xai": ("xAI", "https://api.x.ai"),
    "xiaomi": ("Xiaomi", None),
    "xiaomi-token-plan-ams": ("Xiaomi Token Plan AMS", None),
    "xiaomi-token-plan-cn": ("Xiaomi Token Plan CN", None),
    "xiaomi-token-plan-sgp": ("Xiaomi Token Plan SGP", None),
    "zai": ("Z.AI", "https://api.z.ai"),
    "zai-coding-cn": ("Z.AI Coding CN", None),
}


def _env_vars(provider_id: str) -> list[str]:
    value = PROVIDER_ENV_VARS.get(provider_id)
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)


def create_builtin_provider(provider_id: str) -> Provider:
    name, base_url = PROVIDER_META.get(provider_id, (provider_id, None))
    return create_provider(
        id=provider_id,
        name=name,
        env_vars=_env_vars(provider_id),
        base_url=base_url,
        models=models_for_provider(provider_id),
    )


def builtin_providers() -> list[Provider]:
    return [create_builtin_provider(pid) for pid in PROVIDER_META]


def get_builtin_providers() -> list[str]:
    return list(PROVIDER_META)


def get_builtin_models(provider: str) -> list:
    return models_for_provider(provider)
