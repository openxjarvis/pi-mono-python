"""
llama.cpp provider — mirrors packages/coding-agent/src/extensions/llama/provider.ts
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .client import LlamaClient, LlamaModelInfo, llama_inference_url, normalize_llama_server_url

LLAMA_PROVIDER_ID = "llama.cpp"
DEFAULT_LLAMA_SERVER_URL = "http://127.0.0.1:8080"


def model_is_selectable(model: LlamaModelInfo, router_autoload: bool) -> bool:
    if model.status.value in ("loaded", "sleeping"):
        return True
    return router_autoload and model.status.value == "unloaded" and not model.status.failed and model.source == "preset"


def to_pi_model(model: LlamaModelInfo, server_url: str) -> dict[str, Any]:
    reported = None
    if model.meta:
        reported = model.meta.get("n_ctx") or model.meta.get("n_ctx_train")
    context_window = reported if isinstance(reported, int) and reported > 0 else 128000
    input_modalities = (model.architecture or {}).get("input_modalities") or []
    return {
        "id": model.id,
        "name": model.id,
        "api": "openai-completions",
        "provider": LLAMA_PROVIDER_ID,
        "baseUrl": llama_inference_url(server_url),
        "reasoning": False,
        "input": ["text", "image"] if "image" in input_modalities else ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": context_window,
        "maxTokens": context_window,
    }


@dataclass
class LlamaProviderController:
    provider: dict[str, Any]

    def set_catalog(
        self,
        models: list[LlamaModelInfo],
        server_url: str,
        router_autoload: bool = False,
    ) -> None:
        selectable = [model for model in models if model_is_selectable(model, router_autoload)]
        self.provider["models"] = [to_pi_model(model, server_url) for model in selectable]
        self.provider["baseUrl"] = llama_inference_url(server_url)


def create_llama_provider() -> LlamaProviderController:
    provider = {
        "id": LLAMA_PROVIDER_ID,
        "name": "llama.cpp",
        "baseUrl": llama_inference_url(DEFAULT_LLAMA_SERVER_URL),
        "models": [],
        "api": "openai-completions",
    }
    return LlamaProviderController(provider=provider)
