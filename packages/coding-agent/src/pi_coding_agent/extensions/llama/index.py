"""
llama.cpp extension entry — mirrors packages/coding-agent/src/extensions/llama/index.ts
"""
from __future__ import annotations

from typing import Any

from .client import LlamaClient, LlamaModelInfo, format_bytes, normalize_llama_server_url
from .huggingface import HuggingFaceClient, find_hugging_face_token
from .provider import LLAMA_PROVIDER_ID, create_llama_provider
from .ui import LlamaUi, run_with_progress, show_llama_ui


def model_is_loaded(model: LlamaModelInfo) -> bool:
    return model.status.value in ("loaded", "sleeping")


def parse_hugging_face_model(value: str) -> dict[str, str | None]:
    slash = value.find("/")
    colon = value.find(":", slash + 1) if slash >= 0 else -1
    if colon < 0:
        return {"repository": value, "quantization": None}
    return {"repository": value[:colon], "quantization": value[colon + 1 :]}


async def configured_client(ctx: Any) -> LlamaClient | None:
    get_auth = getattr(getattr(ctx, "model_registry", None), "get_provider_auth", None)
    if not callable(get_auth):
        ui = getattr(ctx, "ui", None)
        if ui is not None and hasattr(ui, "notify"):
            ui.notify(f"Configure llama.cpp with /login {LLAMA_PROVIDER_ID}", "warning")
        return None
    result = await get_auth(LLAMA_PROVIDER_ID)
    if not result:
        ui = getattr(ctx, "ui", None)
        if ui is not None and hasattr(ui, "notify"):
            ui.notify(f"Configure llama.cpp with /login {LLAMA_PROVIDER_ID}", "warning")
        return None
    env = result.get("env") if isinstance(result, dict) else getattr(result, "env", None)
    auth = result.get("auth") if isinstance(result, dict) else getattr(result, "auth", None)
    configured_url = None
    if isinstance(env, dict):
        configured_url = env.get("LLAMA_BASE_URL")
    base_url = configured_url or (auth.get("baseUrl") if isinstance(auth, dict) else getattr(auth, "base_url", "")) or ""
    api_key = auth.get("apiKey") if isinstance(auth, dict) else getattr(auth, "api_key", None)
    return LlamaClient(normalize_llama_server_url(str(base_url)), api_key)


def llama_extension(pi: Any) -> None:
    provider = create_llama_provider()
    if hasattr(pi, "register_provider"):
        pi.register_provider(provider.provider)
    elif hasattr(pi, "registerProvider"):
        pi.registerProvider(provider.provider)

    async def handler(_args: Any, ctx: Any) -> None:
        if getattr(ctx, "mode", "tui") != "tui":
            ui = getattr(ctx, "ui", None)
            if ui is not None and hasattr(ui, "notify"):
                ui.notify("/llama is available in interactive mode", "warning")
            return
        client = await configured_client(ctx)
        if client is None:
            return

        async def run(ui: LlamaUi) -> None:
            try:
                catalog = await client.list()
                provider.set_catalog(catalog, client.server_url)
                action = await ui.show_models(client.server_url, catalog)
                if action.type == "download":
                    token = await find_hugging_face_token()
                    _ = HuggingFaceClient(token)
                    _ = format_bytes
                    _ = run_with_progress
            except Exception as exc:
                notify = getattr(getattr(ctx, "ui", None), "notify", None)
                if callable(notify):
                    notify(str(exc), "error")

        await show_llama_ui(ctx, run)

    if hasattr(pi, "register_command"):
        pi.register_command("llama", {"description": "Manage llama.cpp router models", "handler": handler})
    elif hasattr(pi, "registerCommand"):
        pi.registerCommand("llama", {"description": "Manage llama.cpp router models", "handler": handler})
