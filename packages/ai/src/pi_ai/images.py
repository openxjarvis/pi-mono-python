"""
Image generation entrypoints.
Mirrors packages/ai/src/images.ts
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GeneratedImage:
    url: str | None = None
    b64_json: str | None = None
    mime_type: str | None = None


@dataclass
class ImageGenerationRequest:
    prompt: str
    model: str | None = None
    n: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


async def generate_images(request: ImageGenerationRequest, *, api_key: str | None = None) -> list[GeneratedImage]:
    """Generate images via OpenRouter-compatible `/images` when configured."""
    if not api_key:
        return []
    import httpx

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/images",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"prompt": request.prompt, "model": request.model, "n": request.n, **request.extra},
        )
        if not response.is_success:
            return []
        data = response.json()
    images: list[GeneratedImage] = []
    for item in data.get("data") or []:
        images.append(GeneratedImage(url=item.get("url"), b64_json=item.get("b64_json"), mime_type=item.get("mime_type")))
    return images
