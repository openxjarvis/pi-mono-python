"""Generated catalog slice for openai-codex."""
from __future__ import annotations

from pi_ai.provider_factory import models_for_provider

OPENAI_CODEX_MODELS = {m.id: m for m in models_for_provider("openai-codex")}
