"""Generated catalog slice for deepseek."""
from __future__ import annotations

from pi_ai.provider_factory import models_for_provider

DEEPSEEK_MODELS = {m.id: m for m in models_for_provider("deepseek")}
