"""Generated catalog slice for mistral."""
from __future__ import annotations

from pi_ai.provider_factory import models_for_provider

MISTRAL_MODELS = {m.id: m for m in models_for_provider("mistral")}
