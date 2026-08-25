"""Generated catalog slice for groq."""
from __future__ import annotations

from pi_ai.provider_factory import models_for_provider

GROQ_MODELS = {m.id: m for m in models_for_provider("groq")}
