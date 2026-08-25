"""Generated catalog slice for opencode."""
from __future__ import annotations

from pi_ai.provider_factory import models_for_provider

OPENCODE_MODELS = {m.id: m for m in models_for_provider("opencode")}
