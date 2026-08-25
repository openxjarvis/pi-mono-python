"""Generated catalog slice for amazon-bedrock."""
from __future__ import annotations

from pi_ai.provider_factory import models_for_provider

AMAZON_BEDROCK_MODELS = {m.id: m for m in models_for_provider("amazon-bedrock")}
