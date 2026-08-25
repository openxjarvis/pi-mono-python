"""Generated catalog slice for github-copilot."""
from __future__ import annotations

from pi_ai.provider_factory import models_for_provider

GITHUB_COPILOT_MODELS = {m.id: m for m in models_for_provider("github-copilot")}
