from __future__ import annotations

import os

from .pi_harness import create_pi_coding_agent_harness

pi_coding_agent_harness = create_pi_coding_agent_harness({"noTools": "all"})


async def run_smoke_eval() -> dict:
    result = await pi_coding_agent_harness["run"]("What's the capital of France? Respond with only the city name.")
    if result["output"].strip() != "Paris":
        raise AssertionError(f"Expected Paris, got {result['output']!r}")
    if result["usage"]["provider"] != os.environ.get("PI_PROVIDER"):
        raise AssertionError("Provider mismatch")
    if result["usage"]["model"] != os.environ.get("PI_MODEL"):
        raise AssertionError("Model mismatch")
    if result["usage"]["totalTokens"] <= 0:
        raise AssertionError("Expected token usage")
    return result
