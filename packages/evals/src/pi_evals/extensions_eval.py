from __future__ import annotations

from .pi_harness import create_pi_coding_agent_harness

extensions_harness = create_pi_coding_agent_harness({"name": "pi-coding-agent-extensions"})


async def run_extensions_eval(prompt: str = "Say ready.") -> dict:
    return await extensions_harness["run"](prompt)
