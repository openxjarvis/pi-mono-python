"""Native module candidate paths. Mirrors packages/tui/src/native-module-path.ts"""
from __future__ import annotations

import sys
from pathlib import Path


def get_native_module_candidates(native_path: str, options: dict | None = None) -> list[str]:
    options = options or {}
    module_dir = Path(__file__).resolve().parent
    candidates = [
        str(module_dir.parent / native_path),
        str(module_dir / native_path),
        str(Path(options.get("exec_path") or sys.executable).parent / native_path),
    ]
    return list(dict.fromkeys(candidates))
