"""
Generate `pi_ai/models_generated.py` from TypeScript `models.generated.ts`.

This keeps Python model catalog aligned with `pi-mono/packages/ai/src/models.generated.ts`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROVIDER_BLOCK_RE = re.compile(r'^\t"([^"]+)": \{$')
MODEL_START_RE = re.compile(r'^\t\t"([^"]+)": \{$')
MODEL_END_RE = re.compile(r'^\t\t\} satisfies Model<"([^"]+)">,?$')


def _extract_between(lines: list[str], start: str, end: str) -> str | None:
    try:
        s_idx = lines.index(start)
    except ValueError:
        return None
    for i in range(s_idx + 1, len(lines)):
        if lines[i] == end:
            return "\n".join(lines[s_idx + 1 : i])
    return None


def _extract_scalar(block: str, key: str) -> str | None:
    m = re.search(rf'{re.escape(key)}:\s*"([^"]*)"', block)
    return m.group(1) if m else None


def _extract_bool(block: str, key: str) -> bool:
    m = re.search(rf"{re.escape(key)}:\s*(true|false)", block)
    return (m.group(1) == "true") if m else False


def _extract_int(block: str, key: str, default: int) -> int:
    m = re.search(rf"{re.escape(key)}:\s*([0-9]+)", block)
    return int(m.group(1)) if m else default


def _extract_inputs(block: str) -> list[str]:
    m = re.search(r"input:\s*\[([^\]]*)\]", block)
    if not m:
        return ["text"]
    return re.findall(r'"([^"]+)"', m.group(1)) or ["text"]


def _extract_cost(block: str) -> dict[str, float]:
    m = re.search(r"cost:\s*\{(.*?)\n\t\t\t\},", block, re.DOTALL)
    if not m:
        return {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0}
    c = m.group(1)

    def _v(name: str) -> float:
        mm = re.search(rf"{name}:\s*([0-9]+(?:\.[0-9]+)?)", c)
        return float(mm.group(1)) if mm else 0.0

    return {
        "input": _v("input"),
        "output": _v("output"),
        "cacheRead": _v("cacheRead"),
        "cacheWrite": _v("cacheWrite"),
    }


def _extract_json_object(block: str, key: str) -> dict | None:
    m = re.search(rf"{re.escape(key)}:\s*(\{{.*\}}),?$", block, re.MULTILINE)
    if not m:
        return None
    try:
        value = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_ts_models(ts_path: Path) -> list[dict]:
    text = ts_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    in_models = False
    provider = ""
    in_model = False
    model_key = ""
    model_lines: list[str] = []
    records: list[dict] = []

    for line in lines:
        if not in_models:
            if line.strip() == "export const MODELS = {":
                in_models = True
            continue
        if line.strip() == "} as const;" or line.strip() == "};":
            break

        if not in_model:
            pm = PROVIDER_BLOCK_RE.match(line)
            if pm:
                provider = pm.group(1)
                continue
            mm = MODEL_START_RE.match(line)
            if mm:
                model_key = mm.group(1)
                in_model = True
                model_lines = []
                continue
            continue

        end = MODEL_END_RE.match(line)
        if end:
            api = end.group(1)
            block = "\n".join(model_lines)
            record = {
                "dict_key": f"{provider}/{model_key}",
                "id": _extract_scalar(block, "id") or model_key,
                "name": _extract_scalar(block, "name") or model_key,
                "api": api,
                "provider": _extract_scalar(block, "provider") or provider,
                "baseUrl": _extract_scalar(block, "baseUrl") or "",
                "reasoning": _extract_bool(block, "reasoning"),
                "input": _extract_inputs(block),
                "cost": _extract_cost(block),
                "contextWindow": _extract_int(block, "contextWindow", 128000),
                "maxTokens": _extract_int(block, "maxTokens", 8192),
                "headers": _extract_json_object(block, "headers"),
                "compat": _extract_json_object(block, "compat"),
            }
            records.append(record)
            in_model = False
            model_key = ""
            model_lines = []
            continue

        model_lines.append(line)

    return records


def render_python(records: list[dict], source_hint: str) -> str:
    out: list[str] = []
    out.append('"""')
    out.append("Auto-generated model definitions — mirrors TypeScript models.generated.ts")
    out.append("")
    out.append("DO NOT EDIT MANUALLY.")
    out.append(f"Source: {source_hint}")
    out.append('"""')
    out.append("from __future__ import annotations")
    out.append("")
    out.append("from .types import Model, ModelCost")
    out.append("")
    out.append("MODELS: dict[str, Model] = {")

    for r in sorted(records, key=lambda x: x["dict_key"]):
        cost = r["cost"]
        inputs = ", ".join(f'"{i}"' for i in r["input"])
        out.append(f'    "{r["dict_key"]}": Model(')
        out.append(f'        id={r["id"]!r},')
        out.append(f'        name={r["name"]!r},')
        out.append(f'        api={r["api"]!r},')
        out.append(f'        provider={r["provider"]!r},')
        out.append(f'        base_url={r["baseUrl"]!r},')
        out.append(f'        reasoning={str(r["reasoning"])},')
        out.append(f"        input=[{inputs}],")
        out.append(
            "        cost=ModelCost("
            f"input={cost['input']}, output={cost['output']}, "
            f"cache_read={cost['cacheRead']}, cache_write={cost['cacheWrite']}"
            "),"
        )
        out.append(f'        context_window={r["contextWindow"]},')
        out.append(f'        max_tokens={r["maxTokens"]},')
        if r["headers"] is not None:
            out.append(f'        headers={r["headers"]!r},')
        if r["compat"] is not None:
            out.append(f'        compat={r["compat"]!r},')
        out.append("    ),")

    out.append("}")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Python model catalog from TS generated file")
    parser.add_argument("--source-ts", type=Path, default=None)
    parser.add_argument(
        "--output-py",
        type=Path,
        default=Path(__file__).resolve().parent / "src" / "pi_ai" / "models_generated.py",
    )
    args = parser.parse_args()

    if args.source_ts is None:
        default = Path(__file__).resolve().parents[3] / "pi-mono" / "packages" / "ai" / "src" / "models.generated.ts"
        args.source_ts = default

    if not args.source_ts.exists():
        raise FileNotFoundError(f"TS source not found: {args.source_ts}")

    records = parse_ts_models(args.source_ts)
    if not records:
        raise RuntimeError("No models parsed from TypeScript source")

    rendered = render_python(records, str(args.source_ts))
    args.output_py.write_text(rendered, encoding="utf-8")
    print(f"Wrote {len(records)} models to {args.output_py}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
