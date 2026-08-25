"""Prompt template loading — mirrors harness/prompt-templates.ts."""
from __future__ import annotations

from typing import Any, Callable, Literal, TypedDict

import yaml

from pi_agent.harness.types import ExecutionEnv, FileInfo, PromptTemplate, Result, to_error

PromptTemplateDiagnosticCode = Literal["file_info_failed", "list_failed", "read_failed", "parse_failed"]


class PromptTemplateDiagnostic(TypedDict):
    type: Literal["warning"]
    code: PromptTemplateDiagnosticCode
    message: str
    path: str


async def load_prompt_templates(env: ExecutionEnv, paths: str | list[str]) -> dict[str, Any]:
    prompt_templates: list[PromptTemplate] = []
    diagnostics: list[PromptTemplateDiagnostic] = []
    for path in paths if isinstance(paths, list) else [paths]:
        info_result = await env.file_info(path)
        if not info_result["ok"]:
            if info_result["error"].code != "not_found":
                diagnostics.append(
                    {"type": "warning", "code": "file_info_failed", "message": str(info_result["error"]), "path": path}
                )
            continue
        info = info_result["value"]
        kind = await _resolve_kind(env, info, diagnostics)
        if kind == "directory":
            result = await _load_templates_from_dir(env, info["path"])
            prompt_templates.extend(result["prompt_templates"])
            diagnostics.extend(result["diagnostics"])
        elif kind == "file" and info["name"].endswith(".md"):
            result = await _load_template_from_file(env, info["path"], info["name"])
            if result["prompt_template"]:
                prompt_templates.append(result["prompt_template"])
            diagnostics.extend(result["diagnostics"])
    return {"prompt_templates": prompt_templates, "diagnostics": diagnostics}


async def load_sourced_prompt_templates(
    env: ExecutionEnv,
    inputs: list[dict[str, Any]],
    map_prompt_template: Callable[[PromptTemplate, Any], PromptTemplate] | None = None,
) -> dict[str, Any]:
    prompt_templates: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for item in inputs:
        result = await load_prompt_templates(env, item["path"])
        for template in result["prompt_templates"]:
            prompt_templates.append(
                {
                    "prompt_template": map_prompt_template(template, item["source"]) if map_prompt_template else template,
                    "source": item["source"],
                }
            )
        for diagnostic in result["diagnostics"]:
            diagnostics.append({**diagnostic, "source": item["source"]})
    return {"prompt_templates": prompt_templates, "diagnostics": diagnostics}


async def _load_templates_from_dir(env: ExecutionEnv, directory: str) -> dict[str, Any]:
    prompt_templates: list[PromptTemplate] = []
    diagnostics: list[PromptTemplateDiagnostic] = []
    entries_result = await env.list_dir(directory)
    if not entries_result["ok"]:
        diagnostics.append({"type": "warning", "code": "list_failed", "message": str(entries_result["error"]), "path": directory})
        return {"prompt_templates": prompt_templates, "diagnostics": diagnostics}
    for entry in sorted(entries_result["value"], key=lambda item: item["name"]):
        kind = await _resolve_kind(env, entry, diagnostics)
        if kind != "file" or not entry["name"].endswith(".md"):
            continue
        result = await _load_template_from_file(env, entry["path"], entry["name"])
        if result["prompt_template"]:
            prompt_templates.append(result["prompt_template"])
        diagnostics.extend(result["diagnostics"])
    return {"prompt_templates": prompt_templates, "diagnostics": diagnostics}


async def _load_template_from_file(env: ExecutionEnv, file_path: str, file_name: str) -> dict[str, Any]:
    diagnostics: list[PromptTemplateDiagnostic] = []
    raw = await env.read_text_file(file_path)
    if not raw["ok"]:
        diagnostics.append({"type": "warning", "code": "read_failed", "message": str(raw["error"]), "path": file_path})
        return {"prompt_template": None, "diagnostics": diagnostics}
    parsed = _parse_frontmatter(raw["value"])
    if not parsed["ok"]:
        diagnostics.append({"type": "warning", "code": "parse_failed", "message": str(parsed["error"]), "path": file_path})
        return {"prompt_template": None, "diagnostics": diagnostics}
    frontmatter = parsed["value"]["frontmatter"]
    body = parsed["value"]["body"]
    first_line = next((line for line in body.split("\n") if line.strip()), "")
    description = frontmatter.get("description") if isinstance(frontmatter.get("description"), str) else ""
    if not description and first_line:
        description = first_line[:60]
        if len(first_line) > 60:
            description += "..."
    return {
        "prompt_template": PromptTemplate(name=file_name.replace(".md", "").replace(".MD", ""), description=description, content=body),
        "diagnostics": diagnostics,
    }


async def _resolve_kind(env: ExecutionEnv, info: FileInfo, diagnostics: list[PromptTemplateDiagnostic]) -> str | None:
    if info["kind"] in ("file", "directory"):
        return info["kind"]
    canonical = await env.canonical_path(info["path"])
    if not canonical["ok"]:
        if canonical["error"].code != "not_found":
            diagnostics.append(
                {"type": "warning", "code": "file_info_failed", "message": str(canonical["error"]), "path": info["path"]}
            )
        return None
    target = await env.file_info(canonical["value"])
    if not target["ok"]:
        if target["error"].code != "not_found":
            diagnostics.append(
                {"type": "warning", "code": "file_info_failed", "message": str(target["error"]), "path": info["path"]}
            )
        return None
    return target["value"]["kind"] if target["value"]["kind"] in ("file", "directory") else None


def _parse_frontmatter(content: str) -> Result:
    try:
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.startswith("---"):
            return {"ok": True, "value": {"frontmatter": {}, "body": normalized}}
        end_index = normalized.find("\n---", 3)
        if end_index == -1:
            return {"ok": True, "value": {"frontmatter": {}, "body": normalized}}
        yaml_string = normalized[4:end_index]
        body = normalized[end_index + 4 :].strip()
        return {"ok": True, "value": {"frontmatter": yaml.safe_load(yaml_string) or {}, "body": body}}
    except Exception as error:
        return {"ok": False, "error": to_error(error)}


def parse_command_args(args_string: str) -> list[str]:
    args: list[str] = []
    current = ""
    in_quote: str | None = None
    for char in args_string:
        if in_quote:
            if char == in_quote:
                in_quote = None
            else:
                current += char
        elif char in ('"', "'"):
            in_quote = char
        elif char in (" ", "\t"):
            if current:
                args.append(current)
                current = ""
        else:
            current += char
    if current:
        args.append(current)
    return args


def substitute_args(content: str, args: list[str]) -> str:
    import re

    result = re.sub(r"\$(\d+)", lambda match: args[int(match.group(1)) - 1] if int(match.group(1)) - 1 < len(args) else "", content)

    def slice_args(match: re.Match[str]) -> str:
        start = max(int(match.group(1)) - 1, 0)
        length = match.group(2)
        if length:
            return " ".join(args[start : start + int(length)])
        return " ".join(args[start:])

    result = re.sub(r"\$\{@:(\d+)(?::(\d+))?\}", slice_args, result)
    all_args = " ".join(args)
    result = result.replace("$ARGUMENTS", all_args)
    result = result.replace("$@", all_args)
    return result


def format_prompt_template_invocation(template: PromptTemplate, args: list[str] | None = None) -> str:
    return substitute_args(template.content, args or [])
