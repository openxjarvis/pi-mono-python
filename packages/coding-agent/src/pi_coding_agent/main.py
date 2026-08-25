"""Main entry point — mirrors packages/coding-agent/src/main.ts."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .cli_sub.args import parse_args, print_help
from .cli_sub.file_processor import process_file_arguments
from .cli_sub.list_models import list_models
from .cli_sub.session_picker import select_session
from .config import get_agent_dir
from .core.auth_storage import AuthStorage
from .core.event_bus import create_event_bus
from .core.extensions.loader import load_extensions
from .core.model_registry import ModelRegistry
from .core.sdk import CreateAgentSessionOptions, create_agent_session
from .core.session_manager import SessionManager
from .core.settings_manager import SettingsManager
from .migrations import run_migrations, show_deprecation_warnings
from .modes import run_interactive_mode, run_print_mode, run_rpc_mode
from .package_manager_cli import (
    get_package_command_usage,
    parse_package_command,
    print_package_command_help,
    run_config_command,
    run_package_command,
)


def _load_env_files(cwd: str) -> None:
    """Load .env from current workspace (best-effort)."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    try:
        load_dotenv(os.path.join(cwd, ".env"), override=False)
    except Exception:
        pass


async def _build_initial_prompt(parsed) -> tuple[str, list[dict] | None]:
    """Build initial prompt text/images from @file args + positional messages."""
    text = ""
    images = None
    if parsed.file_args:
        processed = await process_file_arguments(parsed.file_args)
        text = processed.text
        images = processed.images or None

    if parsed.messages:
        # Match TS behavior: first positional message is prompt, rest handled by mode
        text = f"{text}{parsed.messages[0]}"
        parsed.messages = parsed.messages[1:]
    return text, images


async def _read_piped_stdin() -> str | None:
    """Read piped stdin content; return None when stdin is TTY."""
    if sys.stdin.isatty():
        return None
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, sys.stdin.read)
    data = (data or "").strip()
    return data or None


async def _prompt_confirm(message: str) -> bool:
    print(f"{message} [y/N] ", end="", flush=True)
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, sys.stdin.readline)
    answer = (answer or "").strip().lower()
    return answer in {"y", "yes"}


def _looks_like_path(value: str) -> bool:
    return "/" in value or "\\" in value or value.endswith(".jsonl")


async def _resolve_session_path(session_arg: str, cwd: str, session_dir: str | None) -> dict[str, Any]:
    if _looks_like_path(session_arg):
        return {"type": "path", "path": session_arg}

    local_sessions = await SessionManager.list(cwd, session_dir)
    local_matches = [s for s in local_sessions if s.session_id.startswith(session_arg)]
    if local_matches:
        return {"type": "local", "path": local_matches[0].file_path}

    global_sessions = await SessionManager.list_all()
    global_matches = [s for s in global_sessions if s.session_id.startswith(session_arg)]
    if global_matches:
        match = global_matches[0]
        return {"type": "global", "path": match.file_path, "cwd": match.cwd}

    return {"type": "not_found", "arg": session_arg}


async def _create_session_manager(parsed: Any, cwd: str) -> SessionManager | None:
    if parsed.no_session:
        return SessionManager.in_memory(cwd)
    if parsed.session:
        resolved = await _resolve_session_path(parsed.session, cwd, parsed.session_dir)
        rtype = resolved["type"]
        if rtype in {"path", "local"}:
            return SessionManager.open(resolved["path"], parsed.session_dir)
        if rtype == "global":
            print(f"Session found in different project: {resolved['cwd']}", file=sys.stderr)
            if not await _prompt_confirm("Fork this session into current directory?"):
                print("Aborted.")
                return None
            return SessionManager.fork_from(resolved["path"], cwd, parsed.session_dir)
        print(f"No session found matching '{resolved['arg']}'", file=sys.stderr)
        return None
    if parsed.continue_:
        return SessionManager.continue_recent(cwd, parsed.session_dir)
    if parsed.session_dir:
        return SessionManager.create(cwd, parsed.session_dir)
    return None


def _report_settings_errors(settings_manager: SettingsManager, context: str) -> None:
    for item in settings_manager.drain_errors():
        scope = item.get("scope", "unknown")
        error = item.get("error")
        message = str(error) if error else "Unknown settings error"
        print(f"Warning ({context}, {scope} settings): {message}", file=sys.stderr)


def _create_session_options(
    parsed: Any,
    *,
    cwd: str,
    session_manager: SessionManager | None,
    auth_storage: AuthStorage,
    model_registry: ModelRegistry,
    settings_manager: SettingsManager,
) -> CreateAgentSessionOptions:
    """Build SDK options. Leave thinking_level unset so settings/DEFAULT apply."""
    model = None
    if parsed.model or parsed.provider:
        try:
            model = model_registry.resolve_model(model_id=parsed.model, provider=parsed.provider)
        except Exception:
            pass
    return CreateAgentSessionOptions(
        cwd=cwd,
        agent_dir=get_agent_dir(),
        model=model,
        thinking_level=parsed.thinking,
        session_manager=session_manager,
        auth_storage=auth_storage,
        model_registry=model_registry,
        settings_manager=settings_manager,
    )


def _parse_package_command(args: Sequence[str]) -> dict[str, Any] | None:
    parsed = parse_package_command(list(args))
    if not parsed:
        return None
    return {
        "command": parsed["command"],
        "source": parsed.get("source"),
        "local": parsed.get("local", False),
        "help": parsed.get("help", False),
        "invalid_option": parsed.get("invalid_option"),
    }


def _package_usage(command: str) -> str:
    return get_package_command_usage(command)  # type: ignore[arg-type]


def _print_package_help(command: str) -> None:
    print_package_command_help(command)  # type: ignore[arg-type]


async def _handle_package_command(args: Sequence[str]) -> tuple[bool, int]:
    return await run_package_command(list(args))


async def _handle_config_command(args: Sequence[str]) -> tuple[bool, int]:
    return await run_config_command(list(args))


async def _run(args: Sequence[str]) -> int:
    # Load workspace environment variables early so model/api-key resolution
    # can see keys from .env (e.g. GEMINI_API_KEY).
    _load_env_files(os.getcwd())

    handled, exit_code = await _handle_package_command(args)
    if handled:
        return exit_code
    handled, exit_code = await _handle_config_command(args)
    if handled:
        return exit_code

    migration_result = run_migrations(os.getcwd())
    migrated_auth_providers = migration_result.get("migratedAuthProviders", [])
    deprecation_warnings = migration_result.get("deprecationWarnings", [])

    first_pass = parse_args(list(args))
    event_bus = create_event_bus()
    ext_paths = first_pass.extensions or []
    extensions_result = await load_extensions(ext_paths, os.getcwd(), event_bus)
    extension_flags: dict[str, str] = {}
    for ext in extensions_result.extensions:
        for name, flag in ext.flags.items():
            extension_flags[name] = "string" if flag.type == "string" else "boolean"

    parsed = parse_args(list(args), extension_flags=extension_flags)

    if parsed.version:
        from .config import VERSION

        print(VERSION)
        return 0

    if parsed.help:
        print_help()
        return 0

    # Read piped stdin for non-rpc mode
    if parsed.mode != "rpc":
        stdin_content = await _read_piped_stdin()
        if stdin_content is not None:
            parsed.print_mode = True
            parsed.messages.insert(0, stdin_content)

    if parsed.export:
        from .core.export_html import export_from_file

        output_path = parsed.messages[0] if parsed.messages else None
        exported = await export_from_file(parsed.export, output_path=output_path)
        print(f"Exported to: {exported}")
        return 0

    if parsed.mode == "rpc" and parsed.file_args:
        print("Error: @file arguments are not supported in RPC mode", file=sys.stderr)
        return 1

    cwd = os.getcwd()
    settings_manager = SettingsManager.create(cwd, get_agent_dir())
    _report_settings_errors(settings_manager, "startup")
    auth_storage = AuthStorage()
    model_registry = ModelRegistry()
    session_manager = await _create_session_manager(parsed, cwd)
    if parsed.session and session_manager is None:
        return 1

    opts = _create_session_options(
        parsed,
        cwd=cwd,
        session_manager=session_manager,
        auth_storage=auth_storage,
        model_registry=model_registry,
        settings_manager=settings_manager,
    )
    result = await create_agent_session(opts)
    session = result.session

    if parsed.list_models is not None:
        pattern = parsed.list_models if isinstance(parsed.list_models, str) else None
        await list_models(model_registry, pattern)
        return 0

    if parsed.mode == "rpc":
        await run_rpc_mode(session)
        return 0

    initial_prompt, images = await _build_initial_prompt(parsed)

    # Print mode (explicit) or JSON mode
    if parsed.print_mode or parsed.mode in ("text", "json"):
        prompt = initial_prompt
        if not prompt and parsed.messages:
            prompt = parsed.messages[0]
        if not prompt:
            print("No prompt provided. Use --help for usage.", file=sys.stderr)
            return 1
        return await run_print_mode(
            session,
            prompt,
            show_thinking=bool(parsed.verbose),
            json_output=parsed.mode == "json",
        )

    # Default interactive mode
    if deprecation_warnings:
        await show_deprecation_warnings(deprecation_warnings)

    if migrated_auth_providers and parsed.verbose:
        print(f"Migrated auth providers: {', '.join(migrated_auth_providers)}", file=sys.stderr)

    # --resume: interactive session picker
    if parsed.resume:
        selected = await select_session(
            lambda: SessionManager.list(cwd, parsed.session_dir),
            SessionManager.list_all,
        )
        if not selected:
            print("No session selected")
            return 0
        sm = SessionManager.open(selected, parsed.session_dir)
        result = await create_agent_session(
            _create_session_options(
                parsed,
                cwd=cwd,
                session_manager=sm,
                auth_storage=auth_storage,
                model_registry=model_registry,
                settings_manager=settings_manager,
            )
        )
        session = result.session

    initial_messages = []
    if initial_prompt:
        initial_messages.append(initial_prompt)
    initial_messages.extend(parsed.messages[1:] if parsed.messages else [])
    await run_interactive_mode(session, initial_messages=initial_messages or None)
    return 0


def main(args: Sequence[str] | None = None) -> None:
    """CLI entrypoint used by project script."""
    exit_code = asyncio.run(_run(args if args is not None else sys.argv[1:]))
    sys.exit(exit_code)
