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
from .config import APP_NAME, get_agent_dir
from .core.auth_storage import AuthStorage
from .core.event_bus import create_event_bus
from .core.extensions.loader import load_extensions
from .core.model_registry import ModelRegistry
from .core.package_manager import DefaultPackageManager
from .core.sdk import AgentSessionOptions, create_agent_session
from .core.session_manager import SessionManager
from .core.settings_manager import SettingsManager
from .migrations import run_migrations, show_deprecation_warnings
from .modes import run_interactive_mode, run_print_mode, run_rpc_mode


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


def _parse_package_command(args: Sequence[str]) -> dict[str, Any] | None:
    if not args:
        return None
    command = args[0]
    if command not in {"install", "remove", "update", "list"}:
        return None

    source: str | None = None
    local = False
    help_requested = False
    invalid_option: str | None = None
    for arg in args[1:]:
        if arg in {"-h", "--help"}:
            help_requested = True
            continue
        if arg in {"-l", "--local"}:
            if command in {"install", "remove"}:
                local = True
            else:
                invalid_option = invalid_option or arg
            continue
        if arg.startswith("-"):
            invalid_option = invalid_option or arg
            continue
        if source is None:
            source = arg

    return {
        "command": command,
        "source": source,
        "local": local,
        "help": help_requested,
        "invalid_option": invalid_option,
    }


def _package_usage(command: str) -> str:
    if command == "install":
        return f"{APP_NAME} install <source> [-l]"
    if command == "remove":
        return f"{APP_NAME} remove <source> [-l]"
    if command == "update":
        return f"{APP_NAME} update [source]"
    return f"{APP_NAME} list"


def _print_package_help(command: str) -> None:
    usage = _package_usage(command)
    if command == "install":
        print(f"Usage:\n  {usage}\n\nInstall a package and add it to settings.\n")
    elif command == "remove":
        print(f"Usage:\n  {usage}\n\nRemove a package and its source from settings.\n")
    elif command == "update":
        print(f"Usage:\n  {usage}\n\nUpdate installed packages.\n")
    else:
        print(f"Usage:\n  {usage}\n\nList installed packages from user and project settings.\n")


async def _handle_package_command(args: Sequence[str]) -> tuple[bool, int]:
    parsed = _parse_package_command(args)
    if not parsed:
        return False, 0

    command = parsed["command"]
    source = parsed["source"]
    local = parsed["local"]

    if parsed["help"]:
        _print_package_help(command)
        return True, 0
    if parsed["invalid_option"]:
        print(f'Unknown option {parsed["invalid_option"]} for "{command}".', file=sys.stderr)
        print(f'Use "{APP_NAME} --help" or "{_package_usage(command)}".', file=sys.stderr)
        return True, 1
    if command in {"install", "remove"} and not source:
        print(f"Missing {command} source.", file=sys.stderr)
        print(f"Usage: {_package_usage(command)}", file=sys.stderr)
        return True, 1

    cwd = os.getcwd()
    settings_manager = SettingsManager.create(cwd, get_agent_dir())
    _report_settings_errors(settings_manager, "package command")
    package_manager = DefaultPackageManager(cwd=cwd, agent_dir=get_agent_dir(), settings_manager=settings_manager)
    package_manager.set_progress_callback(
        lambda event: print(event.message, file=sys.stderr) if event.type == "start" and event.message else None
    )

    try:
        if command == "install":
            await package_manager.install(source, {"local": local})
            package_manager.add_source_to_settings(source, {"local": local})
            print(f"Installed {source}")
            return True, 0
        if command == "remove":
            await package_manager.remove(source, {"local": local})
            removed = package_manager.remove_source_from_settings(source, {"local": local})
            if not removed:
                print(f"No matching package found for {source}", file=sys.stderr)
                return True, 1
            print(f"Removed {source}")
            return True, 0
        if command == "list":
            global_settings = settings_manager.get_global_settings()
            project_settings = settings_manager.get_project_settings()
            global_packages = global_settings.get("packages", []) or []
            project_packages = project_settings.get("packages", []) or []
            if not global_packages and not project_packages:
                print("No packages installed.")
                return True, 0

            def _format_pkg(pkg: Any, scope: str) -> None:
                source_str = pkg if isinstance(pkg, str) else pkg.get("source", "")
                filtered = isinstance(pkg, dict)
                display = f"{source_str} (filtered)" if filtered else source_str
                print(f"  {display}")
                installed = package_manager.get_installed_path(source_str, "user" if scope == "user" else "project")
                if installed:
                    print(f"    {installed}")

            if global_packages:
                print("User packages:")
                for pkg in global_packages:
                    _format_pkg(pkg, "user")
            if project_packages:
                if global_packages:
                    print()
                print("Project packages:")
                for pkg in project_packages:
                    _format_pkg(pkg, "project")
            return True, 0

        await package_manager.update(source)
        if source:
            print(f"Updated {source}")
        else:
            print("Updated packages")
        return True, 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return True, 1


async def _handle_config_command(args: Sequence[str]) -> tuple[bool, int]:
    if not args or args[0] != "config":
        return False, 0
    cwd = os.getcwd()
    settings_manager = SettingsManager.create(cwd, get_agent_dir())
    _report_settings_errors(settings_manager, "config command")
    package_manager = DefaultPackageManager(cwd=cwd, agent_dir=get_agent_dir(), settings_manager=settings_manager)
    resolved = await package_manager.resolve()

    print("Resolved package resources:")
    for label, entries in (
        ("extensions", resolved.extensions),
        ("skills", resolved.skills),
        ("prompts", resolved.prompts),
        ("themes", resolved.themes),
    ):
        print(f"- {label}: {len(entries)}")
    return True, 0


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

    opts = AgentSessionOptions(
        cwd=cwd,
        model_id=parsed.model,
        provider=parsed.provider,
        api_key=parsed.api_key,
        thinking_level=parsed.thinking or (settings_manager.get_default_thinking_level() or "off"),
        auto_compact=not parsed.no_session,
        sessions_dir=parsed.session_dir,
        session_manager=session_manager,
        auth_storage=auth_storage,
        model_registry=model_registry,
    )
    session = create_agent_session(opts)

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
        session = create_agent_session(
            AgentSessionOptions(
                cwd=cwd,
                model_id=parsed.model,
                provider=parsed.provider,
                api_key=parsed.api_key,
                thinking_level=parsed.thinking or "off",
                auto_compact=not parsed.no_session,
                sessions_dir=parsed.session_dir,
                session_manager=sm,
                auth_storage=auth_storage,
                model_registry=model_registry,
            )
        )

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
