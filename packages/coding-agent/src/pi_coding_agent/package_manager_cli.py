"""Package/config CLI. Mirrors packages/coding-agent/src/package-manager-cli.ts.

Node managed-install and Windows native self-update are intentionally omitted.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Literal, TypedDict

from pi_coding_agent.cli_sub.config_selector import ConfigSelectorOptions, select_config
from pi_coding_agent.config import APP_NAME, CONFIG_DIR_NAME, get_agent_dir
from pi_coding_agent.core.package_manager import DefaultPackageManager
from pi_coding_agent.core.project_trust import AppMode, ResolveProjectTrustedOptions, resolve_project_trusted
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.core.trust_manager import ProjectTrustStore, has_trust_requiring_project_resources

PackageCommand = Literal["install", "remove", "update", "list"]
UpdateTarget = dict[str, Any]


class PackageCommandOptions(TypedDict, total=False):
    command: PackageCommand
    source: str | None
    update_target: UpdateTarget | None
    show_extensions_skipped_note: bool
    local: bool
    force: bool
    project_trust_override: bool | None
    help: bool
    invalid_option: str | None
    invalid_argument: str | None
    missing_option_value: str | None
    conflicting_options: str | None


class PackageCommandRuntimeOptions(TypedDict, total=False):
    extension_factories: list[Any]
    extensionFactories: list[Any]


CONFIG_COMMAND_USAGE = f"{APP_NAME} config [-l] [--approve|--no-approve]"


def get_package_command_usage(command: PackageCommand) -> str:
    if command == "install":
        return f"{APP_NAME} install <source> [-l] [--approve|--no-approve]"
    if command == "remove":
        return f"{APP_NAME} remove <source> [-l] [--approve|--no-approve]"
    if command == "update":
        return (
            f"{APP_NAME} update [source|self|pi] [--self|--extensions|--models|--all] "
            "[--extension <source>] [--approve|--no-approve] [--force]"
        )
    return f"{APP_NAME} list [--approve|--no-approve]"


def print_config_command_help() -> None:
    print(
        f"""Usage:
  {CONFIG_COMMAND_USAGE}

Open the resource configuration TUI to enable or disable package resources.
Without -l, starts in global settings (~/{CONFIG_DIR_NAME}/agent/settings.json).
Press Tab in the TUI to switch between global and project-local modes.

Options:
  -l, --local       Edit project overrides ({CONFIG_DIR_NAME}/settings.json)
  -a, --approve     Trust project-local files for this command with -l
  -na, --no-approve Ignore project-local files for this command with -l
"""
    )


def print_package_command_help(command: PackageCommand) -> None:
    usage = get_package_command_usage(command)
    if command == "install":
        print(
            f"""Usage:
  {usage}

Install a package and add it to settings.

Options:
  -l, --local       Install project-locally ({CONFIG_DIR_NAME}/settings.json)
  -a, --approve     Trust project-local files for this command
  -na, --no-approve Ignore project-local files for this command

Examples:
  {APP_NAME} install npm:@foo/bar
  {APP_NAME} install git:github.com/user/repo
  {APP_NAME} install ./local/path
"""
        )
        return
    if command == "remove":
        print(
            f"""Usage:
  {usage}

Remove a package and its source from settings.
Alias: {APP_NAME} uninstall <source> [-l]

Options:
  -l, --local       Remove from project settings ({CONFIG_DIR_NAME}/settings.json)
  -a, --approve     Trust project-local files for this command
  -na, --no-approve Ignore project-local files for this command
"""
        )
        return
    if command == "update":
        print(
            f"""Usage:
  {usage}

Update pi, installed packages, or model catalogs.

Options:
  --self                  Update pi only (default when no target is given)
  --extensions            Update installed packages only
  --models                Refresh model catalogs only
  --all                   Update pi and installed packages
  --extension <source>    Update one package only
  -a, --approve           Trust project-local files for this command
  -na, --no-approve       Ignore project-local files for this command
  --force                 Reinstall pi even if the current version is latest
"""
        )
        return
    print(
        f"""Usage:
  {usage}

List installed packages from user and project settings.

Options:
  -a, --approve      Trust project-local files for this command
  -na, --no-approve  Ignore project-local files for this command
"""
    )


def parse_package_command(args: list[str]) -> PackageCommandOptions | None:
    if not args:
        return None
    raw_command, *rest = args
    command: PackageCommand | None
    if raw_command == "uninstall":
        command = "remove"
    elif raw_command in {"install", "remove", "update", "list"}:
        command = raw_command  # type: ignore[assignment]
    else:
        command = None
    if command is None:
        return None

    local = False
    force = False
    project_trust_override: bool | None = None
    help_requested = False
    invalid_option: str | None = None
    invalid_argument: str | None = None
    missing_option_value: str | None = None
    conflicting_options: str | None = None
    source: str | None = None
    self_flag = False
    extensions_flag = False
    models_flag = False
    all_flag = False
    extension_flag_source: str | None = None

    index = 0
    while index < len(rest):
        arg = rest[index]
        if arg in {"-h", "--help"}:
            help_requested = True
            index += 1
            continue
        if arg in {"-l", "--local"}:
            if command in {"install", "remove"}:
                local = True
            else:
                invalid_option = invalid_option or arg
            index += 1
            continue
        if arg == "--self":
            if command == "update":
                self_flag = True
            else:
                invalid_option = invalid_option or arg
            index += 1
            continue
        if arg == "--extensions":
            if command == "update":
                extensions_flag = True
            else:
                invalid_option = invalid_option or arg
            index += 1
            continue
        if arg == "--models":
            if command == "update":
                models_flag = True
            else:
                invalid_option = invalid_option or arg
            index += 1
            continue
        if arg == "--all":
            if command == "update":
                all_flag = True
            else:
                invalid_option = invalid_option or arg
            index += 1
            continue
        if arg in {"--approve", "-a"}:
            project_trust_override = True
            index += 1
            continue
        if arg in {"--no-approve", "-na"}:
            project_trust_override = False
            index += 1
            continue
        if arg == "--force":
            if command == "update":
                force = True
            else:
                invalid_option = invalid_option or arg
            index += 1
            continue
        if arg == "--extension":
            if command != "update":
                invalid_option = invalid_option or arg
                index += 1
                continue
            value = rest[index + 1] if index + 1 < len(rest) else None
            if not value or value.startswith("-"):
                missing_option_value = missing_option_value or arg
            elif extension_flag_source:
                conflicting_options = conflicting_options or "--extension can only be provided once"
                index += 1
            else:
                extension_flag_source = value
                index += 1
            index += 1
            continue
        if arg.startswith("-"):
            invalid_option = invalid_option or arg
            index += 1
            continue
        if source is None:
            source = arg
        else:
            invalid_argument = invalid_argument or arg
        index += 1

    update_target: UpdateTarget | None = None
    show_extensions_skipped_note = False
    if command == "update":
        if all_flag and (self_flag or extensions_flag or models_flag or extension_flag_source):
            conflicting_options = (
                conflicting_options or "--all cannot be combined with --self, --extensions, --models, or --extension"
            )
        if all_flag and source:
            conflicting_options = conflicting_options or "--all cannot be combined with a positional source"
        if models_flag:
            if self_flag or extensions_flag or all_flag or extension_flag_source:
                conflicting_options = (
                    conflicting_options
                    or "--models cannot be combined with --self, --extensions, --all, or --extension"
                )
            if source:
                conflicting_options = conflicting_options or "--models cannot be combined with a positional source"
            update_target = {"type": "models"}
        elif extension_flag_source:
            if self_flag or extensions_flag or all_flag:
                conflicting_options = (
                    conflicting_options or "--extension cannot be combined with --self, --extensions, or --all"
                )
            if source:
                conflicting_options = conflicting_options or "--extension cannot be combined with a positional source"
            update_target = {"type": "extensions", "source": extension_flag_source}
        elif source:
            source_is_self = source in {"self", "pi"}
            if source_is_self:
                update_target = {"type": "all"} if extensions_flag else {"type": "self"}
            else:
                if extensions_flag or self_flag or all_flag:
                    conflicting_options = (
                        conflicting_options
                        or "positional update targets cannot be combined with --self, --extensions, or --all"
                    )
                update_target = {"type": "extensions", "source": source}
        elif all_flag:
            update_target = {"type": "all"}
        elif self_flag and extensions_flag:
            update_target = {"type": "all"}
        elif self_flag:
            update_target = {"type": "self"}
        elif extensions_flag:
            update_target = {"type": "extensions"}
        else:
            update_target = {"type": "self"}
            show_extensions_skipped_note = True

    return {
        "command": command,
        "source": source,
        "update_target": update_target,
        "show_extensions_skipped_note": show_extensions_skipped_note,
        "local": local,
        "force": force,
        "project_trust_override": project_trust_override,
        "help": help_requested,
        "invalid_option": invalid_option,
        "invalid_argument": invalid_argument,
        "missing_option_value": missing_option_value,
        "conflicting_options": conflicting_options,
    }


def update_target_includes_self(target: UpdateTarget) -> bool:
    return target.get("type") in {"all", "self"}


def update_target_includes_extensions(target: UpdateTarget) -> bool:
    return target.get("type") in {"all", "extensions"}


def cleanup_managed_install() -> None:
    """No-op. TS managed-install is a Node installer layout."""
    return


def get_command_app_mode() -> AppMode:
    return "interactive" if sys.stdin.isatty() and sys.stdout.isatty() else "print"


def report_project_trust_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)


def report_settings_errors(settings_manager: Any, context: str) -> None:
    drain = getattr(settings_manager, "drain_errors", None)
    if not callable(drain):
        return
    for item in drain() or []:
        scope = item.get("scope", "unknown") if isinstance(item, dict) else "unknown"
        error = item.get("error") if isinstance(item, dict) else item
        print(f"Warning ({context}, {scope} settings): {error}", file=sys.stderr)


class _SilentTrustContext:
    has_ui = False

    class _UI:
        async def select(self, title: str, options: list[str]) -> str | None:
            return None

    ui = _UI()


async def create_command_settings_manager(
    cwd: str,
    agent_dir: str,
    project_trust_override: bool | None = None,
    use_saved_project_trust_only: bool = False,
    extension_factories: list[Any] | None = None,
) -> dict[str, Any]:
    settings_manager = SettingsManager.create(cwd, agent_dir)
    project_trust_warnings: list[str] = []
    trust_store = ProjectTrustStore(agent_dir)
    if use_saved_project_trust_only:
        saved = trust_store.get(cwd) is True
        trusted = saved if project_trust_override is None else project_trust_override
        settings_manager.set_project_trusted(trusted)
        return {"settingsManager": settings_manager, "settings_manager": settings_manager, "projectTrustWarnings": project_trust_warnings}

    if project_trust_override is not None:
        settings_manager.set_project_trusted(project_trust_override)
        return {"settingsManager": settings_manager, "settings_manager": settings_manager, "projectTrustWarnings": project_trust_warnings}

    if not has_trust_requiring_project_resources(cwd):
        settings_manager.set_project_trusted(True)
        return {"settingsManager": settings_manager, "settings_manager": settings_manager, "projectTrustWarnings": project_trust_warnings}

    trusted = await resolve_project_trusted(
        ResolveProjectTrustedOptions(
            cwd=cwd,
            trust_store=trust_store,
            project_trust_context=_SilentTrustContext(),
            trust_override=project_trust_override,
            default_project_trust="ask",
            extensions_result=None,
            on_extension_error=project_trust_warnings.append,
        )
    )
    settings_manager.set_project_trusted(trusted)
    _ = extension_factories
    return {"settingsManager": settings_manager, "settings_manager": settings_manager, "projectTrustWarnings": project_trust_warnings}


async def handle_config_command(
    args: list[str],
    runtime_options: PackageCommandRuntimeOptions | None = None,
) -> bool:
    handled, _code = await run_config_command(args, runtime_options)
    return handled


async def handle_package_command(
    args: list[str],
    runtime_options: PackageCommandRuntimeOptions | None = None,
) -> bool:
    handled, _code = await run_package_command(args, runtime_options)
    return handled


async def run_config_command(
    args: list[str],
    runtime_options: PackageCommandRuntimeOptions | None = None,
) -> tuple[bool, int]:
    if not args or args[0] != "config":
        return False, 0
    rest = list(args[1:])
    if "-h" in rest or "--help" in rest:
        print_config_command_help()
        return True, 0

    local = False
    project_trust_override: bool | None = None
    for arg in rest:
        if arg in {"-l", "--local"}:
            local = True
        elif arg in {"-a", "--approve"}:
            project_trust_override = True
        elif arg in {"-na", "--no-approve"}:
            project_trust_override = False
        elif arg.startswith("-"):
            print(f'Unknown option {arg} for "config".', file=sys.stderr)
            print(f'Use "{APP_NAME} --help" or "{CONFIG_COMMAND_USAGE}".', file=sys.stderr)
            return True, 1
        else:
            print(f"Unexpected argument {arg}.", file=sys.stderr)
            print(f"Usage: {CONFIG_COMMAND_USAGE}", file=sys.stderr)
            return True, 1

    cwd = os.getcwd()
    agent_dir = get_agent_dir()
    runtime_options = runtime_options or {}
    created = await create_command_settings_manager(
        cwd,
        agent_dir,
        project_trust_override=project_trust_override,
        extension_factories=runtime_options.get("extension_factories") or runtime_options.get("extensionFactories"),
    )
    settings_manager = created["settings_manager"]
    report_project_trust_warnings(created["projectTrustWarnings"])
    if local and not settings_manager.is_project_trusted():
        print("Project is not trusted. Use --approve to modify local resource config.", file=sys.stderr)
        return True, 1
    report_settings_errors(settings_manager, "config command")
    global_settings_manager = SettingsManager.create(cwd, agent_dir)
    global_settings_manager.set_project_trusted(False)
    global_resolved = await DefaultPackageManager(
        cwd=cwd, agent_dir=agent_dir, settings_manager=global_settings_manager
    ).resolve()
    project_resolved = (
        await DefaultPackageManager(cwd=cwd, agent_dir=agent_dir, settings_manager=settings_manager).resolve()
        if settings_manager.is_project_trusted()
        else global_resolved
    )
    await select_config(
        ConfigSelectorOptions(
            resolved_paths=project_resolved if local else global_resolved,
            settings_manager=settings_manager,
            cwd=cwd,
            agent_dir=agent_dir,
        )
    )
    return True, 0


async def run_package_command(
    args: list[str],
    runtime_options: PackageCommandRuntimeOptions | None = None,
) -> tuple[bool, int]:
    options = parse_package_command(list(args))
    if not options:
        return False, 0

    command = options["command"]
    if options.get("help"):
        print_package_command_help(command)
        return True, 0
    if options.get("invalid_option"):
        print(f'Unknown option {options["invalid_option"]} for "{command}".', file=sys.stderr)
        print(f'Use "{APP_NAME} --help" or "{get_package_command_usage(command)}".', file=sys.stderr)
        return True, 1
    if options.get("missing_option_value"):
        print(f'Missing value for {options["missing_option_value"]}.', file=sys.stderr)
        print(f"Usage: {get_package_command_usage(command)}", file=sys.stderr)
        return True, 1
    if options.get("invalid_argument"):
        print(f'Unexpected argument {options["invalid_argument"]}.', file=sys.stderr)
        print(f"Usage: {get_package_command_usage(command)}", file=sys.stderr)
        return True, 1
    if options.get("conflicting_options"):
        print(options["conflicting_options"], file=sys.stderr)
        print(f"Usage: {get_package_command_usage(command)}", file=sys.stderr)
        return True, 1

    source = options.get("source")
    if command in {"install", "remove"} and not source:
        print(f"Missing {command} source.", file=sys.stderr)
        print(f"Usage: {get_package_command_usage(command)}", file=sys.stderr)
        return True, 1

    if command == "update" and (options.get("update_target") or {}).get("type") == "models":
        try:
            await _refresh_model_catalogs(get_agent_dir())
        except Exception as error:
            print(f"Error: {error}", file=sys.stderr)
            return True, 1
        return True, 0

    cwd = os.getcwd()
    agent_dir = get_agent_dir()
    writes_project = command in {"install", "remove"} and bool(options.get("local"))
    runtime_options = runtime_options or {}
    created = await create_command_settings_manager(
        cwd,
        agent_dir,
        project_trust_override=options.get("project_trust_override"),
        use_saved_project_trust_only=command == "update",
        extension_factories=runtime_options.get("extension_factories") or runtime_options.get("extensionFactories"),
    )
    settings_manager = created["settings_manager"]
    report_project_trust_warnings(created["projectTrustWarnings"])
    if not settings_manager.is_project_trusted() and writes_project:
        print("Project is not trusted. Use --approve to modify local package config.", file=sys.stderr)
        return True, 1
    report_settings_errors(settings_manager, "package command")

    package_manager = DefaultPackageManager(cwd=cwd, agent_dir=agent_dir, settings_manager=settings_manager)
    package_manager.set_progress_callback(
        lambda event: print(event.message or "", file=sys.stdout) if event.type == "start" and event.message else None
    )

    try:
        if command == "install":
            await package_manager.install_and_persist(source or "", {"local": options.get("local")})
            print(f"Installed {source}")
            return True, 0
        if command == "remove":
            removed = await package_manager.remove_and_persist(source or "", {"local": options.get("local")})
            if not removed:
                print(f"No matching package found for {source}", file=sys.stderr)
                return True, 1
            print(f"Removed {source}")
            return True, 0
        if command == "list":
            configured = package_manager.list_configured_packages()
            user_packages = [pkg for pkg in configured if pkg["scope"] == "user"]
            project_packages = [pkg for pkg in configured if pkg["scope"] == "project"]
            if not configured:
                print("No packages installed.")
                return True, 0

            def format_package(pkg: dict[str, Any]) -> None:
                display = f"{pkg['source']} (filtered)" if pkg.get("filtered") else pkg["source"]
                print(f"  {display}")
                installed = pkg.get("installedPath") or pkg.get("installed_path")
                if installed:
                    print(f"    {installed}")

            if user_packages:
                print("User packages:")
                for pkg in user_packages:
                    format_package(pkg)
            if project_packages:
                if user_packages:
                    print()
                print("Project packages:")
                for pkg in project_packages:
                    format_package(pkg)
            return True, 0

        target = options.get("update_target") or {"type": "self"}
        if options.get("show_extensions_skipped_note"):
            print(f"Extensions are skipped. Run {APP_NAME} update --extensions to update extensions.")
        if update_target_includes_extensions(target):
            update_source = target.get("source") if target.get("type") == "extensions" else None
            await package_manager.update(update_source)
            if update_source:
                print(f"Updated {update_source}")
            else:
                print("Updated packages")
        if update_target_includes_self(target):
            print(
                f"{APP_NAME} Python self-update is not available. Update the package with pip/uv.",
                file=sys.stderr,
            )
            return True, 1
        return True, 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return True, 1


async def _refresh_model_catalogs(agent_dir: str) -> None:
    from pi_coding_agent.core.model_runtime import CreateModelRuntimeOptions, ModelRuntime

    runtime = await ModelRuntime.create(
        CreateModelRuntimeOptions(
            auth_path=os.path.join(agent_dir, "auth.json"),
            models_path=os.path.join(agent_dir, "models.json"),
        )
    )
    result = await runtime.refresh(allow_network=True)
    errors = result.get("errors") if isinstance(result, dict) else None
    if errors:
        raise RuntimeError(f"Could not refresh model catalogs: {errors}")
