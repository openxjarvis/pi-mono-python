"""
Auth CLI commands — mirrors packages/coding-agent/src/cli/auth-command.ts
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pi_coding_agent.config import APP_NAME

AuthCommandKind = Literal["check", "api_key", "bearer_token"]


@dataclass
class AuthCommand:
    kind: AuthCommandKind
    args: list[str] = field(default_factory=list)
    json: bool = False
    credentials: bool = False
    no_refresh: bool = False
    min_expiry_ms: int | None = None


class AuthCommandError(Exception):
    pass


AUTH_COMMAND_USAGE = {
    "check": f"{APP_NAME} auth check --provider <provider> [--json] [--credentials] [--no-refresh]",
    "api_key": f"{APP_NAME} auth print-api-key --provider <provider> [--model <model>]",
    "bearer_token": f"{APP_NAME} auth print-bearer-token --provider <provider> [--model <model>] [--min-expiry <duration>]",
}


def get_auth_command_name(kind: AuthCommandKind) -> str:
    if kind == "check":
        return "auth check"
    if kind == "api_key":
        return "auth print-api-key"
    return "auth print-bearer-token"


def get_auth_command_usage(kind: AuthCommandKind) -> str:
    return AUTH_COMMAND_USAGE[kind]


def is_auth_command_help(args: list[str]) -> bool:
    return (
        bool(args)
        and args[0] == "auth"
        and (len(args) < 2 or args[1] in {"help", "--help", "-h"} or "--help" in args or "-h" in args)
    )


def print_auth_command_help() -> None:
    print(
        """Usage:
  pi auth print-api-key [--provider <provider>] [--model <model>]
  pi auth print-bearer-token [--provider <provider>] [--model <model>] [--min-expiry <duration>]
  pi auth check [--provider <provider>] [--model <model>] [--json] [--credentials] [--no-refresh]

Auth commands require at least one of --provider or --model. Checks refresh expired OAuth credentials by default; --no-refresh prevents this. --credentials emits the credential, or includes it in JSON output."""
    )


def parse_auth_command(args: list[str]) -> AuthCommand | None:
    if not args or args[0] != "auth":
        return None
    kind: AuthCommandKind | None = None
    if len(args) > 1:
        if args[1] == "check":
            kind = "check"
        elif args[1] == "print-api-key":
            kind = "api_key"
        elif args[1] == "print-bearer-token":
            kind = "bearer_token"
    if kind is None:
        raise AuthCommandError(
            f'Unknown auth command "{args[1] if len(args) > 1 else ""}". '
            f'Use "{APP_NAME} auth print-api-key", "{APP_NAME} auth print-bearer-token", or "{APP_NAME} auth check".'
        )

    command_args: list[str] = []
    json_out = False
    credentials = False
    no_refresh = False
    min_expiry_ms: int | None = None
    index = 2
    while index < len(args):
        arg = args[index]
        if arg == "--min-expiry":
            if kind != "bearer_token":
                raise AuthCommandError("--min-expiry is only supported by print-bearer-token")
            index += 1
            value = args[index] if index < len(args) else None
            match = None
            if value:
                import re

                match = re.fullmatch(r"(\d+)(ms|s|m|h)", value, re.IGNORECASE)
            if not match:
                raise AuthCommandError("--min-expiry must use a duration such as 30m or 1h")
            amount = int(match.group(1))
            unit = match.group(2).lower()
            min_expiry_ms = amount * (1 if unit == "ms" else 1000 if unit == "s" else 60_000 if unit == "m" else 3_600_000)
        elif arg == "--json":
            json_out = True
        elif arg == "--credentials":
            credentials = True
        elif arg == "--no-refresh":
            no_refresh = True
        else:
            command_args.append(arg)
        index += 1
    return AuthCommand(
        kind=kind,
        args=command_args,
        json=json_out,
        credentials=credentials,
        no_refresh=no_refresh,
        min_expiry_ms=min_expiry_ms,
    )
