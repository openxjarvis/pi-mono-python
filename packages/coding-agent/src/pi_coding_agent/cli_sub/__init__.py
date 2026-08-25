"""CLI helper subpackage — mirrors packages/coding-agent/src/cli/ in the TypeScript source."""
from .args import Args, Mode, parse_args, print_help
from .auth_command import AuthCommand, AuthCommandError, parse_auth_command, print_auth_command_help
from .file_processor import ProcessedFiles, ProcessFileOptions, process_file_arguments
from .list_models import list_models
from .session_picker import select_session
from .startup_ui import create_startup_tui

__all__ = [
    "Args",
    "Mode",
    "ProcessFileOptions",
    "ProcessedFiles",
    "parse_args",
    "print_help",
    "process_file_arguments",
    "list_models",
    "select_session",
    "AuthCommand",
    "AuthCommandError",
    "parse_auth_command",
    "print_auth_command_help",
    "create_startup_tui",
]
