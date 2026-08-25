"""
Utilities for pi_coding_agent.
"""

from .abort import AbortError, operation_signal, race_with_abort_signal
from .ansi import strip_ansi
from .changelog import ChangelogEntry, compare_versions, get_new_entries, parse_changelog
from .deprecation import clear_deprecation_warnings_for_tests, warn_deprecation
from .frontmatter import parse_frontmatter, stringify_frontmatter, strip_frontmatter
from .fs_watch import FS_WATCH_RETRY_DELAY_MS, close_watcher, watch_with_error_handler
from .git import GitSource, parse_git_url
from .html import DecodedHtmlEntity, decode_html_entity, decode_html_entity_at
from .mime import detect_supported_image_mime_type, detect_supported_image_mime_type_from_file
from .open_browser import open_browser
from .paths import canonicalize_path, normalize_path, resolve_path
from .shell import get_shell_config, get_shell_env, kill_process_tree, sanitize_binary_output
from .sleep import sleep
from .text import load_json_file, loads_json, split_bom, strip_bom
from .tool_result_images import normalize_tool_result_images
from .tools_manager import ToolConfig, ensure_tool, get_tool_path

__all__ = [
    "AbortError",
    "ChangelogEntry",
    "DecodedHtmlEntity",
    "FS_WATCH_RETRY_DELAY_MS",
    "GitSource",
    "ToolConfig",
    "canonicalize_path",
    "clear_deprecation_warnings_for_tests",
    "close_watcher",
    "compare_versions",
    "decode_html_entity",
    "decode_html_entity_at",
    "detect_supported_image_mime_type",
    "detect_supported_image_mime_type_from_file",
    "ensure_tool",
    "get_new_entries",
    "get_shell_config",
    "get_shell_env",
    "get_tool_path",
    "kill_process_tree",
    "load_json_file",
    "loads_json",
    "normalize_path",
    "normalize_tool_result_images",
    "open_browser",
    "operation_signal",
    "parse_changelog",
    "parse_frontmatter",
    "parse_git_url",
    "race_with_abort_signal",
    "resolve_path",
    "sanitize_binary_output",
    "sleep",
    "split_bom",
    "stringify_frontmatter",
    "strip_ansi",
    "strip_bom",
    "strip_frontmatter",
    "warn_deprecation",
    "watch_with_error_handler",
]
