"""
Shell configuration and process utilities.

Mirrors utils/shell.ts
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys

_cached_shell_config: tuple[str, list[str]] | None = None


def _find_bash_on_path() -> str | None:
    """Find bash executable on PATH (cross-platform)."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["where", "bash.exe"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                first = result.stdout.strip().splitlines()[0]
                if first and os.path.exists(first):
                    return first
        except Exception:
            pass
        return None
    else:
        bash = shutil.which("bash")
        return bash


def get_shell_config() -> tuple[str, list[str]]:
    """Return (shell, args) for the current platform.

    Resolution order:
    1. On Windows: Git Bash in known locations, then bash on PATH
    2. On Unix: /bin/bash, then bash on PATH, then fallback to sh
    """
    global _cached_shell_config
    if _cached_shell_config:
        return _cached_shell_config

    if sys.platform == "win32":
        candidates = []
        for env_var in ("ProgramFiles", "ProgramFiles(x86)"):
            d = os.environ.get(env_var, "")
            if d:
                candidates.append(os.path.join(d, "Git", "bin", "bash.exe"))

        for path in candidates:
            if os.path.exists(path):
                _cached_shell_config = (path, ["-c"])
                return _cached_shell_config

        bash = _find_bash_on_path()
        if bash:
            _cached_shell_config = (bash, ["-c"])
            return _cached_shell_config

        raise RuntimeError("No bash shell found. Install Git for Windows or add bash to PATH.")

    if os.path.exists("/bin/bash"):
        _cached_shell_config = ("/bin/bash", ["-c"])
        return _cached_shell_config

    bash = _find_bash_on_path()
    if bash:
        _cached_shell_config = (bash, ["-c"])
        return _cached_shell_config

    _cached_shell_config = ("sh", ["-c"])
    return _cached_shell_config


def reset_shell_config_cache() -> None:
    """Reset shell config cache (for testing)."""
    global _cached_shell_config
    _cached_shell_config = None


def get_shell_env() -> dict[str, str]:
    """Return environment dict for shell execution, prepending bin dir."""
    env = dict(os.environ)
    try:
        from pi_coding_agent.config import get_bin_dir
        bin_dir = get_bin_dir()
        path_key = next((k for k in env if k.upper() == "PATH"), "PATH")
        current_path = env.get(path_key, "")
        entries = [e for e in current_path.split(os.pathsep) if e]
        if bin_dir not in entries:
            env[path_key] = os.pathsep.join([bin_dir, current_path])
    except Exception:
        pass
    return env


_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_BINARY_GARBAGE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufff9-\ufffb]")


def sanitize_binary_output(text: str) -> str:
    """Remove characters that cause display issues.

    Removes:
    - Control characters (except tab, newline, CR)
    - Lone surrogates
    - Unicode Format characters
    """
    result = []
    for ch in text:
        cp = ord(ch)
        if cp in (0x09, 0x0A, 0x0D):  # tab, LF, CR
            result.append(ch)
        elif cp <= 0x1F:
            continue
        elif 0xFFF9 <= cp <= 0xFFFB:
            continue
        else:
            result.append(ch)
    return "".join(result)


def kill_process_tree(pid: int) -> None:
    """Kill a process and all its children (cross-platform)."""
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                check=False,
                capture_output=True,
            )
        except Exception:
            pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
