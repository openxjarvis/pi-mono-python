"""
Tool binary manager.

Downloads and manages external binary tools (rg, fd) from GitHub releases.
Checks local tools directory first, then system PATH, then downloads if needed.

Mirrors utils/tools-manager.ts
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Callable, Literal

import httpx


ToolName = Literal["fd", "rg"]

_PLATFORM = sys.platform  # "darwin", "linux", "win32"
_ARCH = platform.machine()  # "x86_64", "arm64", "aarch64"


def _normalize_arch() -> str:
    arch = _ARCH.lower()
    if arch in ("arm64", "aarch64"):
        return "aarch64"
    return "x86_64"


@dataclass
class ToolConfig:
    name: str
    repo: str
    binary_name: str
    tag_prefix: str
    get_asset_name: Callable[[str, str, str], str | None]


def _fd_asset(version: str, plat: str, arch: str) -> str | None:
    a = "aarch64" if arch == "aarch64" else "x86_64"
    if plat == "darwin":
        return f"fd-v{version}-{a}-apple-darwin.tar.gz"
    if plat == "linux":
        return f"fd-v{version}-{a}-unknown-linux-gnu.tar.gz"
    if plat == "win32":
        return f"fd-v{version}-{a}-pc-windows-msvc.zip"
    return None


def _rg_asset(version: str, plat: str, arch: str) -> str | None:
    a = "aarch64" if arch == "aarch64" else "x86_64"
    if plat == "darwin":
        return f"ripgrep-{version}-{a}-apple-darwin.tar.gz"
    if plat == "linux":
        if arch == "aarch64":
            return f"ripgrep-{version}-aarch64-unknown-linux-gnu.tar.gz"
        return f"ripgrep-{version}-x86_64-unknown-linux-musl.tar.gz"
    if plat == "win32":
        return f"ripgrep-{version}-{a}-pc-windows-msvc.zip"
    return None


TOOLS: dict[str, ToolConfig] = {
    "fd": ToolConfig(
        name="fd",
        repo="sharkdp/fd",
        binary_name="fd",
        tag_prefix="v",
        get_asset_name=_fd_asset,
    ),
    "rg": ToolConfig(
        name="ripgrep",
        repo="BurntSushi/ripgrep",
        binary_name="rg",
        tag_prefix="",
        get_asset_name=_rg_asset,
    ),
}

_APP_NAME = "pi"


def _get_bin_dir() -> str:
    try:
        from pi_coding_agent.config import get_bin_dir
        return get_bin_dir()
    except Exception:
        home = os.path.expanduser("~")
        return os.path.join(home, ".pi", "bin")


def _command_exists(cmd: str) -> bool:
    try:
        result = subprocess.run(
            [cmd, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        return True
    except (FileNotFoundError, PermissionError, OSError):
        return False
    except Exception:
        return False


def get_tool_path(tool: ToolName) -> str | None:
    """Return path to the tool binary, or None if not found."""
    config = TOOLS.get(tool)
    if not config:
        return None

    tools_dir = _get_bin_dir()
    ext = ".exe" if _PLATFORM == "win32" else ""
    local_path = os.path.join(tools_dir, config.binary_name + ext)
    if os.path.exists(local_path):
        return local_path

    if _command_exists(config.binary_name):
        return config.binary_name

    return None


async def _get_latest_version(repo: str) -> str:
    """Fetch latest release version from GitHub API."""
    async with httpx.AsyncClient(
        headers={"User-Agent": f"{_APP_NAME}-coding-agent"},
        follow_redirects=True,
        timeout=30.0,
    ) as client:
        resp = await client.get(f"https://api.github.com/repos/{repo}/releases/latest")
        resp.raise_for_status()
        data = resp.json()
        tag = data.get("tag_name", "")
        return tag.lstrip("v")


async def _download_file(url: str, dest: str) -> None:
    """Download a file from URL to dest path."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=300.0) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    f.write(chunk)


def _extract_binary(archive_path: str, asset_name: str, config: ToolConfig, dest_dir: str) -> str:
    """Extract binary from archive and return path to extracted binary."""
    ext = ".exe" if _PLATFORM == "win32" else ""
    binary_name = config.binary_name + ext

    with tempfile.TemporaryDirectory() as extract_dir:
        if asset_name.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
        elif asset_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(path=extract_dir)
        else:
            raise ValueError(f"Unknown archive format: {asset_name}")

        # Look for binary recursively in extracted files
        for root, dirs, files in os.walk(extract_dir):
            for fname in files:
                if fname == binary_name:
                    src = os.path.join(root, fname)
                    dest = os.path.join(dest_dir, binary_name)
                    shutil.move(src, dest)
                    return dest

    raise FileNotFoundError(f"Binary '{binary_name}' not found in archive: {archive_path}")


async def _download_tool(tool: ToolName) -> str:
    """Download and install a tool binary. Returns the installed path."""
    config = TOOLS.get(tool)
    if not config:
        raise ValueError(f"Unknown tool: {tool}")

    arch = _normalize_arch()
    version = await _get_latest_version(config.repo)
    asset_name = config.get_asset_name(version, _PLATFORM, arch)
    if not asset_name:
        raise RuntimeError(f"Unsupported platform: {_PLATFORM}/{arch}")

    tools_dir = _get_bin_dir()
    os.makedirs(tools_dir, exist_ok=True)

    download_url = (
        f"https://github.com/{config.repo}/releases/download/"
        f"{config.tag_prefix}{version}/{asset_name}"
    )

    archive_path = os.path.join(tools_dir, asset_name)
    await _download_file(download_url, archive_path)

    try:
        binary_path = _extract_binary(archive_path, asset_name, config, tools_dir)
        if _PLATFORM != "win32":
            os.chmod(binary_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        return binary_path
    finally:
        try:
            os.remove(archive_path)
        except OSError:
            pass


_TERMUX_PACKAGES: dict[str, str] = {
    "fd": "fd",
    "rg": "ripgrep",
}


async def ensure_tool(tool: ToolName, silent: bool = False) -> str | None:
    """Ensure a tool is available, downloading if necessary.

    Returns the path to the tool executable, or None if unavailable.
    """
    existing = get_tool_path(tool)
    if existing:
        return existing

    config = TOOLS.get(tool)
    if not config:
        return None

    # On Android/Termux, pre-built binaries won't work
    if _PLATFORM == "android":
        pkg = _TERMUX_PACKAGES.get(tool, tool)
        if not silent:
            print(f"[yellow]{config.name} not found. Install with: pkg install {pkg}[/yellow]")
        return None

    if not silent:
        print(f"{config.name} not found. Downloading...")

    try:
        path = await _download_tool(tool)
        if not silent:
            print(f"{config.name} installed to {path}")
        return path
    except Exception as e:
        if not silent:
            print(f"Failed to download {config.name}: {e}")
        return None
