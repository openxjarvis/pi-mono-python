"""
Git URL parsing utilities.

Parses git repository URLs into structured GitSource objects,
handling SSH, HTTPS, SCP-like, and shorthand formats.

Mirrors utils/git.ts
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class GitSource:
    """Parsed git URL information."""

    type: str = "git"
    repo: str = ""
    host: str = ""
    path: str = ""
    ref: str | None = None
    pinned: bool = False


_SCP_RE = re.compile(r"^git@([^:]+):(.+)$")
_PROTOCOL_RE = re.compile(r"^(https?|ssh|git)://", re.IGNORECASE)


def _split_ref(url: str) -> tuple[str, str | None]:
    """Split URL into (repo, ref) components."""
    scp = _SCP_RE.match(url)
    if scp:
        path_with_ref = scp.group(2)
        at_idx = path_with_ref.find("@")
        if at_idx < 0:
            return url, None
        repo_path = path_with_ref[:at_idx]
        ref = path_with_ref[at_idx + 1:]
        if not repo_path or not ref:
            return url, None
        return f"git@{scp.group(1)}:{repo_path}", ref

    if "://" in url:
        try:
            parsed = urlparse(url)
            path = parsed.path.lstrip("/")
            at_idx = path.find("@")
            if at_idx < 0:
                return url, None
            repo_path = path[:at_idx]
            ref = path[at_idx + 1:]
            if not repo_path or not ref:
                return url, None
            new_path = f"/{repo_path}"
            new_url = parsed._replace(path=new_path).geturl().rstrip("/")
            return new_url, ref
        except Exception:
            return url, None

    slash_idx = url.find("/")
    if slash_idx < 0:
        return url, None
    host = url[:slash_idx]
    path_with_ref = url[slash_idx + 1:]
    at_idx = path_with_ref.find("@")
    if at_idx < 0:
        return url, None
    repo_path = path_with_ref[:at_idx]
    ref = path_with_ref[at_idx + 1:]
    if not repo_path or not ref:
        return url, None
    return f"{host}/{repo_path}", ref


def _parse_generic_git_url(url: str) -> GitSource | None:
    """Parse a git URL that isn't a recognized hosted shorthand."""
    repo_without_ref, ref = _split_ref(url)
    repo = repo_without_ref
    host = ""
    path = ""

    scp = _SCP_RE.match(repo_without_ref)
    if scp:
        host = scp.group(1)
        path = scp.group(2)
    elif any(repo_without_ref.startswith(p) for p in ("https://", "http://", "ssh://", "git://")):
        try:
            parsed = urlparse(repo_without_ref)
            host = parsed.hostname or ""
            path = parsed.path.lstrip("/")
        except Exception:
            return None
    else:
        slash_idx = repo_without_ref.find("/")
        if slash_idx < 0:
            return None
        host = repo_without_ref[:slash_idx]
        path = repo_without_ref[slash_idx + 1:]
        if "." not in host and host != "localhost":
            return None
        repo = f"https://{repo_without_ref}"

    normalized_path = re.sub(r"\.git$", "", path).lstrip("/")
    if not host or not normalized_path or len(normalized_path.split("/")) < 2:
        return None

    return GitSource(
        type="git",
        repo=repo,
        host=host,
        path=normalized_path,
        ref=ref,
        pinned=bool(ref),
    )


def parse_git_url(source: str) -> GitSource | None:
    """Parse a git URL string into a GitSource.

    Rules:
    - With ``git:`` prefix, accepts shorthand forms.
    - Without ``git:`` prefix, only accepts explicit protocol URLs.
    """
    trimmed = source.strip()
    has_git_prefix = trimmed.startswith("git:")
    url = trimmed[4:].strip() if has_git_prefix else trimmed

    if not has_git_prefix and not _PROTOCOL_RE.match(url):
        return None

    repo_without_ref, ref = _split_ref(url)

    # Try common hosted providers via simple heuristics
    parsed = _try_hosted_parse(repo_without_ref, ref)
    if parsed:
        return parsed

    return _parse_generic_git_url(url)


def _try_hosted_parse(repo: str, ref: str | None) -> GitSource | None:
    """Try to parse well-known hosted git URLs."""
    # Handle HTTPS / SSH URLs for github.com, gitlab.com, bitbucket.org
    scp = _SCP_RE.match(repo)
    if scp:
        host = scp.group(1)
        path = re.sub(r"\.git$", "", scp.group(2)).lstrip("/")
        if len(path.split("/")) >= 2:
            return GitSource(type="git", repo=repo, host=host, path=path, ref=ref, pinned=bool(ref))

    if "://" in repo:
        try:
            parsed = urlparse(repo)
            host = parsed.hostname or ""
            path = re.sub(r"\.git$", "", parsed.path.lstrip("/"))
            if host and len(path.split("/")) >= 2:
                return GitSource(type="git", repo=repo, host=host, path=path, ref=ref, pinned=bool(ref))
        except Exception:
            pass

    return None
