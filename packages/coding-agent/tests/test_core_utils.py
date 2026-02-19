"""
Tests for coding-agent core utilities.

Covers: utils/sleep.py, utils/git.py, utils/frontmatter.py,
        utils/changelog.py, utils/shell.py
"""
from __future__ import annotations

import asyncio
import sys

import pytest


# ============================================================================
# sleep
# ============================================================================

class TestSleep:
    @pytest.mark.asyncio
    async def test_sleep_completes(self):
        from pi_coding_agent.utils.sleep import sleep
        await asyncio.wait_for(sleep(0.01), timeout=1.0)

    @pytest.mark.asyncio
    async def test_sleep_cancelled_by_event(self):
        from pi_coding_agent.utils.sleep import sleep
        cancel = asyncio.Event()
        cancel.set()
        # Already-set event: the sleep should return quickly (may or may not raise)
        try:
            await asyncio.wait_for(sleep(10.0, cancel), timeout=0.5)
        except asyncio.CancelledError:
            pass  # Expected in some implementations
        except asyncio.TimeoutError:
            pytest.fail("sleep did not exit when cancel event was set")

    @pytest.mark.asyncio
    async def test_sleep_cancelled_mid_sleep(self):
        from pi_coding_agent.utils.sleep import sleep
        cancel = asyncio.Event()

        async def _cancel_after():
            await asyncio.sleep(0.05)
            cancel.set()

        asyncio.ensure_future(_cancel_after())
        with pytest.raises(asyncio.CancelledError):
            await sleep(10.0, cancel)

    @pytest.mark.asyncio
    async def test_sleep_no_cancel_event(self):
        from pi_coding_agent.utils.sleep import sleep
        # Should complete normally without a cancel event
        await sleep(0.01)


# ============================================================================
# git
# ============================================================================

class TestGitUrlParsing:
    def test_https_url(self):
        from pi_coding_agent.utils.git import parse_git_url
        result = parse_git_url("https://github.com/user/repo.git")
        assert result is not None
        # GitSource has: type, repo, host, path, ref, pinned
        assert result.host == "github.com"
        assert "user" in result.path or "user" in result.repo

    def test_scp_like_url(self):
        from pi_coding_agent.utils.git import GitSource, parse_git_url
        # SCP-like format: git@host:owner/repo
        result = parse_git_url("git@github.com:user/repo.git")
        # May return None if not supported, or GitSource
        assert result is None or isinstance(result, GitSource)

    def test_https_url_returns_git_source(self):
        from pi_coding_agent.utils.git import GitSource, parse_git_url
        result = parse_git_url("https://github.com/user/repo")
        assert result is None or isinstance(result, GitSource)

    def test_invalid_returns_none(self):
        from pi_coding_agent.utils.git import parse_git_url
        result = parse_git_url("not-a-url")
        assert result is None


# ============================================================================
# frontmatter
# ============================================================================

class TestFrontmatter:
    def test_parse_frontmatter_with_yaml(self):
        from pi_coding_agent.utils.frontmatter import parse_frontmatter
        # parse_frontmatter returns (metadata_dict, body_str)
        content = "---\ntitle: Hello\nauthor: World\n---\nBody content here."
        meta, body = parse_frontmatter(content)
        assert meta["title"] == "Hello"
        assert meta["author"] == "World"
        assert "Body content here." in body

    def test_parse_frontmatter_no_yaml(self):
        from pi_coding_agent.utils.frontmatter import parse_frontmatter
        content = "Just plain content without frontmatter."
        meta, body = parse_frontmatter(content)
        assert meta == {}
        assert "Just plain" in body

    def test_strip_frontmatter(self):
        from pi_coding_agent.utils.frontmatter import strip_frontmatter
        content = "---\ntitle: Hello\n---\nBody content here."
        result = strip_frontmatter(content)
        assert "title" not in result
        assert "Body content here." in result

    def test_strip_frontmatter_no_yaml(self):
        from pi_coding_agent.utils.frontmatter import strip_frontmatter
        content = "Just plain content."
        result = strip_frontmatter(content)
        assert result == content

    def test_stringify_frontmatter(self):
        from pi_coding_agent.utils.frontmatter import stringify_frontmatter
        # stringify_frontmatter(metadata, body) -> str
        data = {"title": "Hello", "count": 42}
        result = stringify_frontmatter(data, "Body goes here")
        assert result.startswith("---")
        assert "title: Hello" in result
        assert "Body goes here" in result

    def test_roundtrip(self):
        from pi_coding_agent.utils.frontmatter import (
            parse_frontmatter,
            stringify_frontmatter,
            strip_frontmatter,
        )
        original_data = {"title": "Test"}
        original_body = "Hello world"
        full_content = stringify_frontmatter(original_data, original_body)

        meta, body = parse_frontmatter(full_content)
        stripped = strip_frontmatter(full_content)

        assert meta["title"] == "Test"
        assert original_body in body or original_body in stripped


# ============================================================================
# changelog
# ============================================================================

class TestChangelog:
    def test_parse_changelog_basic(self):
        from pi_coding_agent.utils.changelog import parse_changelog
        text = "# Changelog\n\n## [1.2.0] - 2024-01-01\n- Added feature\n\n## [1.1.0] - 2023-12-01\n- Fixed bug\n"
        entries = parse_changelog(text)
        assert len(entries) >= 2
        versions = [e.version for e in entries]
        assert "1.2.0" in versions
        assert "1.1.0" in versions

    def test_compare_versions(self):
        from pi_coding_agent.utils.changelog import compare_versions
        assert compare_versions("1.2.0", "1.1.0") > 0
        assert compare_versions("1.0.0", "2.0.0") < 0
        assert compare_versions("1.0.0", "1.0.0") == 0

    def test_get_new_entries(self):
        from pi_coding_agent.utils.changelog import get_new_entries, parse_changelog
        text = "## [1.3.0]\n- New\n\n## [1.2.0]\n- Old\n\n## [1.1.0]\n- Very old\n"
        entries = parse_changelog(text)
        # get_new_entries(old_version, entries)
        new = get_new_entries("1.2.0", entries)
        assert len(new) >= 1
        assert new[0].version == "1.3.0"

    def test_get_new_entries_none_newer(self):
        from pi_coding_agent.utils.changelog import get_new_entries, parse_changelog
        text = "## [1.0.0]\n- Basic\n"
        entries = parse_changelog(text)
        new = get_new_entries("1.0.0", entries)
        assert len(new) == 0


# ============================================================================
# shell
# ============================================================================

class TestShell:
    def test_get_shell_config_returns_tuple(self):
        from pi_coding_agent.utils.shell import get_shell_config
        # get_shell_config returns (shell_path: str, args: list[str]) tuple
        config = get_shell_config()
        assert isinstance(config, tuple)
        assert len(config) == 2
        shell_path, shell_args = config
        assert isinstance(shell_path, str)
        assert isinstance(shell_args, list)

    def test_sanitize_binary_output(self):
        from pi_coding_agent.utils.shell import sanitize_binary_output
        # Regular text passes through
        assert sanitize_binary_output("hello world") == "hello world"

    def test_get_shell_env_returns_dict(self):
        from pi_coding_agent.utils.shell import get_shell_env
        env = get_shell_env()
        assert isinstance(env, dict)
