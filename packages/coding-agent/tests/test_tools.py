"""
Tests for coding-agent tools — mirrors packages/coding-agent/test/ tool tests.
"""
from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from pi_coding_agent.core.tools import (
    create_bash_tool,
    create_edit_tool,
    create_find_tool,
    create_ls_tool,
    create_read_tool,
    create_write_tool,
)
from pi_coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    truncate_head,
    truncate_tail,
)
from pi_ai.types import ImageContent, TextContent


# ── Truncation tests ──────────────────────────────────────────────────────────

def test_truncate_head_no_truncation():
    text = "line1\nline2\nline3"
    result = truncate_head(text)
    assert result.truncated is False
    assert result.content == text


def test_truncate_head_by_lines():
    lines = [f"line{i}" for i in range(DEFAULT_MAX_LINES + 10)]
    text = "\n".join(lines)
    result = truncate_head(text)
    assert result.truncated is True
    assert result.truncated_by == "lines"
    assert result.output_lines == DEFAULT_MAX_LINES


def test_truncate_head_by_bytes():
    # Create text larger than 30KB
    big_text = "x" * (DEFAULT_MAX_BYTES + 1000)
    result = truncate_head(big_text)
    assert result.truncated is True
    assert result.truncated_by == "bytes"


def test_truncate_tail_keeps_end():
    lines = [f"line{i}" for i in range(DEFAULT_MAX_LINES + 10)]
    text = "\n".join(lines)
    result = truncate_tail(text)
    assert result.truncated is True
    # The last lines should be preserved
    last_line = f"line{DEFAULT_MAX_LINES + 9}"
    assert last_line in result.content


# ── Read tool tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_read_tool_reads_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Hello, World!\nLine 2\nLine 3")

        tool = create_read_tool(tmpdir)
        result = await tool.execute("tc1", {"path": "test.txt"})

        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert "Hello, World!" in result.content[0].text


@pytest.mark.asyncio
async def test_read_tool_offset_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        lines = [f"line{i}" for i in range(1, 21)]
        with open(test_file, "w") as f:
            f.write("\n".join(lines))

        tool = create_read_tool(tmpdir)
        result = await tool.execute("tc1", {"path": "test.txt", "offset": 5, "limit": 3})

        text = result.content[0].text
        assert "line5" in text
        assert "line6" in text
        assert "line7" in text
        assert "line8" not in text


@pytest.mark.asyncio
async def test_read_tool_file_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        tool = create_read_tool(tmpdir)
        with pytest.raises(FileNotFoundError):
            await tool.execute("tc1", {"path": "nonexistent.txt"})


# ── Write tool tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_tool_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tool = create_write_tool(tmpdir)
        result = await tool.execute("tc1", {"path": "new_file.txt", "content": "Hello!"})

        assert isinstance(result.content[0], TextContent)
        assert "Successfully wrote" in result.content[0].text

        with open(os.path.join(tmpdir, "new_file.txt")) as f:
            assert f.read() == "Hello!"


@pytest.mark.asyncio
async def test_write_tool_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tool = create_write_tool(tmpdir)
        result = await tool.execute("tc1", {
            "path": "subdir/nested/file.txt",
            "content": "nested content",
        })

        full_path = os.path.join(tmpdir, "subdir", "nested", "file.txt")
        assert os.path.exists(full_path)
        with open(full_path) as f:
            assert f.read() == "nested content"


@pytest.mark.asyncio
async def test_write_tool_overwrites():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("original")

        tool = create_write_tool(tmpdir)
        await tool.execute("tc1", {"path": "test.txt", "content": "overwritten"})

        with open(test_file) as f:
            assert f.read() == "overwritten"


# ── Edit tool tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edit_tool_replaces_text():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Hello World\nThis is a test\nGoodbye")

        tool = create_edit_tool(tmpdir)
        result = await tool.execute("tc1", {
            "path": "test.txt",
            "oldText": "Hello World",
            "newText": "Hi There",
        })

        assert "Successfully replaced" in result.content[0].text

        with open(test_file) as f:
            content = f.read()
        assert "Hi There" in content
        assert "Hello World" not in content


@pytest.mark.asyncio
async def test_edit_tool_text_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Hello World")

        tool = create_edit_tool(tmpdir)
        with pytest.raises(ValueError, match="Could not find"):
            await tool.execute("tc1", {
                "path": "test.txt",
                "oldText": "nonexistent text",
                "newText": "replacement",
            })


@pytest.mark.asyncio
async def test_edit_tool_file_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        tool = create_edit_tool(tmpdir)
        with pytest.raises(FileNotFoundError):
            await tool.execute("tc1", {
                "path": "nonexistent.txt",
                "oldText": "x",
                "newText": "y",
            })


# ── Bash tool tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bash_tool_executes_command():
    with tempfile.TemporaryDirectory() as tmpdir:
        tool = create_bash_tool(tmpdir)
        result = await tool.execute("tc1", {"command": "echo hello"})

        assert isinstance(result.content[0], TextContent)
        assert "hello" in result.content[0].text


@pytest.mark.asyncio
async def test_bash_tool_exit_code_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        tool = create_bash_tool(tmpdir)
        with pytest.raises(RuntimeError, match="exited with code"):
            await tool.execute("tc1", {"command": "exit 1"})


@pytest.mark.asyncio
async def test_bash_tool_captures_stderr():
    with tempfile.TemporaryDirectory() as tmpdir:
        tool = create_bash_tool(tmpdir)
        # This should produce output on stderr
        try:
            await tool.execute("tc1", {"command": "ls /nonexistent_path_xyz"})
        except RuntimeError as e:
            assert len(str(e)) > 0  # Should have error output


# ── LS tool tests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ls_tool_lists_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files
        for name in ["file1.txt", "file2.py", "subdir"]:
            path = os.path.join(tmpdir, name)
            if name == "subdir":
                os.mkdir(path)
            else:
                open(path, "w").close()

        tool = create_ls_tool(tmpdir)
        result = await tool.execute("tc1", {})

        text = result.content[0].text
        assert "file1.txt" in text
        assert "file2.py" in text
        assert "subdir/" in text  # Directories have trailing slash


@pytest.mark.asyncio
async def test_ls_tool_sorted():
    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ["zzz.txt", "aaa.txt", "mmm.txt"]:
            open(os.path.join(tmpdir, name), "w").close()

        tool = create_ls_tool(tmpdir)
        result = await tool.execute("tc1", {})

        lines = result.content[0].text.split("\n")
        names = [l.strip() for l in lines if l.strip()]
        assert names == sorted(names, key=str.lower)


@pytest.mark.asyncio
async def test_ls_tool_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tool = create_ls_tool(tmpdir)
        result = await tool.execute("tc1", {})
        assert "(empty directory)" in result.content[0].text


@pytest.mark.asyncio
async def test_ls_tool_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        tool = create_ls_tool(tmpdir)
        with pytest.raises(FileNotFoundError):
            await tool.execute("tc1", {"path": "nonexistent_dir"})


# ── Find tool tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_find_tool_matches_pattern():
    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ["test.py", "test.ts", "main.py", "README.md"]:
            open(os.path.join(tmpdir, name), "w").close()

        tool = create_find_tool(tmpdir)
        result = await tool.execute("tc1", {"pattern": "*.py"})

        text = result.content[0].text
        assert "test.py" in text
        assert "main.py" in text
        # .ts and .md should not match
        assert "test.ts" not in text
        assert "README.md" not in text


@pytest.mark.asyncio
async def test_find_tool_no_matches():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "test.txt"), "w").close()

        tool = create_find_tool(tmpdir)
        result = await tool.execute("tc1", {"pattern": "*.xyz"})

        assert "No files found" in result.content[0].text
