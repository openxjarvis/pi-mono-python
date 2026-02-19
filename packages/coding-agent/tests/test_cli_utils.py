"""
Tests for cli_sub/ subpackage.

Covers: args.py, file_processor.py, list_models.py, session_picker.py
"""
from __future__ import annotations

import os
import tempfile

import pytest


# ============================================================================
# Args parsing
# ============================================================================

class TestArgParsing:
    def test_parse_empty(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args([])
        assert args.messages == []
        assert args.file_args == []
        assert args.provider is None

    def test_parse_messages(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["Hello", "world"])
        assert args.messages == ["Hello", "world"]

    def test_parse_provider(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--provider", "anthropic"])
        assert args.provider == "anthropic"

    def test_parse_model(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--model", "claude-3-5-sonnet"])
        assert args.model == "claude-3-5-sonnet"

    def test_parse_mode(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--mode", "rpc"])
        assert args.mode == "rpc"

    def test_parse_invalid_mode(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--mode", "invalid"])
        assert args.mode is None

    def test_parse_help(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--help"])
        assert args.help is True

    def test_parse_version(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["-v"])
        assert args.version is True

    def test_parse_continue(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["-c"])
        assert args.continue_ is True

    def test_parse_resume(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["-r"])
        assert args.resume is True

    def test_parse_print(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["-p"])
        assert args.print_mode is True

    def test_parse_file_args(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["@file.txt", "@image.png"])
        assert "file.txt" in args.file_args
        assert "image.png" in args.file_args

    def test_parse_no_tools(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--no-tools"])
        assert args.no_tools is True

    def test_parse_tools(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--tools", "read,bash"])
        assert args.tools == ["read", "bash"]

    def test_parse_invalid_tool_skipped(self, capsys):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--tools", "read,nonexistent"])
        assert "nonexistent" not in (args.tools or [])
        assert "read" in (args.tools or [])

    def test_parse_thinking_level(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--thinking", "high"])
        assert args.thinking == "high"

    def test_parse_invalid_thinking_level(self, capsys):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--thinking", "invalid"])
        assert args.thinking is None

    def test_parse_no_session(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--no-session"])
        assert args.no_session is True

    def test_parse_session_path(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--session", "/path/to/session.jsonl"])
        assert args.session == "/path/to/session.jsonl"

    def test_parse_multiple_extensions(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["-e", "ext1.py", "-e", "ext2.py"])
        assert args.extensions == ["ext1.py", "ext2.py"]

    def test_parse_list_models_flag_only(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--list-models"])
        assert args.list_models is True

    def test_parse_list_models_with_search(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--list-models", "sonnet"])
        assert args.list_models == "sonnet"

    def test_parse_verbose(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--verbose"])
        assert args.verbose is True

    def test_is_valid_thinking_level(self):
        from pi_coding_agent.cli_sub.args import is_valid_thinking_level
        assert is_valid_thinking_level("high") is True
        assert is_valid_thinking_level("invalid") is False
        assert is_valid_thinking_level("off") is True

    def test_print_help_runs(self, capsys):
        from pi_coding_agent.cli_sub.args import print_help
        print_help()
        captured = capsys.readouterr()
        assert "pi" in captured.out.lower() or "usage" in captured.out.lower()


# ============================================================================
# File processor
# ============================================================================

class TestFileProcessor:
    @pytest.mark.asyncio
    async def test_process_empty_file_args(self):
        from pi_coding_agent.cli_sub.file_processor import process_file_arguments
        result = await process_file_arguments([])
        assert result.text == ""
        assert result.images == []

    @pytest.mark.asyncio
    async def test_process_text_file(self):
        from pi_coding_agent.cli_sub.file_processor import process_file_arguments
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Hello, world!")
            path = f.name
        try:
            result = await process_file_arguments([path])
            assert "Hello, world!" in result.text
            assert path in result.text
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_process_empty_file_skipped(self):
        from pi_coding_agent.cli_sub.file_processor import process_file_arguments
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            path = f.name
        try:
            result = await process_file_arguments([path])
            assert result.text == ""
        finally:
            os.unlink(path)


# ============================================================================
# List models
# ============================================================================

class TestListModels:
    @pytest.mark.asyncio
    async def test_list_models_empty_registry(self, capsys):
        from pi_coding_agent.cli_sub.list_models import list_models

        class MockRegistry:
            async def get_available(self):
                return []

        await list_models(MockRegistry())
        captured = capsys.readouterr()
        assert "No models" in captured.out

    @pytest.mark.asyncio
    async def test_list_models_with_data(self, capsys):
        from pi_coding_agent.cli_sub.list_models import list_models

        class MockModel:
            provider = "anthropic"
            id = "claude-3-5-sonnet"
            contextWindow = 200000
            maxTokens = 8192
            reasoning = False
            input = ["text", "image"]

        class MockRegistry:
            async def get_available(self):
                return [MockModel()]

        await list_models(MockRegistry())
        captured = capsys.readouterr()
        assert "anthropic" in captured.out
        assert "claude-3-5-sonnet" in captured.out

    @pytest.mark.asyncio
    async def test_list_models_with_search(self, capsys):
        from pi_coding_agent.cli_sub.list_models import list_models

        class MockModel:
            provider = "anthropic"
            id = "claude-3-5-haiku"
            contextWindow = 200000
            maxTokens = 4096
            reasoning = False
            input = ["text"]

        class MockModel2:
            provider = "openai"
            id = "gpt-4o"
            contextWindow = 128000
            maxTokens = 4096
            reasoning = False
            input = ["text", "image"]

        class MockRegistry:
            async def get_available(self):
                return [MockModel(), MockModel2()]

        await list_models(MockRegistry(), search_pattern="haiku")
        captured = capsys.readouterr()
        assert "haiku" in captured.out
        assert "gpt-4o" not in captured.out

    def test_format_token_count(self):
        from pi_coding_agent.cli_sub.list_models import _format_token_count
        assert _format_token_count(200000) == "200K"
        assert _format_token_count(1000000) == "1M"
        assert _format_token_count(500) == "500"
        assert _format_token_count(1500000) == "1.5M"
