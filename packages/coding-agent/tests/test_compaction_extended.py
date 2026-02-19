"""
Extended tests for core/compaction/ subpackage.

Covers: utils.py, branch_summarization.py data structures, and compaction.py logic.
"""
from __future__ import annotations

import pytest


# ============================================================================
# FileOperations (utils.py)
# ============================================================================

class TestFileOperations:
    def test_track_write(self):
        from pi_coding_agent.core.compaction.utils import FileOperations
        # FileOperations has: read (set), written (set), edited (set)
        ops = FileOperations()
        ops.written.add("/tmp/foo.py")
        assert "/tmp/foo.py" in ops.written

    def test_track_read(self):
        from pi_coding_agent.core.compaction.utils import FileOperations
        ops = FileOperations()
        ops.read.add("/tmp/foo.py")
        assert "/tmp/foo.py" in ops.read

    def test_track_multiple(self):
        from pi_coding_agent.core.compaction.utils import FileOperations
        ops = FileOperations()
        ops.written.add("/a.py")
        ops.written.add("/b.py")
        ops.read.add("/c.py")
        assert len(ops.written) == 2
        assert len(ops.read) == 1

    def test_create_file_ops(self):
        from pi_coding_agent.core.compaction.utils import create_file_ops
        ops = create_file_ops()
        assert isinstance(ops.read, set)
        assert isinstance(ops.written, set)


class TestExtractFileOps:
    def test_extract_from_non_assistant_message(self):
        from pi_coding_agent.core.compaction.utils import FileOperations, extract_file_ops_from_message
        # extract_file_ops_from_message(message, file_ops) -> None (modifies file_ops in place)
        # Only processes assistant messages, so user messages are ignored
        from types import SimpleNamespace
        msg = SimpleNamespace(role="user", content=[{"type": "text", "text": "Hello"}])
        ops = FileOperations()
        extract_file_ops_from_message(msg, ops)
        # No changes expected for non-assistant message
        assert len(ops.written) == 0

    def test_extract_from_assistant_message(self):
        from pi_coding_agent.core.compaction.utils import FileOperations, extract_file_ops_from_message
        from types import SimpleNamespace
        # Should handle assistant messages without error
        msg = SimpleNamespace(role="assistant", content=[{"type": "text", "text": "I wrote some code"}])
        ops = FileOperations()
        extract_file_ops_from_message(msg, ops)
        assert isinstance(ops, FileOperations)


class TestSerializeConversation:
    def test_serialize_empty(self):
        from pi_coding_agent.core.compaction.utils import serialize_conversation
        result = serialize_conversation([])
        assert isinstance(result, str)

    def test_serialize_returns_string(self):
        from pi_coding_agent.core.compaction.utils import serialize_conversation
        # serialize_conversation may handle various message formats
        messages = [{"role": "user", "content": "Hello"}]
        result = serialize_conversation(messages)
        assert isinstance(result, str)


# ============================================================================
# should_compact / compact_context
# ============================================================================

class TestShouldCompact:
    def _make_user_messages(self, total_chars: int) -> list:
        """Create UserMessage objects with approximately total_chars characters."""
        from pi_ai.types import TextContent, UserMessage
        text = "x" * total_chars
        return [UserMessage(content=[TextContent(type="text", text=text)], timestamp=0)]

    def test_should_compact_false_when_low_tokens(self):
        from pi_coding_agent.core.compaction.compaction import should_compact
        messages = self._make_user_messages(400)  # 400 chars = ~100 tokens
        result = should_compact(messages, context_window=200000, threshold=0.9)
        assert result is False

    def test_should_compact_true_when_over_threshold(self):
        from pi_coding_agent.core.compaction.compaction import should_compact
        # context_window=100 tokens, threshold=0.5 → need >=50 tokens
        # 200 chars → ~50 tokens
        messages = self._make_user_messages(200)
        result = should_compact(messages, context_window=100, threshold=0.5)
        assert result is True

    def test_should_compact_default_threshold(self):
        from pi_coding_agent.core.compaction.compaction import should_compact
        messages = self._make_user_messages(40)  # ~10 tokens, well below threshold
        result = should_compact(messages, context_window=200000)
        assert result is False


# ============================================================================
# BranchSummaryDetails dataclass
# ============================================================================

class TestBranchSummaryDetails:
    def test_branch_summary_details_creation(self):
        from pi_coding_agent.core.compaction.branch_summarization import BranchSummaryDetails
        # BranchSummaryDetails has: read_files, modified_files
        details = BranchSummaryDetails(
            read_files=["/bar.py"],
            modified_files=["/foo.py"],
        )
        assert "/foo.py" in details.modified_files
        assert "/bar.py" in details.read_files

    def test_branch_summary_result_creation(self):
        from pi_coding_agent.core.compaction.branch_summarization import BranchSummaryResult
        # BranchSummaryResult has: summary, read_files, modified_files, aborted, error
        result = BranchSummaryResult(
            summary="Summary of changes",
            read_files=["/a.py"],
            modified_files=["/b.py"],
        )
        assert result.summary == "Summary of changes"
        assert result.aborted is False

    def test_collect_entries_result(self):
        from pi_coding_agent.core.compaction.branch_summarization import CollectEntriesResult
        result = CollectEntriesResult(entries=[], common_ancestor_id="abc")
        assert result.common_ancestor_id == "abc"

    def test_branch_preparation(self):
        from pi_coding_agent.core.compaction.branch_summarization import BranchPreparation
        prep = BranchPreparation(messages=[], total_tokens=500)
        assert prep.total_tokens == 500


# ============================================================================
# SUMMARIZATION_SYSTEM_PROMPT
# ============================================================================

class TestSummarizationPrompt:
    def test_prompt_is_string(self):
        from pi_coding_agent.core.compaction.utils import SUMMARIZATION_SYSTEM_PROMPT
        assert isinstance(SUMMARIZATION_SYSTEM_PROMPT, str)
        assert len(SUMMARIZATION_SYSTEM_PROMPT) > 50
