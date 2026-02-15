"""Tests for Phase 1 hook scripts -- 22 tests."""

from __future__ import annotations

import json
import os
import threading
from unittest.mock import patch

import pytest

from claude_mem_lite.hooks.capture import _process_event as capture_process_event
from claude_mem_lite.hooks.cleanup import _process_event as cleanup_process_event
from claude_mem_lite.hooks.context import (
    _get_basic_context,
    _process_event as context_process_event,
)
from claude_mem_lite.hooks.summary import (
    _process_event as summary_process_event,
    _should_skip,
)
from claude_mem_lite.logging.logger import MemLogger
from claude_mem_lite.storage.models import SessionStatus
from claude_mem_lite.storage.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_posttooluse_event(
    *,
    session_id: str = "test-session-001",
    tool_name: str = "Write",
    tool_input: dict | None = None,
    tool_response: dict | None = None,
    cwd: str = "/home/user/project",
) -> dict:
    """Build a PostToolUse hook event dict."""
    return {
        "session_id": session_id,
        "transcript_path": "/tmp/test-transcript.jsonl",
        "cwd": cwd,
        "permission_mode": "default",
        "hook_event_name": "PostToolUse",
        "tool_name": tool_name,
        "tool_input": tool_input or {},
        "tool_response": tool_response or {},
        "tool_use_id": "toolu_01ABC123",
    }


def _make_sessionstart_event(
    *,
    session_id: str = "test-session-001",
    source: str = "startup",
    cwd: str = "/home/user/project",
) -> dict:
    """Build a SessionStart hook event dict."""
    return {
        "session_id": session_id,
        "transcript_path": "/tmp/test-transcript.jsonl",
        "cwd": cwd,
        "permission_mode": "default",
        "hook_event_name": "SessionStart",
        "source": source,
    }


def _make_stop_event(
    *,
    session_id: str = "test-session-001",
    stop_hook_active: bool = False,
) -> dict:
    """Build a Stop hook event dict."""
    return {
        "session_id": session_id,
        "transcript_path": "/tmp/test-transcript.jsonl",
        "cwd": "/home/user/project",
        "permission_mode": "default",
        "hook_event_name": "Stop",
        "stop_hook_active": stop_hook_active,
    }


def _make_sessionend_event(
    *,
    session_id: str = "test-session-001",
    reason: str = "other",
) -> dict:
    """Build a SessionEnd hook event dict."""
    return {
        "session_id": session_id,
        "transcript_path": "/tmp/test-transcript.jsonl",
        "cwd": "/home/user/project",
        "permission_mode": "default",
        "hook_event_name": "SessionEnd",
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# TestCapture -- 6 tests
# ---------------------------------------------------------------------------
class TestCapture:
    """Tests for capture.py (_process_event)."""

    def test_capture_writes_to_queue(self, store):
        """Call _process_event with a Write tool event. Verify item in pending_queue."""
        event = _make_posttooluse_event(
            tool_name="Write",
            tool_input={"file_path": "/src/main.py", "content": "print('hello')"},
            tool_response={"filePath": "/src/main.py", "success": True},
        )
        capture_process_event(event, store)

        count = store.count_pending(event["session_id"])
        assert count == 1

        # Verify the item via dequeue
        batch = store.dequeue_batch(n=1)
        assert len(batch) == 1
        item = batch[0]
        assert item.tool_name == "Write"
        assert item.session_id == "test-session-001"

        raw = json.loads(item.raw_output)
        assert "tool_input" in raw
        assert "tool_response" in raw

    def test_capture_priority_high_for_mutations(self, store):
        """Mutation tools (Write, Edit, MultiEdit, Bash) get priority='high'."""
        high_tools = ["Write", "Edit", "MultiEdit", "Bash"]
        for tool in high_tools:
            event = _make_posttooluse_event(tool_name=tool, session_id=f"sess-{tool}")
            capture_process_event(event, store)

        batch = store.dequeue_batch(n=10)
        for item in batch:
            assert item.priority == "high", f"{item.tool_name} should be high priority"

    def test_capture_priority_low_for_reads(self, store):
        """Read-only tools (Read, Glob, Grep) get priority='low'."""
        low_tools = ["Read", "Glob", "Grep"]
        for tool in low_tools:
            event = _make_posttooluse_event(tool_name=tool, session_id=f"sess-{tool}")
            capture_process_event(event, store)

        batch = store.dequeue_batch(n=10)
        for item in batch:
            assert item.priority == "low", f"{item.tool_name} should be low priority"

    def test_capture_priority_normal_for_others(self, store):
        """Non-mutation, non-read tools (WebFetch) get priority='normal'."""
        event = _make_posttooluse_event(tool_name="WebFetch")
        capture_process_event(event, store)

        batch = store.dequeue_batch(n=1)
        assert len(batch) == 1
        assert batch[0].priority == "normal"

    def test_capture_extracts_files_from_write(self, store):
        """Write tool_input with file_path populates files_touched."""
        event = _make_posttooluse_event(
            tool_name="Write",
            tool_input={"file_path": "/src/auth.py", "content": "code"},
        )
        capture_process_event(event, store)

        batch = store.dequeue_batch(n=1)
        assert len(batch) == 1
        files = json.loads(batch[0].files_touched)
        assert "/src/auth.py" in files

    def test_capture_uses_claude_project_dir(self, store):
        """CLAUDE_PROJECT_DIR env var takes precedence over event cwd."""
        # With CLAUDE_PROJECT_DIR set
        with patch.dict(os.environ, {"CLAUDE_PROJECT_DIR": "/canonical/root"}):
            event = _make_posttooluse_event(cwd="/other/dir")
            capture_process_event(event, store)

        batch = store.dequeue_batch(n=1)
        raw = json.loads(batch[0].raw_output)
        assert raw["project_dir"] == "/canonical/root"

        # Without CLAUDE_PROJECT_DIR -- fallback to cwd
        env_without = {k: v for k, v in os.environ.items() if k != "CLAUDE_PROJECT_DIR"}
        with patch.dict(os.environ, env_without, clear=True):
            event = _make_posttooluse_event(cwd="/fallback/dir")
            capture_process_event(event, store)

        batch = store.dequeue_batch(n=1)
        raw = json.loads(batch[0].raw_output)
        assert raw["project_dir"] == "/fallback/dir"


# ---------------------------------------------------------------------------
# TestContext -- 5 tests
# ---------------------------------------------------------------------------
class TestContext:
    """Tests for context.py (_process_event, _get_basic_context)."""

    def test_context_creates_session(self, store):
        """SessionStart event creates a session in the DB."""
        event = _make_sessionstart_event(session_id="ctx-sess-001")
        context_process_event(event, store)

        session = store.get_session("ctx-sess-001")
        assert session is not None
        assert session.project_dir != ""

    def test_context_idempotent_on_resume(self, store):
        """Resume source with existing session_id does not create a duplicate."""
        # Create session via first startup event
        event = _make_sessionstart_event(session_id="ctx-sess-002", source="startup")
        context_process_event(event, store)

        # Resume with same session_id
        event_resume = _make_sessionstart_event(session_id="ctx-sess-002", source="resume")
        context_process_event(event_resume, store)

        sessions = store.list_sessions()
        assert len(sessions) == 1

    def test_context_builds_minimal_context(self, store):
        """_get_basic_context returns header and session entries."""
        store.create_session(project_dir="/proj/a")
        store.create_session(project_dir="/proj/b")
        store.create_session(project_dir="/proj/c")

        result = _get_basic_context(store)
        assert "Recent Sessions" in result
        # Should have at least some session entries
        assert result.count("-") >= 3  # dash-prefixed list items

    def test_context_empty_db_returns_empty(self, store):
        """_get_basic_context with no sessions returns empty string."""
        result = _get_basic_context(store)
        assert result == ""

    def test_context_uses_claude_project_dir(self, store):
        """CLAUDE_PROJECT_DIR env var used for session project_dir."""
        with patch.dict(os.environ, {"CLAUDE_PROJECT_DIR": "/env/project/root"}):
            event = _make_sessionstart_event(session_id="ctx-sess-env", cwd="/different/cwd")
            context_process_event(event, store)

        session = store.get_session("ctx-sess-env")
        assert session is not None
        assert session.project_dir == "/env/project/root"


# ---------------------------------------------------------------------------
# TestSummary -- 4 tests
# ---------------------------------------------------------------------------
class TestSummary:
    """Tests for summary.py (_process_event, _should_skip)."""

    def test_summary_logs_stop_event(self, store, tmp_config):
        """_process_event logs a 'hook.stop' event to event_log."""
        logger = MemLogger(tmp_config.log_dir, store.conn)
        session = store.create_session(project_dir="/tmp/proj")

        event = _make_stop_event(session_id=session.id)
        summary_process_event(event, store, logger)

        events = store.query_events(event_type="hook.stop")
        assert len(events) == 1

    def test_summary_fast_exit_on_stop_hook_active(self):
        """_should_skip returns True when stop_hook_active is True."""
        event_active = _make_stop_event(stop_hook_active=True)
        assert _should_skip(event_active) is True

        event_normal = _make_stop_event(stop_hook_active=False)
        assert _should_skip(event_normal) is False

    def test_summary_counts_pending(self, store, tmp_config):
        """_process_event event_log data includes pending_observations count."""
        logger = MemLogger(tmp_config.log_dir, store.conn)
        session = store.create_session(project_dir="/tmp/proj")

        # Enqueue 3 items for this session
        for i in range(3):
            store.enqueue(
                session_id=session.id,
                tool_name=f"Tool{i}",
                raw_output=f"output-{i}",
            )

        event = _make_stop_event(session_id=session.id)
        summary_process_event(event, store, logger)

        events = store.query_events(event_type="hook.stop")
        assert len(events) == 1
        data = json.loads(events[0].data)
        assert data["pending_observations"] == 3

    def test_summary_handles_missing_session(self, store, tmp_config):
        """_process_event with non-existent session_id does not raise."""
        logger = MemLogger(tmp_config.log_dir, store.conn)

        event = _make_stop_event(session_id="nonexistent-session")
        # Should not raise
        summary_process_event(event, store, logger)


# ---------------------------------------------------------------------------
# TestCleanup -- 4 tests
# ---------------------------------------------------------------------------
class TestCleanup:
    """Tests for cleanup.py (_process_event)."""

    def test_cleanup_closes_session(self, store):
        """SessionEnd event sets session status to 'closed' and ended_at."""
        session = store.create_session(project_dir="/tmp/proj")
        assert store.get_session(session.id).status == SessionStatus.ACTIVE

        logger = MemLogger(store.db_path.parent / "logs", store.conn)
        event = _make_sessionend_event(session_id=session.id)
        cleanup_process_event(event, store, logger)

        updated = store.get_session(session.id)
        assert updated.status == SessionStatus.CLOSED
        assert updated.ended_at is not None

    def test_cleanup_calls_passive_checkpoint(self, store):
        """_process_event calls checkpoint with mode='PASSIVE'."""
        session = store.create_session(project_dir="/tmp/proj")
        logger = MemLogger(store.db_path.parent / "logs", store.conn)
        event = _make_sessionend_event(session_id=session.id)

        with patch.object(store, "checkpoint") as mock_checkpoint:
            cleanup_process_event(event, store, logger)
            mock_checkpoint.assert_called_once_with(mode="PASSIVE")

    def test_cleanup_logs_event(self, store, tmp_config):
        """_process_event logs a 'hook.session_end' event to event_log."""
        session = store.create_session(project_dir="/tmp/proj")
        logger = MemLogger(tmp_config.log_dir, store.conn)

        event = _make_sessionend_event(session_id=session.id)
        cleanup_process_event(event, store, logger)

        events = store.query_events(event_type="hook.session_end")
        assert len(events) == 1

    def test_cleanup_handles_unknown_session(self, store, tmp_config):
        """_process_event with unknown session_id does not raise."""
        logger = MemLogger(tmp_config.log_dir, store.conn)
        event = _make_sessionend_event(session_id="unknown-session-xyz")
        # Should not raise
        cleanup_process_event(event, store, logger)


# ---------------------------------------------------------------------------
# TestIntegration -- 3 tests
# ---------------------------------------------------------------------------
class TestIntegration:
    """Integration tests for the full hook lifecycle."""

    @pytest.mark.integration
    def test_full_lifecycle(self, tmp_config):
        """Run full cycle: context -> capture x2 -> summary -> cleanup."""
        store = SQLiteStore(tmp_config.db_path)
        logger = MemLogger(tmp_config.log_dir, store.conn)
        session_id = "lifecycle-test-001"

        # 1. SessionStart
        with patch.dict(os.environ, {"CLAUDE_PROJECT_DIR": "/test/project"}):
            context_process_event(_make_sessionstart_event(session_id=session_id), store)

        # Verify session exists
        session = store.get_session(session_id)
        assert session is not None
        assert session.project_dir == "/test/project"

        # 2. PostToolUse x2
        with patch.dict(os.environ, {"CLAUDE_PROJECT_DIR": "/test/project"}):
            capture_process_event(
                _make_posttooluse_event(
                    session_id=session_id,
                    tool_name="Write",
                    tool_input={"file_path": "/src/a.py", "content": "code"},
                ),
                store,
            )
            capture_process_event(
                _make_posttooluse_event(
                    session_id=session_id,
                    tool_name="Edit",
                    tool_input={"file_path": "/src/b.py"},
                ),
                store,
            )

        # Verify 2 queue items
        assert store.count_pending(session_id) == 2

        # 3. Stop
        summary_process_event(_make_stop_event(session_id=session_id), store, logger)

        # Verify stop event logged
        stop_events = store.query_events(event_type="hook.stop")
        assert len(stop_events) == 1

        # 4. SessionEnd
        cleanup_process_event(_make_sessionend_event(session_id=session_id), store, logger)

        # Verify session closed
        final_session = store.get_session(session_id)
        assert final_session is not None
        assert final_session.status == SessionStatus.CLOSED
        assert final_session.ended_at is not None

        # Verify session_end event logged
        end_events = store.query_events(event_type="hook.session_end")
        assert len(end_events) == 1

        store.close()

    @pytest.mark.integration
    def test_parallel_captures(self, tmp_config):
        """10 threads each capture a tool event. All 10 items in pending_queue."""
        errors: list[tuple[int, str]] = []
        session_id = "parallel-test-001"

        def capture_thread(thread_id: int) -> None:
            try:
                thread_store = SQLiteStore(tmp_config.db_path)
                event = _make_posttooluse_event(
                    session_id=session_id,
                    tool_name=f"Tool{thread_id}",
                )
                with patch.dict(os.environ, {"CLAUDE_PROJECT_DIR": "/test/project"}):
                    capture_process_event(event, thread_store)
                thread_store.close()
            except Exception as exc:
                errors.append((thread_id, str(exc)))

        threads = [threading.Thread(target=capture_thread, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Thread errors: {errors}"

        # Verify all 10 items exist
        verify_store = SQLiteStore(tmp_config.db_path)
        stats = verify_store.get_queue_stats()
        assert stats.get("raw", 0) == 10
        verify_store.close()

    @pytest.mark.integration
    def test_capture_before_session_start(self, tmp_config):
        """Capture before SessionStart does not FK-violate (no FK on pending_queue)."""
        store = SQLiteStore(tmp_config.db_path)
        session_id = "orphan-capture-001"

        # Capture BEFORE any session is created
        with patch.dict(os.environ, {"CLAUDE_PROJECT_DIR": "/test/project"}):
            event = _make_posttooluse_event(
                session_id=session_id,
                tool_name="Write",
                tool_input={"file_path": "/src/new.py", "content": "code"},
            )
            capture_process_event(event, store)

        # Verify item exists in queue despite no session
        assert store.count_pending(session_id) == 1
        assert store.get_session(session_id) is None

        store.close()
