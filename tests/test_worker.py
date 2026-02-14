"""Tests for worker lifecycle management and processor queue processing."""

from __future__ import annotations

import sqlite3
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from claude_mem_lite.storage.migrations import migrate
from claude_mem_lite.storage.models import CompressedObservation
from claude_mem_lite.storage.sqlite_store import SQLiteStore
from claude_mem_lite.worker.exceptions import RetryableError
from claude_mem_lite.worker.lifecycle import WorkerLifecycle
from claude_mem_lite.worker.processor import IdleTracker, Processor


class TestWorkerLifecycle:
    """Tests for WorkerLifecycle daemon management."""

    def test_start_creates_pid_file(self, tmp_config):
        """Starting the worker writes the subprocess PID to the PID file."""
        lifecycle = WorkerLifecycle(tmp_config)

        mock_proc = MagicMock()
        mock_proc.pid = 12345

        def create_socket(*args, **kwargs):
            """Simulate the worker creating the socket file."""
            tmp_config.socket_path.touch()
            return mock_proc

        with patch(
            "claude_mem_lite.worker.lifecycle.subprocess.Popen",
            side_effect=create_socket,
        ):
            pid = lifecycle.start(daemon=True)

        assert pid == 12345
        assert tmp_config.pid_path.exists()
        assert tmp_config.pid_path.read_text().strip() == "12345"

    def test_stop_sends_sigterm(self, tmp_config):
        """Stopping the worker sends SIGTERM and cleans up PID file."""
        lifecycle = WorkerLifecycle(tmp_config)

        # Start a real process we can actually kill
        proc = subprocess.Popen(["sleep", "60"])
        try:
            tmp_config.pid_path.write_text(str(proc.pid))

            result = lifecycle.stop()

            assert result is True
            assert not tmp_config.pid_path.exists()

            # Verify the process is actually dead
            proc.wait(timeout=5)
            assert proc.returncode is not None
        finally:
            # Safety cleanup in case test fails
            proc.kill()
            proc.wait()

    def test_is_running_detects_dead_process(self, tmp_config):
        """is_running() returns False for a dead PID and cleans up stale files."""
        lifecycle = WorkerLifecycle(tmp_config)

        # Write a PID that almost certainly doesn't exist
        tmp_config.pid_path.write_text("999999")

        assert lifecycle.is_running() is False
        # Stale PID file should be cleaned up
        assert not tmp_config.pid_path.exists()

    def test_stale_file_cleanup(self, tmp_config):
        """Starting with stale PID/socket files cleans them and starts fresh."""
        lifecycle = WorkerLifecycle(tmp_config)

        # Create stale files with a dead PID
        tmp_config.pid_path.write_text("999999")
        tmp_config.socket_path.touch()

        mock_proc = MagicMock()
        mock_proc.pid = 54321

        def create_socket(*args, **kwargs):
            """Simulate the worker creating the socket file."""
            tmp_config.socket_path.touch()
            return mock_proc

        with patch(
            "claude_mem_lite.worker.lifecycle.subprocess.Popen",
            side_effect=create_socket,
        ):
            pid = lifecycle.start(daemon=True)

        assert pid == 54321
        assert tmp_config.pid_path.read_text().strip() == "54321"

    def test_start_idempotent_when_running(self, tmp_config):
        """Calling start() when worker is already running returns existing PID."""
        lifecycle = WorkerLifecycle(tmp_config)

        # Start a real process to simulate an already-running worker
        proc = subprocess.Popen(["sleep", "60"])
        try:
            tmp_config.pid_path.write_text(str(proc.pid))

            # start() should detect the running process and return its PID
            with patch(
                "claude_mem_lite.worker.lifecycle.subprocess.Popen",
            ) as mock_popen:
                pid = lifecycle.start(daemon=True)

            assert pid == proc.pid
            # Popen should NOT have been called (no new process)
            mock_popen.assert_not_called()
        finally:
            proc.kill()
            proc.wait()

    def test_get_pid_no_file(self, tmp_config):
        """get_pid() returns None when no PID file exists."""
        lifecycle = WorkerLifecycle(tmp_config)

        assert lifecycle.get_pid() is None


# ---------------------------------------------------------------------------
# Phase 3: Processor tests
# ---------------------------------------------------------------------------


@pytest.fixture
async def async_db(tmp_config):
    """Async SQLite connection for worker tests."""
    # Run sync migrations first
    sync_conn = sqlite3.connect(str(tmp_config.db_path))
    migrate(sync_conn)
    sync_conn.close()

    db = await aiosqlite.connect(str(tmp_config.db_path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await db.execute("PRAGMA busy_timeout=3000")
    yield db
    await db.close()


@pytest.fixture
def mock_compressor():
    """Mock compressor that returns valid CompressedObservation."""
    comp = MagicMock()
    comp.compress = AsyncMock(
        return_value=CompressedObservation(
            title="Test observation",
            summary="Test summary",
            detail=None,
            files_touched=["test.py"],
            functions_changed=[],
            tokens_in=100,
            tokens_out=50,
        )
    )
    return comp


@pytest.fixture
def idle_tracker():
    """IdleTracker with default timeout."""
    return IdleTracker(timeout_minutes=30)


class TestProcessor:
    """Tests for the Processor queue processing logic."""

    async def test_dequeue_batch_claims_items(
        self, tmp_config, async_db, idle_tracker, mock_compressor
    ):
        """dequeue_batch atomically claims raw items and sets status to processing."""
        # Seed data via sync store
        sync_store = SQLiteStore(tmp_config.db_path)
        session = sync_store.create_session("/test/project")
        sync_store.enqueue(session_id=session.id, tool_name="Read", raw_output="output1")
        sync_store.enqueue(session_id=session.id, tool_name="Write", raw_output="output2")
        sync_store.enqueue(session_id=session.id, tool_name="Bash", raw_output="output3")
        sync_store.close()

        processor = Processor(db=async_db, compressor=mock_compressor, idle_tracker=idle_tracker)
        items = await processor.dequeue_batch()

        assert len(items) == 3
        # Verify all items are now 'processing' in DB
        cursor = await async_db.execute(
            "SELECT status FROM pending_queue WHERE status = 'processing'"
        )
        rows = await cursor.fetchall()
        assert len(rows) == 3

    async def test_process_item_success(self, tmp_config, async_db, idle_tracker, mock_compressor):
        """Successful process_item creates observation and marks queue item done."""
        # Seed data via sync store
        sync_store = SQLiteStore(tmp_config.db_path)
        session = sync_store.create_session("/test/project")
        item = sync_store.enqueue(session_id=session.id, tool_name="Read", raw_output="raw content")
        sync_store.close()

        # Manually set item to 'processing' (simulating dequeue)
        await async_db.execute(
            "UPDATE pending_queue SET status = 'processing' WHERE id = ?", (item.id,)
        )
        await async_db.commit()

        processor = Processor(db=async_db, compressor=mock_compressor, idle_tracker=idle_tracker)
        await processor.process_item(item)

        # Verify observation was created
        cursor = await async_db.execute(
            "SELECT * FROM observations WHERE session_id = ?", (session.id,)
        )
        obs_rows = await cursor.fetchall()
        assert len(obs_rows) == 1
        assert dict(obs_rows[0])["title"] == "Test observation"
        assert dict(obs_rows[0])["summary"] == "Test summary"

        # Verify queue item is marked done
        cursor = await async_db.execute("SELECT status FROM pending_queue WHERE id = ?", (item.id,))
        row = await cursor.fetchone()
        assert dict(row)["status"] == "done"

    async def test_process_item_retryable_error(
        self, tmp_config, async_db, idle_tracker, mock_compressor
    ):
        """RetryableError resets item to raw with incremented attempts."""
        sync_store = SQLiteStore(tmp_config.db_path)
        session = sync_store.create_session("/test/project")
        item = sync_store.enqueue(session_id=session.id, tool_name="Read", raw_output="raw content")
        sync_store.close()

        await async_db.execute(
            "UPDATE pending_queue SET status = 'processing' WHERE id = ?", (item.id,)
        )
        await async_db.commit()

        mock_compressor.compress = AsyncMock(side_effect=RetryableError("API timeout"))
        processor = Processor(db=async_db, compressor=mock_compressor, idle_tracker=idle_tracker)
        await processor.process_item(item)

        cursor = await async_db.execute(
            "SELECT status, attempts FROM pending_queue WHERE id = ?", (item.id,)
        )
        row = dict(await cursor.fetchone())
        assert row["status"] == "raw"
        assert row["attempts"] == 1

    async def test_process_item_max_retries_marks_error(
        self, tmp_config, async_db, idle_tracker, mock_compressor
    ):
        """Item at MAX_ATTEMPTS-1 gets marked error on next RetryableError."""
        sync_store = SQLiteStore(tmp_config.db_path)
        session = sync_store.create_session("/test/project")
        item = sync_store.enqueue(session_id=session.id, tool_name="Read", raw_output="raw content")
        sync_store.close()

        # Set attempts to MAX_ATTEMPTS - 1 = 2
        await async_db.execute(
            "UPDATE pending_queue SET status = 'processing', attempts = 2 WHERE id = ?",
            (item.id,),
        )
        await async_db.commit()

        # Update the item model to reflect current attempts
        item.attempts = 2

        mock_compressor.compress = AsyncMock(side_effect=RetryableError("API timeout"))
        processor = Processor(db=async_db, compressor=mock_compressor, idle_tracker=idle_tracker)
        await processor.process_item(item)

        cursor = await async_db.execute(
            "SELECT status, attempts FROM pending_queue WHERE id = ?", (item.id,)
        )
        row = dict(await cursor.fetchone())
        assert row["status"] == "error"
        assert row["attempts"] == 3

    async def test_recover_orphaned_items(
        self, tmp_config, async_db, idle_tracker, mock_compressor
    ):
        """recover_orphaned_items resets all processing items to raw."""
        sync_store = SQLiteStore(tmp_config.db_path)
        session = sync_store.create_session("/test/project")
        item1 = sync_store.enqueue(session_id=session.id, tool_name="Read", raw_output="output1")
        item2 = sync_store.enqueue(session_id=session.id, tool_name="Write", raw_output="output2")
        sync_store.close()

        # Manually set items to 'processing' (simulating orphaned state)
        await async_db.execute(
            "UPDATE pending_queue SET status = 'processing' WHERE id IN (?, ?)",
            (item1.id, item2.id),
        )
        await async_db.commit()

        processor = Processor(db=async_db, compressor=mock_compressor, idle_tracker=idle_tracker)
        await processor.recover_orphaned_items()

        cursor = await async_db.execute(
            "SELECT status FROM pending_queue WHERE id IN (?, ?)",
            (item1.id, item2.id),
        )
        rows = await cursor.fetchall()
        assert all(dict(r)["status"] == "raw" for r in rows)

    async def test_empty_queue_no_error(self, tmp_config, async_db, idle_tracker, mock_compressor):
        """dequeue_batch on empty queue returns empty list without errors."""
        processor = Processor(db=async_db, compressor=mock_compressor, idle_tracker=idle_tracker)
        items = await processor.dequeue_batch()

        assert items == []

    async def test_idle_tracker_touch(self):
        """touch() sets last_activity to a positive timestamp."""
        tracker = IdleTracker(timeout_minutes=30)
        assert tracker.last_activity == 0.0

        tracker.touch()

        assert tracker.last_activity > 0.0

    async def test_idle_tracker_should_shutdown(self):
        """should_shutdown returns True after shutdown event is set."""
        tracker = IdleTracker(timeout_minutes=0)
        assert tracker.should_shutdown is False

        tracker.touch()
        # Directly set the shutdown event (simulating timeout expiry)
        tracker._shutdown_event.set()

        assert tracker.should_shutdown is True


# ---------------------------------------------------------------------------
# Phase 3: Auto-summarization trigger tests
# ---------------------------------------------------------------------------


class TestAutoSummarization:
    """Tests for auto-summarization trigger in the Processor."""

    async def test_auto_summarize_session_with_stop_event(
        self, tmp_config, async_db, idle_tracker, mock_compressor
    ):
        """Session with hook.stop event, observations, and no pending items gets summarized."""
        # Arrange — seed data via sync store
        sync_store = SQLiteStore(tmp_config.db_path)
        session = sync_store.create_session("/test/project")
        sync_store.create_observation(
            session_id=session.id,
            tool_name="Read",
            title="Obs 1",
            summary="First observation",
        )
        sync_store.create_observation(
            session_id=session.id,
            tool_name="Write",
            title="Obs 2",
            summary="Second observation",
        )
        sync_store.log_event(
            "hook.stop",
            {"event": "hook.stop"},
            session_id=session.id,
        )
        # Backdate the event to 5 minutes ago so debounce passes
        sync_store.conn.execute(
            "UPDATE event_log SET created_at = datetime('now', '-5 minutes') WHERE session_id = ?",
            (session.id,),
        )
        sync_store.close()

        mock_summarizer = AsyncMock()
        mock_summarizer.summarize_session = AsyncMock()

        processor = Processor(
            db=async_db,
            compressor=mock_compressor,
            idle_tracker=idle_tracker,
            summarizer=mock_summarizer,
        )

        # Act
        await processor._check_pending_summaries()

        # Assert
        mock_summarizer.summarize_session.assert_called_once_with(session.id)

    async def test_no_summarize_with_pending_queue(
        self, tmp_config, async_db, idle_tracker, mock_compressor
    ):
        """Session with remaining pending queue items is NOT summarized."""
        # Arrange
        sync_store = SQLiteStore(tmp_config.db_path)
        session = sync_store.create_session("/test/project")
        sync_store.create_observation(
            session_id=session.id,
            tool_name="Read",
            title="Obs 1",
            summary="An observation",
        )
        sync_store.log_event(
            "hook.stop",
            {"event": "hook.stop"},
            session_id=session.id,
        )
        sync_store.conn.execute(
            "UPDATE event_log SET created_at = datetime('now', '-5 minutes') WHERE session_id = ?",
            (session.id,),
        )
        # Add a pending queue item with status='raw'
        sync_store.enqueue(
            session_id=session.id,
            tool_name="Bash",
            raw_output="some output",
        )
        sync_store.close()

        mock_summarizer = AsyncMock()
        mock_summarizer.summarize_session = AsyncMock()

        processor = Processor(
            db=async_db,
            compressor=mock_compressor,
            idle_tracker=idle_tracker,
            summarizer=mock_summarizer,
        )

        # Act
        await processor._check_pending_summaries()

        # Assert
        mock_summarizer.summarize_session.assert_not_called()

    async def test_no_summarize_without_observations(
        self, tmp_config, async_db, idle_tracker, mock_compressor
    ):
        """Session with no observations is NOT summarized even with hook.stop."""
        # Arrange
        sync_store = SQLiteStore(tmp_config.db_path)
        session = sync_store.create_session("/test/project")
        sync_store.log_event(
            "hook.stop",
            {"event": "hook.stop"},
            session_id=session.id,
        )
        sync_store.conn.execute(
            "UPDATE event_log SET created_at = datetime('now', '-5 minutes') WHERE session_id = ?",
            (session.id,),
        )
        sync_store.close()

        mock_summarizer = AsyncMock()
        mock_summarizer.summarize_session = AsyncMock()

        processor = Processor(
            db=async_db,
            compressor=mock_compressor,
            idle_tracker=idle_tracker,
            summarizer=mock_summarizer,
        )

        # Act
        await processor._check_pending_summaries()

        # Assert
        mock_summarizer.summarize_session.assert_not_called()

    async def test_no_summarize_without_summarizer(
        self, tmp_config, async_db, idle_tracker, mock_compressor
    ):
        """Processor with no summarizer returns early without error."""
        # Arrange — processor with summarizer=None (default)
        processor = Processor(
            db=async_db,
            compressor=mock_compressor,
            idle_tracker=idle_tracker,
        )

        # Act + Assert — should not raise
        await processor._check_pending_summaries()
