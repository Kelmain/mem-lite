"""Tests for storage layer — 31 tests (migrations, CRUD, concurrency)."""

import json
import sqlite3
import threading

import pytest

from claude_mem_lite.storage.migrations import LATEST_VERSION, MIGRATIONS, get_version, migrate
from claude_mem_lite.storage.models import (
    CallResolution,
    ChangeType,
    FunctionKind,
    LearningCategory,
    QueueStatus,
    SessionStatus,
)
from claude_mem_lite.storage.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Migrations — 7 tests
# ---------------------------------------------------------------------------
class TestMigrations:
    def test_fresh_db_creates_all_tables(self, store):
        """Fresh DB migration creates all 7 tables."""
        tables = {
            row["name"]
            for row in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected = {
            "sessions",
            "observations",
            "function_map",
            "call_graph",
            "learnings",
            "pending_queue",
            "event_log",
        }
        assert expected.issubset(tables)

    def test_version_set_after_migration(self, store):
        """PRAGMA user_version is set to latest after migration."""
        version = store.conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == LATEST_VERSION

    def test_idempotent_migration(self, store):
        """Running migrate twice does not raise or duplicate tables."""
        # Should not raise
        migrate(store.conn)

    def test_wal_mode_enabled(self, store):
        """Database uses WAL journal mode."""
        mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_enabled(self, store):
        """Foreign keys are enforced."""
        fk = store.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_indexes_created(self, store):
        """All expected indexes exist."""
        indexes = {
            row["name"]
            for row in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            ).fetchall()
        }
        expected = {
            "idx_obs_session",
            "idx_fn_file",
            "idx_fn_session",
            "idx_fn_change",
            "idx_cg_caller",
            "idx_cg_callee",
            "idx_queue_status",
            "idx_log_session",
            "idx_log_type",
            "idx_log_time",
        }
        assert expected.issubset(indexes)

    def test_version_increments(self, tmp_config):
        """Each migration increments user_version."""
        conn = sqlite3.connect(str(tmp_config.db_path))
        conn.row_factory = sqlite3.Row
        assert get_version(conn) == 0
        migrate(conn)
        assert get_version(conn) == MIGRATIONS[-1][0]
        conn.close()


# ---------------------------------------------------------------------------
# Sessions CRUD — 5 tests
# ---------------------------------------------------------------------------
class TestSessionsCRUD:
    def test_create_and_get_session(self, store):
        """Create a session and retrieve it by ID."""
        session = store.create_session(project_dir="/tmp/project")
        fetched = store.get_session(session.id)
        assert fetched is not None
        assert fetched.project_dir == "/tmp/project"
        assert fetched.status == SessionStatus.ACTIVE

    def test_update_session_status(self, store):
        """Update session status and ended_at."""
        session = store.create_session(project_dir="/tmp/project")
        store.update_session(session.id, status=SessionStatus.CLOSED)
        fetched = store.get_session(session.id)
        assert fetched.status == SessionStatus.CLOSED

    def test_list_sessions_ordered(self, store):
        """List sessions ordered by started_at DESC."""
        store.create_session(project_dir="/tmp/a")
        s2 = store.create_session(project_dir="/tmp/b")
        sessions = store.list_sessions()
        assert len(sessions) == 2
        # Most recent first
        assert sessions[0].id == s2.id

    def test_list_sessions_filter_by_status(self, store):
        """Filter sessions by status."""
        s1 = store.create_session(project_dir="/tmp/a")
        s2 = store.create_session(project_dir="/tmp/b")
        store.update_session(s1.id, status=SessionStatus.CLOSED)
        active = store.list_sessions(status=SessionStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].id == s2.id

    def test_session_observation_count(self, store):
        """observation_count increments when observations are added."""
        session = store.create_session(project_dir="/tmp/project")
        store.create_observation(
            session_id=session.id,
            tool_name="Write",
            title="Created file",
            summary="Created src/main.py",
        )
        fetched = store.get_session(session.id)
        assert fetched.observation_count == 1


# ---------------------------------------------------------------------------
# Observations CRUD — 2 tests
# ---------------------------------------------------------------------------
class TestObservationsCRUD:
    def test_create_observation_with_metadata(self, store):
        """Create observation with files_touched and tokens."""
        session = store.create_session(project_dir="/tmp/project")
        obs = store.create_observation(
            session_id=session.id,
            tool_name="Edit",
            title="Edited config",
            summary="Updated database settings",
            files_touched=["src/config.py", "tests/test_config.py"],
            tokens_raw=5000,
            tokens_compressed=150,
        )
        fetched = store.get_observation(obs.id)
        assert fetched.tool_name == "Edit"
        assert json.loads(fetched.files_touched) == ["src/config.py", "tests/test_config.py"]
        assert fetched.tokens_raw == 5000

    def test_list_observations_by_session(self, store):
        """List observations filtered by session_id."""
        s1 = store.create_session(project_dir="/tmp/a")
        s2 = store.create_session(project_dir="/tmp/b")
        store.create_observation(session_id=s1.id, tool_name="Write", title="t1", summary="s1")
        store.create_observation(session_id=s1.id, tool_name="Read", title="t2", summary="s2")
        store.create_observation(session_id=s2.id, tool_name="Bash", title="t3", summary="s3")
        obs = store.list_observations_by_session(s1.id)
        assert len(obs) == 2


# ---------------------------------------------------------------------------
# Function map — 2 tests
# ---------------------------------------------------------------------------
class TestFunctionMap:
    def test_upsert_function_map(self, store):
        """Upsert creates new entry and updates on change."""
        session = store.create_session(project_dir="/tmp/project")
        entry = store.upsert_function_map(
            session_id=session.id,
            file_path="src/main.py",
            qualified_name="main",
            kind=FunctionKind.FUNCTION,
            signature="def main() -> None",
            body_hash="abc123",
            change_type=ChangeType.NEW,
        )
        assert entry.qualified_name == "main"
        # Update with new hash
        updated = store.upsert_function_map(
            session_id=session.id,
            file_path="src/main.py",
            qualified_name="main",
            kind=FunctionKind.FUNCTION,
            signature="def main(args: list) -> None",
            body_hash="def456",
            change_type=ChangeType.MODIFIED,
        )
        assert updated.body_hash == "def456"

    def test_get_changed_functions(self, store):
        """get_changed_functions excludes unchanged entries."""
        session = store.create_session(project_dir="/tmp/project")
        store.upsert_function_map(
            session_id=session.id,
            file_path="src/a.py",
            qualified_name="func_a",
            kind=FunctionKind.FUNCTION,
            signature="def func_a()",
            body_hash="aaa",
            change_type=ChangeType.NEW,
        )
        store.upsert_function_map(
            session_id=session.id,
            file_path="src/b.py",
            qualified_name="func_b",
            kind=FunctionKind.FUNCTION,
            signature="def func_b()",
            body_hash="bbb",
            change_type=ChangeType.UNCHANGED,
        )
        changed = store.get_changed_functions(session.id)
        assert len(changed) == 1
        assert changed[0].qualified_name == "func_a"


# ---------------------------------------------------------------------------
# Call graph — 1 test
# ---------------------------------------------------------------------------
class TestCallGraph:
    def test_add_and_query_edges(self, store):
        """Add edges and query callers/callees."""
        session = store.create_session(project_dir="/tmp/project")
        store.add_call_graph_edge(
            caller_file="src/a.py",
            caller_function="func_a",
            callee_file="src/b.py",
            callee_function="func_b",
            resolution=CallResolution.DIRECT,
            session_id=session.id,
        )
        store.add_call_graph_edge(
            caller_file="src/a.py",
            caller_function="func_a",
            callee_file="src/c.py",
            callee_function="func_c",
            resolution=CallResolution.IMPORT,
            session_id=session.id,
        )
        callees = store.get_callees("src/a.py", "func_a")
        assert len(callees) == 2
        callers = store.get_callers("src/b.py", "func_b")
        assert len(callers) == 1
        assert callers[0].caller_function == "func_a"


# ---------------------------------------------------------------------------
# Learnings — 5 tests
# ---------------------------------------------------------------------------
class TestLearnings:
    def test_create_and_list_learnings(self, store):
        """Create a learning and list all active."""
        session = store.create_session(project_dir="/tmp/project")
        store.create_learning(
            category=LearningCategory.CONVENTION,
            content="Always use UTC datetimes",
            source_session_id=session.id,
        )
        active = store.get_active_learnings()
        assert len(active) == 1
        assert active[0].content == "Always use UTC datetimes"

    def test_filter_by_category(self, store):
        """Filter learnings by category."""
        session = store.create_session(project_dir="/tmp/project")
        store.create_learning(
            category=LearningCategory.CONVENTION,
            content="Use UTC",
            source_session_id=session.id,
        )
        store.create_learning(
            category=LearningCategory.GOTCHA,
            content="SQLite busy timeout",
            source_session_id=session.id,
        )
        conventions = store.get_active_learnings(category=LearningCategory.CONVENTION)
        assert len(conventions) == 1

    def test_confidence_threshold(self, store):
        """Filter learnings by minimum confidence."""
        session = store.create_session(project_dir="/tmp/project")
        store.create_learning(
            category=LearningCategory.PATTERN,
            content="High confidence",
            source_session_id=session.id,
            confidence=0.9,
        )
        store.create_learning(
            category=LearningCategory.PATTERN,
            content="Low confidence",
            source_session_id=session.id,
            confidence=0.2,
        )
        high = store.get_active_learnings(min_confidence=0.5)
        assert len(high) == 1
        assert high[0].content == "High confidence"

    def test_update_learning(self, store):
        """Update learning confidence."""
        session = store.create_session(project_dir="/tmp/project")
        learning = store.create_learning(
            category=LearningCategory.ARCHITECTURE,
            content="Use repository pattern",
            source_session_id=session.id,
        )
        store.update_learning(learning.id, confidence=0.95)
        fetched = store.get_active_learnings()
        assert fetched[0].confidence == pytest.approx(0.95)

    def test_soft_delete_learning(self, store):
        """Soft delete excludes from active learnings."""
        session = store.create_session(project_dir="/tmp/project")
        learning = store.create_learning(
            category=LearningCategory.DEPENDENCY,
            content="Use pydantic v2",
            source_session_id=session.id,
        )
        store.soft_delete_learning(learning.id)
        active = store.get_active_learnings()
        assert len(active) == 0


# ---------------------------------------------------------------------------
# Pending queue — 5 tests
# ---------------------------------------------------------------------------
class TestPendingQueue:
    def test_enqueue_and_dequeue(self, store):
        """Enqueue item and dequeue it atomically."""
        session = store.create_session(project_dir="/tmp/project")
        store.enqueue(
            session_id=session.id,
            tool_name="Write",
            raw_output="file content here" * 100,
        )
        batch = store.dequeue_batch(n=5)
        assert len(batch) == 1
        assert batch[0].status == QueueStatus.PROCESSING

    def test_dequeue_respects_limit(self, store):
        """dequeue_batch returns at most N items."""
        session = store.create_session(project_dir="/tmp/project")
        for i in range(10):
            store.enqueue(
                session_id=session.id,
                tool_name="Write",
                raw_output=f"output {i}",
            )
        batch = store.dequeue_batch(n=3)
        assert len(batch) == 3
        # Remaining should still be raw
        stats = store.get_queue_stats()
        assert stats["raw"] == 7

    def test_complete_queue_item(self, store):
        """Mark dequeued item as done."""
        session = store.create_session(project_dir="/tmp/project")
        store.enqueue(
            session_id=session.id,
            tool_name="Edit",
            raw_output="diff output",
        )
        batch = store.dequeue_batch(n=1)
        store.complete_queue_item(batch[0].id)
        stats = store.get_queue_stats()
        assert stats.get("raw", 0) == 0
        assert stats["done"] == 1

    def test_retry_failed(self, store):
        """retry_failed resets error items with attempts < max."""
        session = store.create_session(project_dir="/tmp/project")
        store.enqueue(
            session_id=session.id,
            tool_name="Bash",
            raw_output="command output",
        )
        batch = store.dequeue_batch(n=1)
        store.fail_queue_item(batch[0].id)
        store.retry_failed(max_attempts=3)
        stats = store.get_queue_stats()
        assert stats.get("raw", 0) == 1  # Reset to raw for retry

    def test_queue_stats(self, store):
        """get_queue_stats returns counts by status."""
        session = store.create_session(project_dir="/tmp/project")
        store.enqueue(session_id=session.id, tool_name="A", raw_output="a")
        store.enqueue(session_id=session.id, tool_name="B", raw_output="b")
        batch = store.dequeue_batch(n=1)
        store.complete_queue_item(batch[0].id)
        stats = store.get_queue_stats()
        assert stats["raw"] == 1
        assert stats["done"] == 1


# ---------------------------------------------------------------------------
# Event log — 2 tests
# ---------------------------------------------------------------------------
class TestEventLog:
    def test_log_and_query_events(self, store):
        """Log event and query by type and session."""
        session = store.create_session(project_dir="/tmp/project")
        store.log_event(
            event_type="hook.capture",
            data={"tool": "Write", "size": 1024},
            session_id=session.id,
            duration_ms=45,
        )
        events = store.query_events(event_type="hook.capture")
        assert len(events) == 1
        assert json.loads(events[0].data)["tool"] == "Write"
        assert events[0].duration_ms == 45

    def test_event_log_never_raises(self, store):
        """Event logging is best-effort — swallows errors."""
        # Close connection to force error
        store.conn.close()
        # Should not raise
        store.log_event(event_type="test.error", data={"msg": "should not crash"})


# ---------------------------------------------------------------------------
# Concurrency — 2 tests
# ---------------------------------------------------------------------------
class TestConcurrency:
    def test_wal_concurrent_read_write(self, tmp_config):
        """WAL allows concurrent reads while writing."""
        writer = SQLiteStore(tmp_config.db_path)
        reader = SQLiteStore(tmp_config.db_path)

        writer.create_session(project_dir="/tmp/concurrent")
        # Reader can list while writer has an open session
        sessions = reader.list_sessions()
        assert len(sessions) == 1
        writer.close()
        reader.close()

    def test_threaded_writes_no_errors(self, tmp_config):
        """10 threads writing concurrently produce zero errors."""
        errors = []

        def writer_thread(thread_id):
            try:
                s = SQLiteStore(tmp_config.db_path)
                session = s.create_session(project_dir=f"/tmp/thread-{thread_id}")
                for i in range(5):
                    s.enqueue(
                        session_id=session.id,
                        tool_name=f"tool-{thread_id}-{i}",
                        raw_output=f"output-{thread_id}-{i}",
                    )
                s.close()
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = [threading.Thread(target=writer_thread, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Thread errors: {errors}"
        # Verify all data written
        s = SQLiteStore(tmp_config.db_path)
        sessions = s.list_sessions()
        assert len(sessions) == 10
        stats = s.get_queue_stats()
        assert stats.get("raw", 0) == 50
        s.close()
