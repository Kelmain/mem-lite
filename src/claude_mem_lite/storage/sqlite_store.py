"""SQLite storage with WAL mode and CRUD for all 7 tables."""

import json
import sqlite3
import sys
import uuid
from pathlib import Path

from claude_mem_lite.storage.migrations import migrate
from claude_mem_lite.storage.models import (
    CallGraphEdge,
    CallResolution,
    ChangeType,
    EventLogEntry,
    FunctionKind,
    FunctionMapEntry,
    Learning,
    LearningCategory,
    Observation,
    PendingQueueItem,
    Session,
    SessionStatus,
)


class SQLiteStore:
    """Thread-safe SQLite store. Each thread/process should use its own instance."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(
            str(db_path),
            isolation_level=None,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=3000")
        self.conn.execute("PRAGMA wal_autocheckpoint=1000")
        migrate(self.conn)

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def checkpoint(self, mode: str = "PASSIVE") -> None:
        """WAL checkpoint.

        Modes: PASSIVE (non-blocking), RESTART, TRUNCATE.
        """
        assert mode in ("PASSIVE", "RESTART", "TRUNCATE"), f"Invalid mode: {mode}"
        self.conn.execute(f"PRAGMA wal_checkpoint({mode})")

    # -----------------------------------------------------------------------
    # Sessions
    # -----------------------------------------------------------------------
    def create_session(self, project_dir: str, *, id: str | None = None) -> Session:
        """Create a new session. Idempotent if id already exists (INSERT OR IGNORE)."""
        session_id = id or str(uuid.uuid4())
        self.conn.execute(
            "INSERT OR IGNORE INTO sessions (id, project_dir) VALUES (?, ?)",
            (session_id, project_dir),
        )
        session = self.get_session(session_id)
        assert session is not None
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get session by ID."""
        row = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return Session(**dict(row))

    def update_session(
        self,
        session_id: str,
        *,
        status: SessionStatus | None = None,
        summary: str | None = None,
        ended_at: str | None = None,
    ) -> None:
        """Update session fields."""
        updates = []
        params = []
        if status is not None:
            updates.append("status = ?")
            params.append(str(status))
        if summary is not None:
            updates.append("summary = ?")
            params.append(summary)
        if ended_at is not None:
            updates.append("ended_at = ?")
            params.append(ended_at)
        if status == SessionStatus.CLOSED and ended_at is None:
            updates.append("ended_at = datetime('now')")
        if updates:
            params.append(session_id)
            self.conn.execute(
                f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?",
                params,
            )

    def list_sessions(
        self,
        status: SessionStatus | None = None,
        *,
        limit: int | None = None,
        status_filter: SessionStatus | None = None,
    ) -> list[Session]:
        """List sessions ordered by started_at DESC, optionally filtered by status."""
        effective_status = status_filter or status
        query = "SELECT * FROM sessions"
        params: list = []
        if effective_status is not None:
            query += " WHERE status = ?"
            params.append(str(effective_status))
        query += " ORDER BY started_at DESC, rowid DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [Session(**dict(r)) for r in rows]

    # -----------------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------------
    def create_observation(
        self,
        *,
        session_id: str,
        tool_name: str,
        title: str,
        summary: str,
        detail: str | None = None,
        files_touched: list[str] | None = None,
        functions_changed: list[str] | None = None,
        tokens_raw: int = 0,
        tokens_compressed: int = 0,
    ) -> Observation:
        """Create an observation and increment session observation_count."""
        obs_id = str(uuid.uuid4())
        files_json = json.dumps(files_touched or [])
        funcs_json = json.dumps(functions_changed or [])
        self.conn.execute(
            """INSERT INTO observations
               (id, session_id, tool_name, title, summary, detail,
                files_touched, functions_changed, tokens_raw, tokens_compressed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                obs_id,
                session_id,
                tool_name,
                title,
                summary,
                detail,
                files_json,
                funcs_json,
                tokens_raw,
                tokens_compressed,
            ),
        )
        self.conn.execute(
            "UPDATE sessions SET observation_count = observation_count + 1 WHERE id = ?",
            (session_id,),
        )
        obs = self.get_observation(obs_id)
        assert obs is not None  # Just inserted
        return obs

    def get_observation(self, obs_id: str) -> Observation | None:
        """Get observation by ID."""
        row = self.conn.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
        if row is None:
            return None
        return Observation(**dict(row))

    def list_observations_by_session(self, session_id: str) -> list[Observation]:
        """List observations for a session ordered by created_at."""
        rows = self.conn.execute(
            "SELECT * FROM observations WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [Observation(**dict(r)) for r in rows]

    # -----------------------------------------------------------------------
    # Function map
    # -----------------------------------------------------------------------
    def upsert_function_map(
        self,
        *,
        session_id: str,
        file_path: str,
        qualified_name: str,
        kind: FunctionKind,
        signature: str,
        body_hash: str,
        change_type: ChangeType = ChangeType.NEW,
        docstring: str | None = None,
        decorators: list[str] | None = None,
    ) -> FunctionMapEntry:
        """Insert or update a function map entry (by file_path + qualified_name)."""
        decorators_json = json.dumps(decorators or [])
        existing = self.conn.execute(
            "SELECT id FROM function_map WHERE file_path = ? AND qualified_name = ?",
            (file_path, qualified_name),
        ).fetchone()
        if existing:
            self.conn.execute(
                """UPDATE function_map SET
                       session_id = ?, kind = ?, signature = ?, docstring = ?,
                       body_hash = ?, decorators = ?, change_type = ?,
                       updated_at = datetime('now')
                   WHERE id = ?""",
                (
                    session_id,
                    str(kind),
                    signature,
                    docstring,
                    body_hash,
                    decorators_json,
                    str(change_type),
                    existing["id"],
                ),
            )
            entry_id = existing["id"]
        else:
            entry_id = str(uuid.uuid4())
            self.conn.execute(
                """INSERT INTO function_map
                   (id, session_id, file_path, qualified_name, kind, signature,
                    docstring, body_hash, decorators, change_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry_id,
                    session_id,
                    file_path,
                    qualified_name,
                    str(kind),
                    signature,
                    docstring,
                    body_hash,
                    decorators_json,
                    str(change_type),
                ),
            )
        row = self.conn.execute("SELECT * FROM function_map WHERE id = ?", (entry_id,)).fetchone()
        return FunctionMapEntry(**dict(row))

    def get_latest_functions(self, file_path: str) -> list[FunctionMapEntry]:
        """Get latest function entries for a file."""
        rows = self.conn.execute(
            "SELECT * FROM function_map WHERE file_path = ? ORDER BY updated_at DESC",
            (file_path,),
        ).fetchall()
        return [FunctionMapEntry(**dict(r)) for r in rows]

    def get_changed_functions(self, session_id: str) -> list[FunctionMapEntry]:
        """Get functions changed in a session (excludes 'unchanged')."""
        rows = self.conn.execute(
            "SELECT * FROM function_map WHERE session_id = ? AND change_type != 'unchanged'",
            (session_id,),
        ).fetchall()
        return [FunctionMapEntry(**dict(r)) for r in rows]

    # -----------------------------------------------------------------------
    # Call graph
    # -----------------------------------------------------------------------
    def add_call_graph_edge(
        self,
        *,
        caller_file: str,
        caller_function: str,
        callee_file: str,
        callee_function: str,
        resolution: CallResolution,
        session_id: str,
        confidence: float = 1.0,
        source: str = "ast",
    ) -> CallGraphEdge:
        """Add a call graph edge."""
        edge_id = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO call_graph
               (id, caller_file, caller_function, callee_file, callee_function,
                resolution, confidence, source, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                edge_id,
                caller_file,
                caller_function,
                callee_file,
                callee_function,
                str(resolution),
                confidence,
                source,
                session_id,
            ),
        )
        row = self.conn.execute("SELECT * FROM call_graph WHERE id = ?", (edge_id,)).fetchone()
        return CallGraphEdge(**dict(row))

    def get_callers(self, file_path: str, function: str) -> list[CallGraphEdge]:
        """Get all edges calling into this function."""
        rows = self.conn.execute(
            "SELECT * FROM call_graph WHERE callee_file = ? AND callee_function = ?",
            (file_path, function),
        ).fetchall()
        return [CallGraphEdge(**dict(r)) for r in rows]

    def get_callees(self, file_path: str, function: str) -> list[CallGraphEdge]:
        """Get all edges called by this function."""
        rows = self.conn.execute(
            "SELECT * FROM call_graph WHERE caller_file = ? AND caller_function = ?",
            (file_path, function),
        ).fetchall()
        return [CallGraphEdge(**dict(r)) for r in rows]

    # -----------------------------------------------------------------------
    # Learnings
    # -----------------------------------------------------------------------
    def create_learning(
        self,
        *,
        category: LearningCategory,
        content: str,
        source_session_id: str,
        confidence: float = 0.5,
    ) -> Learning:
        """Create a new learning."""
        learning_id = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO learnings (id, category, content, confidence, source_session_id)
               VALUES (?, ?, ?, ?, ?)""",
            (learning_id, str(category), content, confidence, source_session_id),
        )
        row = self.conn.execute("SELECT * FROM learnings WHERE id = ?", (learning_id,)).fetchone()
        data = dict(row)
        data["is_active"] = bool(data["is_active"])
        return Learning(**data)

    def get_active_learnings(
        self,
        category: LearningCategory | None = None,
        min_confidence: float = 0.0,
    ) -> list[Learning]:
        """Get active learnings, optionally filtered."""
        query = "SELECT * FROM learnings WHERE is_active = 1 AND confidence >= ?"
        params: list = [min_confidence]
        if category is not None:
            query += " AND category = ?"
            params.append(str(category))
        query += " ORDER BY confidence DESC"
        rows = self.conn.execute(query, params).fetchall()
        result = []
        for r in rows:
            data = dict(r)
            data["is_active"] = bool(data["is_active"])
            result.append(Learning(**data))
        return result

    def update_learning(
        self,
        learning_id: str,
        *,
        confidence: float | None = None,
        content: str | None = None,
    ) -> None:
        """Update learning fields."""
        updates = ["updated_at = datetime('now')"]
        params: list = []
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        params.append(learning_id)
        self.conn.execute(
            f"UPDATE learnings SET {', '.join(updates)} WHERE id = ?",
            params,
        )

    def soft_delete_learning(self, learning_id: str) -> None:
        """Soft delete a learning (set is_active=0)."""
        self.conn.execute(
            "UPDATE learnings SET is_active = 0, updated_at = datetime('now') WHERE id = ?",
            (learning_id,),
        )

    # -----------------------------------------------------------------------
    # Pending queue
    # -----------------------------------------------------------------------
    def enqueue(
        self,
        *,
        session_id: str,
        tool_name: str,
        raw_output: str,
        files_touched: list[str] | None = None,
        priority: str = "normal",
    ) -> PendingQueueItem:
        """Add item to the processing queue."""
        item_id = str(uuid.uuid4())
        files_json = json.dumps(files_touched or [])
        self.conn.execute(
            """INSERT INTO pending_queue (id, session_id, tool_name, raw_output, files_touched, priority)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (item_id, session_id, tool_name, raw_output, files_json, priority),
        )
        row = self.conn.execute("SELECT * FROM pending_queue WHERE id = ?", (item_id,)).fetchone()
        return PendingQueueItem(**dict(row))

    def dequeue_batch(self, n: int = 5) -> list[PendingQueueItem]:
        """Atomically claim N pending items for processing.

        Uses BEGIN IMMEDIATE + UPDATE ... RETURNING for single-statement atomicity.
        """
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self.conn.execute(
                """
                UPDATE pending_queue SET status = 'processing'
                WHERE status = 'raw'
                  AND id IN (
                    SELECT id FROM pending_queue
                    WHERE status = 'raw'
                    ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'normal' THEN 1 WHEN 'low' THEN 2 ELSE 1 END, created_at
                    LIMIT ?
                  )
                RETURNING *
                """,
                (n,),
            ).fetchall()
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return [PendingQueueItem(**dict(r)) for r in rows]

    def complete_queue_item(self, item_id: str) -> None:
        """Mark a queue item as done."""
        self.conn.execute(
            "UPDATE pending_queue SET status = 'done' WHERE id = ?",
            (item_id,),
        )

    def fail_queue_item(self, item_id: str) -> None:
        """Mark a queue item as error and increment attempts."""
        self.conn.execute(
            "UPDATE pending_queue SET status = 'error', attempts = attempts + 1 WHERE id = ?",
            (item_id,),
        )

    def retry_failed(self, max_attempts: int = 3) -> int:
        """Reset error items with attempts < max_attempts back to raw. Returns count."""
        cursor = self.conn.execute(
            "UPDATE pending_queue SET status = 'raw' WHERE status = 'error' AND attempts < ?",
            (max_attempts,),
        )
        return cursor.rowcount

    def get_queue_stats(self) -> dict[str, int]:
        """Get counts by queue status."""
        rows = self.conn.execute(
            "SELECT status, COUNT(*) as count FROM pending_queue GROUP BY status"
        ).fetchall()
        return {row["status"]: row["count"] for row in rows}

    def count_pending(self, session_id: str) -> int:
        """Count pending queue items for a session."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM pending_queue WHERE session_id = ? AND status = 'raw'",
            (session_id,),
        ).fetchone()
        return int(row["cnt"])

    # -----------------------------------------------------------------------
    # Event log (best-effort — never raises)
    # -----------------------------------------------------------------------
    def log_event(
        self,
        event_type: str,
        data: dict | None = None,
        *,
        session_id: str | None = None,
        duration_ms: int | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> None:
        """Log event to SQLite. Best-effort: swallows errors."""
        try:
            event_id = str(uuid.uuid4())
            data_json = json.dumps(data or {})
            self.conn.execute(
                """INSERT INTO event_log
                   (id, session_id, event_type, data, duration_ms, tokens_in, tokens_out)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (event_id, session_id, event_type, data_json, duration_ms, tokens_in, tokens_out),
            )
        except Exception:
            print(f"WARNING: Failed to log event {event_type}", file=sys.stderr)

    def query_events(
        self,
        event_type: str | None = None,
        session_id: str | None = None,
    ) -> list[EventLogEntry]:
        """Query events by type and/or session."""
        query = "SELECT * FROM event_log WHERE 1=1"
        params: list = []
        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type)
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at"
        rows = self.conn.execute(query, params).fetchall()
        return [EventLogEntry(**dict(r)) for r in rows]
