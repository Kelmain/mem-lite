"""SQLite migration system using PRAGMA user_version."""

import sqlite3

# Each migration is (version, sql). Append-only, sequential.
MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            project_dir TEXT NOT NULL,
            started_at TEXT NOT NULL DEFAULT (datetime('now')),
            ended_at TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            summary TEXT,
            observation_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS observations (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            tool_name TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            detail TEXT,
            files_touched TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(files_touched)),
            functions_changed TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(functions_changed)),
            tokens_raw INTEGER NOT NULL DEFAULT 0,
            tokens_compressed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS function_map (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            qualified_name TEXT NOT NULL,
            kind TEXT NOT NULL,
            signature TEXT NOT NULL,
            docstring TEXT,
            body_hash TEXT NOT NULL,
            decorators TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(decorators)),
            change_type TEXT NOT NULL DEFAULT 'new',
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS call_graph (
            id TEXT PRIMARY KEY,
            caller_file TEXT NOT NULL,
            caller_function TEXT NOT NULL,
            callee_file TEXT NOT NULL,
            callee_function TEXT NOT NULL,
            resolution TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            times_confirmed INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL DEFAULT 'ast',
            session_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS learnings (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            source_session_id TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS pending_queue (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            raw_output TEXT NOT NULL,
            files_touched TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(files_touched)),
            status TEXT NOT NULL DEFAULT 'raw',
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS event_log (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            event_type TEXT NOT NULL,
            data TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(data)),
            duration_ms INTEGER,
            tokens_in INTEGER,
            tokens_out INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_obs_session ON observations(session_id);
        CREATE INDEX IF NOT EXISTS idx_fn_file ON function_map(file_path);
        CREATE INDEX IF NOT EXISTS idx_fn_session ON function_map(session_id);
        CREATE INDEX IF NOT EXISTS idx_fn_change ON function_map(change_type);
        CREATE INDEX IF NOT EXISTS idx_cg_caller ON call_graph(caller_file, caller_function);
        CREATE INDEX IF NOT EXISTS idx_cg_callee ON call_graph(callee_file, callee_function);
        CREATE INDEX IF NOT EXISTS idx_queue_status ON pending_queue(status, created_at);
        CREATE INDEX IF NOT EXISTS idx_log_session ON event_log(session_id);
        CREATE INDEX IF NOT EXISTS idx_log_type ON event_log(event_type);
        CREATE INDEX IF NOT EXISTS idx_log_time ON event_log(created_at);
        """,
    ),
    (
        2,
        """
        ALTER TABLE pending_queue ADD COLUMN priority TEXT NOT NULL DEFAULT 'normal';
        DROP INDEX IF EXISTS idx_queue_status;
        CREATE INDEX IF NOT EXISTS idx_queue_status ON pending_queue(status, priority, created_at);
        """,
    ),
    (
        3,
        """
        ALTER TABLE observations ADD COLUMN embedding_status TEXT DEFAULT 'pending';
        """,
    ),
]

LATEST_VERSION = MIGRATIONS[-1][0]


def get_version(conn: sqlite3.Connection) -> int:
    """Get the current schema version."""
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def migrate(conn: sqlite3.Connection) -> None:
    """Apply all pending migrations sequentially.

    Uses EXCLUSIVE lock to prevent concurrent migration races.
    """
    current = get_version(conn)
    for version, sql in MIGRATIONS:
        if version > current:
            # Acquire exclusive lock, re-check version to handle races
            conn.execute("BEGIN EXCLUSIVE")
            try:
                actual = int(conn.execute("PRAGMA user_version").fetchone()[0])
                if version > actual:
                    for stmt in _split_sql(sql):
                        conn.execute(stmt)
                    conn.execute(f"PRAGMA user_version = {version}")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            current = version


def _split_sql(sql: str) -> list[str]:
    """Split a multi-statement SQL string into individual statements."""
    return [s.strip() for s in sql.strip().split(";") if s.strip()]
