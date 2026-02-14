"""JSONL + SQLite dual-write logger."""

import json
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


class MemLogger:
    """Dual-write logger: JSONL file (source of truth) + SQLite event_log (best-effort)."""

    def __init__(self, log_dir: Path, db_conn: sqlite3.Connection | None = None) -> None:
        self.log_dir = log_dir
        self.db_conn = db_conn
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _log_file(self) -> Path:
        """Current log file (one per day)."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        return self.log_dir / f"claude-mem-{today}.jsonl"

    def log(
        self,
        event_type: str,
        data: dict,
        *,
        session_id: str | None = None,
        duration_ms: int | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> None:
        """Write to JSONL first (source of truth), then SQLite (best-effort).

        JSONL failure propagates. SQLite failure is swallowed.
        """
        entry = {
            "event_type": event_type,
            "data": data,
            "session_id": session_id,
            "duration_ms": duration_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # 1. JSONL — source of truth (propagates on failure)
        with self._log_file.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        # 2. SQLite — best-effort
        if self.db_conn is not None:
            try:
                event_id = str(uuid.uuid4())
                data_json = json.dumps(data)
                self.db_conn.execute(
                    """INSERT INTO event_log
                       (id, session_id, event_type, data, duration_ms, tokens_in, tokens_out)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event_id,
                        session_id,
                        event_type,
                        data_json,
                        duration_ms,
                        tokens_in,
                        tokens_out,
                    ),
                )
            except Exception:
                print(
                    f"WARNING: Failed to write event {event_type} to SQLite",
                    file=sys.stderr,
                )

    @contextmanager
    def timed(self, event_type: str, **kwargs):
        """Context manager that auto-captures duration and status."""
        context = {"status": "started"}
        start = time.monotonic()
        try:
            yield context
            context["status"] = "success"
        except Exception as e:
            context["status"] = "error"
            context["error"] = str(e)
            raise
        finally:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.log(event_type, context, duration_ms=duration_ms, **kwargs)
