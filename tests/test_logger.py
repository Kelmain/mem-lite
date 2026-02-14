"""Tests for MemLogger — 5 tests."""

import json
import time

import pytest

from claude_mem_lite.logging.logger import MemLogger


class TestMemLogger:
    def test_log_writes_valid_jsonl(self, logger, tmp_config):
        """Log entry produces valid JSONL in log file."""
        logger.log("test.event", {"key": "value"})
        log_files = list(tmp_config.log_dir.glob("*.jsonl"))
        assert len(log_files) == 1
        lines = log_files[0].read_text().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event_type"] == "test.event"
        assert entry["data"]["key"] == "value"

    def test_log_writes_to_sqlite(self, logger, store):
        """Log entry also writes to SQLite event_log."""
        logger.log("test.sqlite", {"msg": "hello"}, session_id="sess-1")
        events = store.query_events(event_type="test.sqlite")
        assert len(events) == 1

    def test_timed_captures_duration(self, logger, tmp_config):
        """timed() context manager records duration_ms."""
        with logger.timed("test.timed"):
            time.sleep(0.05)  # 50ms
        log_files = list(tmp_config.log_dir.glob("*.jsonl"))
        entry = json.loads(log_files[0].read_text().splitlines()[-1])
        assert entry["duration_ms"] >= 40  # Allow some tolerance
        assert entry["data"]["status"] == "success"

    def test_timed_captures_error_status(self, logger, tmp_config):
        """timed() records error status on exception."""
        with pytest.raises(ValueError, match="test error"), logger.timed("test.error"):
            raise ValueError("test error")  # noqa: EM101
        log_files = list(tmp_config.log_dir.glob("*.jsonl"))
        entry = json.loads(log_files[0].read_text().splitlines()[-1])
        assert entry["data"]["status"] == "error"
        assert "test error" in entry["data"]["error"]

    def test_file_only_mode(self, tmp_config):
        """Logger works without DB connection (file-only mode)."""
        file_logger = MemLogger(tmp_config.log_dir, db_conn=None)
        file_logger.log("test.fileonly", {"standalone": True})
        log_files = list(tmp_config.log_dir.glob("*.jsonl"))
        assert len(log_files) == 1
        entry = json.loads(log_files[0].read_text().splitlines()[0])
        assert entry["event_type"] == "test.fileonly"
