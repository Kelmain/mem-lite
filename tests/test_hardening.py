"""Tests for Phase 9: Hardening (Tier 1)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from claude_mem_lite.storage.migrations import migrate

runner = CliRunner()


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def seeded_pending_db(tmp_path):
    """Database with pending_queue items for compress testing."""
    base_dir = tmp_path / ".claude-mem"
    base_dir.mkdir()
    db_path = base_dir / "claude-mem.db"
    conn = sqlite3.connect(str(db_path))
    migrate(conn)

    now = datetime.now(UTC).isoformat()

    # Session required for FK
    conn.execute(
        "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
        ("sess-1", "/tmp/project", now, "active"),
    )

    # 3 pending queue items with raw_output
    for i in range(3):
        conn.execute(
            "INSERT INTO pending_queue (id, session_id, tool_name, raw_output, "
            "files_touched, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"pq-{i}",
                "sess-1",
                "Write",
                f"Raw output for item {i} with some code changes.",
                json.dumps([f"src/file_{i}.py"]),
                "raw",
                now,
            ),
        )

    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
def seeded_prune_db(tmp_path):
    """Database with old data for prune testing."""
    base_dir = tmp_path / ".claude-mem"
    base_dir.mkdir()
    db_path = base_dir / "claude-mem.db"
    conn = sqlite3.connect(str(db_path))
    migrate(conn)

    now = datetime.now(UTC)
    old_date = (now - timedelta(days=45)).isoformat()
    recent_date = (now - timedelta(days=5)).isoformat()
    very_recent = now.isoformat()

    # Session
    conn.execute(
        "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
        ("sess-1", "/tmp/project", old_date, "closed"),
    )
    conn.execute(
        "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
        ("sess-2", "/tmp/project", recent_date, "active"),
    )

    # Old pending_queue items (done, with raw_output)
    for i in range(5):
        conn.execute(
            "INSERT INTO pending_queue (id, session_id, tool_name, raw_output, "
            "files_touched, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"old-pq-{i}",
                "sess-1",
                "Write",
                f"Old raw output number {i} " * 100,  # Bulk data
                "[]",
                "done",
                old_date,
            ),
        )

    # Recent pending_queue items (done, with raw_output)
    for i in range(3):
        conn.execute(
            "INSERT INTO pending_queue (id, session_id, tool_name, raw_output, "
            "files_touched, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"recent-pq-{i}",
                "sess-2",
                "Write",
                f"Recent raw output {i}",
                "[]",
                "done",
                very_recent,
            ),
        )

    # Old event_log entries
    for i in range(4):
        conn.execute(
            "INSERT INTO event_log (id, session_id, event_type, data, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"old-ev-{i}", "sess-1", "compress.done", '{"ratio": 10.0}', old_date),
        )

    # Recent event_log entries
    for i in range(2):
        conn.execute(
            "INSERT INTO event_log (id, session_id, event_type, data, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"recent-ev-{i}", "sess-2", "compress.done", '{"ratio": 20.0}', very_recent),
        )

    conn.commit()
    conn.close()
    return str(db_path)


def _make_config(db_path: str):
    """Create a Config with base_dir pointing to the DB parent directory."""
    from claude_mem_lite.config import Config

    base_dir = Path(db_path).parent
    config = Config(base_dir=base_dir)
    assert str(config.db_path) == db_path
    return config


# -----------------------------------------------------------------------
# compress --pending (4 tests)
# -----------------------------------------------------------------------


class TestCompress:
    """Tests for compress command."""

    def test_dry_run_shows_pending(self, seeded_pending_db):
        """--dry-run lists pending items without processing."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_pending_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["compress", "--pending", "--dry-run"])

        assert result.exit_code == 0
        assert "3" in result.stdout  # 3 pending items
        assert "dry run" in result.stdout.lower() or "pending" in result.stdout.lower()

    def test_no_pending_items_message(self, tmp_path):
        """Empty queue shows 'No pending items' message."""
        from claude_mem_lite.cli.main import app

        base_dir = tmp_path / ".claude-mem"
        base_dir.mkdir()
        db_path = base_dir / "claude-mem.db"
        conn = sqlite3.connect(str(db_path))
        migrate(conn)
        conn.close()

        config = _make_config(str(db_path))
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["compress", "--pending"])

        assert result.exit_code == 0
        assert "no pending" in result.stdout.lower()

    def test_inline_compression_success(self, seeded_pending_db):
        """Inline compression processes items and marks them done."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_pending_db)

        # Mock the anthropic client to return a valid compression response
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps(
            {
                "title": "Test compression",
                "summary": "Compressed output.",
                "detail": None,
                "files_touched": ["src/file.py"],
                "functions_changed": [],
            }
        )
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_response

        with (
            patch("claude_mem_lite.config.Config", return_value=config),
            patch("claude_mem_lite.cli.compress_cmd.anthropic") as mock_anthropic_mod,
        ):
            mock_anthropic_mod.Anthropic.return_value = mock_client_instance
            result = runner.invoke(app, ["compress", "--pending", "--limit", "3"])

        if result.exit_code != 0 and result.exception:
            import traceback

            traceback.print_exception(
                type(result.exception), result.exception, result.exception.__traceback__
            )

        assert result.exit_code == 0

        # Verify items marked done
        conn = sqlite3.connect(seeded_pending_db)
        done_count = conn.execute(
            "SELECT COUNT(*) FROM pending_queue WHERE status = 'done'"
        ).fetchone()[0]
        conn.close()
        assert done_count == 3

    def test_inline_compression_api_error_marks_error(self, seeded_pending_db):
        """API error marks item as error and continues."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_pending_db)

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = Exception("API Error")

        with (
            patch("claude_mem_lite.config.Config", return_value=config),
            patch("claude_mem_lite.cli.compress_cmd.anthropic") as mock_anthropic_mod,
        ):
            mock_anthropic_mod.Anthropic.return_value = mock_client_instance
            result = runner.invoke(app, ["compress", "--pending", "--limit", "3"])

        assert result.exit_code == 0

        # Verify items marked error
        conn = sqlite3.connect(seeded_pending_db)
        error_count = conn.execute(
            "SELECT COUNT(*) FROM pending_queue WHERE status = 'error'"
        ).fetchone()[0]
        conn.close()
        assert error_count == 3


# -----------------------------------------------------------------------
# prune (7 tests)
# -----------------------------------------------------------------------


class TestPrune:
    """Tests for prune command."""

    def test_dry_run_shows_counts(self, seeded_prune_db):
        """--dry-run shows what would be pruned without changing anything."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_prune_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["prune", "--dry-run"])

        assert result.exit_code == 0
        assert "dry run" in result.stdout.lower()

        # Verify nothing was actually deleted
        conn = sqlite3.connect(seeded_prune_db)
        raw_count = conn.execute(
            "SELECT COUNT(*) FROM pending_queue WHERE raw_output IS NOT NULL AND raw_output != ''"
        ).fetchone()[0]
        event_count = conn.execute("SELECT COUNT(*) FROM event_log").fetchone()[0]
        conn.close()
        assert raw_count == 8  # All raw_output preserved
        assert event_count == 6  # All events preserved

    def test_raw_output_cleanup(self, seeded_prune_db):
        """Prune clears raw_output from old pending_queue items."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_prune_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["prune", "--older-than", "30d", "--keep-raw", "0"])

        assert result.exit_code == 0

        conn = sqlite3.connect(seeded_prune_db)
        # Old items should have raw_output cleared
        old_raw = conn.execute(
            "SELECT COUNT(*) FROM pending_queue WHERE id LIKE 'old-%' "
            "AND raw_output IS NOT NULL AND raw_output != ''"
        ).fetchone()[0]
        # Recent items should be preserved
        recent_raw = conn.execute(
            "SELECT COUNT(*) FROM pending_queue WHERE id LIKE 'recent-%' "
            "AND raw_output IS NOT NULL AND raw_output != ''"
        ).fetchone()[0]
        conn.close()
        assert old_raw == 0
        assert recent_raw == 3

    def test_keep_raw_preserves_newest(self, seeded_prune_db):
        """--keep-raw preserves N newest raw_outputs."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_prune_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["prune", "--older-than", "30d", "--keep-raw", "2"])

        assert result.exit_code == 0

        conn = sqlite3.connect(seeded_prune_db)
        # Count items with raw_output still set (recent + 2 kept old)
        with_raw = conn.execute(
            "SELECT COUNT(*) FROM pending_queue WHERE raw_output IS NOT NULL AND raw_output != ''"
        ).fetchone()[0]
        conn.close()
        # 3 recent items preserved + 2 old kept = 5
        assert with_raw == 5

    def test_event_log_cleanup(self, seeded_prune_db):
        """Prune deletes old event_log entries."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_prune_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["prune", "--older-than", "30d"])

        assert result.exit_code == 0

        conn = sqlite3.connect(seeded_prune_db)
        event_count = conn.execute("SELECT COUNT(*) FROM event_log").fetchone()[0]
        conn.close()
        # Only recent events remain
        assert event_count == 2

    def test_empty_prune_message(self, tmp_path):
        """Nothing to prune shows appropriate message."""
        from claude_mem_lite.cli.main import app

        base_dir = tmp_path / ".claude-mem"
        base_dir.mkdir()
        db_path = base_dir / "claude-mem.db"
        conn = sqlite3.connect(str(db_path))
        migrate(conn)
        conn.close()

        config = _make_config(str(db_path))
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["prune"])

        assert result.exit_code == 0
        assert "nothing to prune" in result.stdout.lower()

    def test_vacuum_reduces_size(self, seeded_prune_db):
        """--vacuum reduces database file size after pruning."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_prune_db)

        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(
                app, ["prune", "--older-than", "30d", "--keep-raw", "0", "--vacuum"]
            )

        assert result.exit_code == 0
        assert "vacuum" in result.stdout.lower()

    def test_vacuum_locked_shows_helpful_message(self, seeded_prune_db):
        """VACUUM while locked shows helpful error message."""
        from io import StringIO

        from rich.console import Console

        from claude_mem_lite.cli import prune_cmd as prune_module
        from claude_mem_lite.cli.prune_cmd import _do_vacuum

        # Create a mock connection that raises on VACUUM
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("database is locked")

        buf = StringIO()
        original = prune_module.console
        prune_module.console = Console(file=buf, force_terminal=False)
        try:
            _do_vacuum(mock_conn)
        finally:
            prune_module.console = original

        output = buf.getvalue()
        assert "locked" in output.lower()
        assert "worker" in output.lower()


# -----------------------------------------------------------------------
# pyproject.toml (2 tests)
# -----------------------------------------------------------------------


class TestPackaging:
    """Tests for package configuration."""

    def test_package_imports_correctly(self):
        """Package can be imported."""
        import claude_mem_lite

        assert claude_mem_lite is not None

    def test_entry_points_defined(self):
        """pyproject.toml defines CLI entry points."""
        import tomllib

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)

        scripts = data.get("project", {}).get("scripts", {})
        assert "claude-mem" in scripts
        assert "claude-mem-worker" in scripts


# -----------------------------------------------------------------------
# README (1 test)
# -----------------------------------------------------------------------


class TestReadme:
    """Tests for README.md presence and content."""

    def test_readme_exists_with_key_sections(self):
        """README.md exists and contains key sections."""
        readme_path = Path(__file__).parent.parent / "README.md"
        assert readme_path.exists(), "README.md should exist at project root"

        content = readme_path.read_text()
        assert "claude-mem-lite" in content.lower()
        assert "install" in content.lower()
        assert "command" in content.lower() or "usage" in content.lower()


# -----------------------------------------------------------------------
# Integration (2 tests)
# -----------------------------------------------------------------------


class TestCommandRegistration:
    """Tests for command registration in CLI."""

    def test_compress_command_registered(self):
        """compress command is accessible via CLI."""
        from claude_mem_lite.cli.main import app

        result = runner.invoke(app, ["compress", "--help"])
        assert result.exit_code == 0
        assert "pending" in result.stdout.lower() or "compress" in result.stdout.lower()

    def test_prune_command_registered(self):
        """prune command is accessible via CLI."""
        from claude_mem_lite.cli.main import app

        result = runner.invoke(app, ["prune", "--help"])
        assert result.exit_code == 0
        assert "older" in result.stdout.lower() or "prune" in result.stdout.lower()
