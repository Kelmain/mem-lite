"""Tests for Phase 8: CLI Reports."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from claude_mem_lite.storage.migrations import migrate

runner = CliRunner()


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def empty_db(tmp_path):
    """Migrated database with no data, at Config-compatible path."""
    base_dir = tmp_path / ".claude-mem"
    base_dir.mkdir()
    db_path = base_dir / "claude-mem.db"
    conn = sqlite3.connect(str(db_path))
    migrate(conn)
    conn.close()
    return str(db_path)


@pytest.fixture
def seeded_db(tmp_path):
    """Database with realistic test data across all tables."""
    base_dir = tmp_path / ".claude-mem"
    base_dir.mkdir()
    db_path = base_dir / "claude-mem.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    migrate(conn)

    now = datetime.now(UTC).isoformat()

    # Sessions
    conn.execute(
        "INSERT INTO sessions (id, project_dir, started_at, status, observation_count) "
        "VALUES (?, ?, ?, ?, ?)",
        ("sess-1", "/tmp/project-a", now, "active", 3),
    )
    conn.execute(
        "INSERT INTO sessions (id, project_dir, started_at, status, observation_count) "
        "VALUES (?, ?, ?, ?, ?)",
        ("sess-2", "/tmp/project-b", now, "closed", 1),
    )

    # Observations (triggers will auto-populate FTS)
    for i in range(3):
        conn.execute(
            "INSERT INTO observations "
            "(id, session_id, tool_name, title, summary, detail, "
            "tokens_raw, tokens_compressed, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"obs-{i}",
                "sess-1",
                "Write",
                f"Add feature {i}",
                f"Added feature {i} with tests.",
                f"Detail for feature {i}",
                5000,
                120,
                now,
            ),
        )
    conn.execute(
        "INSERT INTO observations "
        "(id, session_id, tool_name, title, summary, tokens_raw, tokens_compressed, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("obs-3", "sess-2", "Read", "Review code", "Reviewed existing code.", 2000, 80, now),
    )

    # Function map
    conn.execute(
        "INSERT INTO function_map "
        "(id, session_id, file_path, qualified_name, kind, signature, "
        "body_hash, decorators, change_type, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            "sess-1",
            "src/auth.py",
            "authenticate",
            "function",
            "authenticate(email: str, password: str) -> Token",
            "abc123",
            "[]",
            "new",
            now,
        ),
    )
    conn.execute(
        "INSERT INTO function_map "
        "(id, session_id, file_path, qualified_name, kind, signature, "
        "body_hash, decorators, change_type, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            "sess-1",
            "src/auth.py",
            "validate_token",
            "function",
            "validate_token(token: str) -> bool",
            "def456",
            "[]",
            "modified",
            now,
        ),
    )
    conn.execute(
        "INSERT INTO function_map "
        "(id, session_id, file_path, qualified_name, kind, signature, "
        "body_hash, decorators, change_type, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            "sess-1",
            "src/legacy.py",
            "old_handler",
            "function",
            "old_handler() -> None",
            "ghi789",
            "[]",
            "deleted",
            now,
        ),
    )

    # Call graph
    conn.execute(
        "INSERT INTO call_graph "
        "(id, caller_file, caller_function, callee_file, callee_function, "
        "resolution, confidence, session_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            "src/auth.py",
            "authenticate",
            "src/auth.py",
            "validate_token",
            "direct",
            1.0,
            "sess-1",
            now,
        ),
    )

    # Learnings
    conn.execute(
        "INSERT INTO learnings "
        "(id, category, content, confidence, source_session_id, source_sessions, "
        "times_seen, is_manual, is_active, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            "convention",
            "Always use from __future__ import annotations.",
            0.95,
            "sess-1",
            json.dumps(["sess-1"]),
            3,
            0,
            1,
            now,
            now,
        ),
    )
    conn.execute(
        "INSERT INTO learnings "
        "(id, category, content, confidence, source_session_id, source_sessions, "
        "times_seen, is_manual, is_active, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            "gotcha",
            "LanceDB FTS requires string column, not list.",
            0.80,
            "sess-1",
            json.dumps(["sess-1"]),
            2,
            0,
            1,
            now,
            now,
        ),
    )

    # Event log
    conn.execute(
        "INSERT INTO event_log (id, session_id, event_type, data, duration_ms, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), "sess-1", "compress.done", '{"ratio": 42.0}', 87, now),
    )

    conn.commit()
    conn.close()
    return str(db_path)


def _make_config(db_path: str):
    """Create a Config with base_dir pointing to the DB parent directory.

    The Config.db_path property returns base_dir / 'claude-mem.db',
    so base_dir must be the directory containing the DB file.
    """
    from claude_mem_lite.config import Config

    base_dir = Path(db_path).parent
    config = Config(base_dir=base_dir)
    assert str(config.db_path) == db_path, f"Config mismatch: {config.db_path} != {db_path}"
    return config


# -----------------------------------------------------------------------
# ReportBuilder (5 tests)
# -----------------------------------------------------------------------


class TestReportBuilder:
    """Tests for ReportBuilder data queries."""

    def test_empty_db_returns_empty_report(self, empty_db):
        """Empty DB produces report with zero counts."""
        from claude_mem_lite.cli.report import ReportBuilder

        builder = ReportBuilder(empty_db)
        data = builder.build(days=1)

        assert data.observation_count == 0
        assert len(data.sessions) == 0
        assert data.function_changes.new == 0
        assert len(data.learnings) == 0

    def test_single_session_with_seeded_data(self, seeded_db):
        """Seeded DB produces correct counts."""
        from claude_mem_lite.cli.report import ReportBuilder

        builder = ReportBuilder(seeded_db)
        data = builder.build(days=1)

        assert data.observation_count == 4
        assert len(data.sessions) >= 1

    def test_multi_day_period_filters(self, seeded_db):
        """Multi-day period includes all recent data."""
        from claude_mem_lite.cli.report import ReportBuilder

        builder = ReportBuilder(seeded_db)
        data = builder.build(days=30)

        # All sessions within 30 days
        assert len(data.sessions) == 2

    def test_session_specific_report(self, seeded_db):
        """Session-specific report scopes to one session."""
        from claude_mem_lite.cli.report import ReportBuilder

        builder = ReportBuilder(seeded_db)
        data = builder.build(days=30, session_id="sess-1")

        assert len(data.sessions) == 1
        assert data.sessions[0].id == "sess-1"
        assert data.observation_count == 3
        assert data.session_detail is not None
        assert data.session_detail.session_id == "sess-1"

    def test_function_changes_aggregation(self, seeded_db):
        """Function change counts are aggregated correctly."""
        from claude_mem_lite.cli.report import ReportBuilder

        builder = ReportBuilder(seeded_db)
        data = builder.build(days=30)

        assert data.function_changes.new == 1
        assert data.function_changes.modified == 1
        assert data.function_changes.deleted == 1


# -----------------------------------------------------------------------
# Report rendering (3 tests)
# -----------------------------------------------------------------------


class TestReportRendering:
    """Tests for report rendering functions."""

    def test_render_report_empty_data_no_crash(self):
        """render_report with empty data does not crash."""
        from claude_mem_lite.cli.report import ReportData, render_report

        data = ReportData()
        # Should not raise
        render_report(data)

    def test_render_report_full_data_contains_expected(self, seeded_db):
        """render_report with full data produces output with expected strings."""
        from io import StringIO

        from rich.console import Console

        from claude_mem_lite.cli.report import ReportBuilder, render_report

        builder = ReportBuilder(seeded_db)
        data = builder.build(days=30)

        # Capture output
        buf = StringIO()
        from claude_mem_lite.cli import report as report_module

        original = report_module.console
        report_module.console = Console(file=buf, force_terminal=False)
        try:
            render_report(data)
        finally:
            report_module.console = original

        output = buf.getvalue()
        assert "Activity Report" in output
        assert "Sessions" in output

    def test_render_markdown_valid(self, seeded_db):
        """render_markdown returns valid markdown with expected headings."""
        from claude_mem_lite.cli.report import ReportBuilder, render_markdown

        builder = ReportBuilder(seeded_db)
        data = builder.build(days=30)

        md = render_markdown(data)
        assert "# Activity Report" in md
        assert "## Sessions" in md
        assert "## Function Changes" in md


# -----------------------------------------------------------------------
# Search (5 tests)
# -----------------------------------------------------------------------


class TestSearch:
    """Tests for search command functionality."""

    def test_fts_fallback_returns_results(self, seeded_db):
        """FTS fallback returns results from observations_fts."""
        from claude_mem_lite.cli.search_cmd import _search_fts

        results = _search_fts("feature", 5, seeded_db)
        assert len(results) > 0
        assert "title" in results[0]

    def test_fts_missing_table_returns_empty(self, tmp_path):
        """FTS with missing virtual table returns empty list, no crash."""
        # Create DB without migration v5 FTS
        db_path = tmp_path / "no_fts.db"
        conn = sqlite3.connect(str(db_path))
        # Only apply up to migration 4
        conn.execute("PRAGMA user_version = 4")
        conn.execute(
            "CREATE TABLE observations (id TEXT, title TEXT, summary TEXT, "
            "detail TEXT, created_at TEXT, tool_name TEXT, files_touched TEXT, "
            "rowid INTEGER PRIMARY KEY AUTOINCREMENT)"
        )
        conn.close()

        from claude_mem_lite.cli.search_cmd import _search_fts

        results = _search_fts("test", 5, str(db_path))
        assert results == []

    def test_search_json_output(self, seeded_db):
        """--json produces valid JSON output."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["search", "feature", "--json"])

        # Debug on failure
        if result.exit_code != 0:
            print("STDOUT:", result.stdout)
            if result.exception:
                import traceback

                traceback.print_exception(
                    type(result.exception), result.exception, result.exception.__traceback__
                )

        # Even if no worker, FTS should work
        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert "results" in parsed

    def test_worker_error_falls_back_to_fts(self, seeded_db):
        """Worker search error falls back to FTS (Amendment 5)."""
        from claude_mem_lite.cli.search_cmd import _search_fts, _search_worker

        # Simulate worker returning None (error)
        worker_result = _search_worker("test", 5, "/nonexistent/socket")
        assert worker_result is None

        # FTS fallback should work
        results = _search_fts("feature", 5, seeded_db)
        assert len(results) > 0

    def test_empty_query_result_message(self, seeded_db):
        """Empty query result shows 'No results' message."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["search", "zzzznonexistent_term_xyz"])

        assert result.exit_code == 0
        assert "No results" in result.stdout


# -----------------------------------------------------------------------
# Mermaid (4 tests)
# -----------------------------------------------------------------------


class TestMermaid:
    """Tests for mermaid command functionality."""

    def test_single_file_graph_contains_subgraph(self, seeded_db):
        """Single file graph output contains subgraph block."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["mermaid", "src/auth.py"])

        assert result.exit_code == 0
        assert "subgraph" in result.stdout

    def test_multi_file_multiple_subgraphs(self, seeded_db):
        """Multi-file graph contains multiple subgraph blocks."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["mermaid"])

        assert result.exit_code == 0
        # Should have subgraphs for auth.py and legacy.py
        assert result.stdout.count("subgraph") >= 2

    def test_empty_session_no_changes_message(self, empty_db):
        """Empty session produces 'No function changes' message."""
        from claude_mem_lite.cli.main import app

        config = _make_config(empty_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["mermaid"])

        assert result.exit_code == 0
        assert "No function changes" in result.stdout

    def test_all_flag_includes_unchanged(self, seeded_db):
        """--all flag includes unchanged functions."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["mermaid", "--all"])

        # With --all, should include everything
        assert result.exit_code == 0
        assert "graph TD" in result.stdout


# -----------------------------------------------------------------------
# Status (2 tests)
# -----------------------------------------------------------------------


class TestStatus:
    """Tests for status command."""

    def test_healthy_system_shows_green(self, seeded_db):
        """Healthy system shows green indicators."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Database: OK" in result.stdout
        assert "FTS index: OK" in result.stdout

    def test_missing_db_shows_error(self, tmp_path):
        """Missing DB shows error and exits with code 1."""
        from claude_mem_lite.cli.main import app
        from claude_mem_lite.config import Config

        config = Config(base_dir=tmp_path / "nonexistent")
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 1
        assert "NOT FOUND" in result.stdout


# -----------------------------------------------------------------------
# CLI integration (4 tests)
# -----------------------------------------------------------------------


class TestCLIIntegration:
    """Tests for CLI integration via typer.testing.CliRunner."""

    def test_report_runs_successfully(self, seeded_db):
        """report command runs successfully."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["report", "--days", "30"])

        assert result.exit_code == 0

    def test_search_runs_successfully(self, seeded_db):
        """search command runs successfully."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["search", "feature"])

        assert result.exit_code == 0

    def test_mermaid_runs_successfully(self, seeded_db):
        """mermaid command runs successfully."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["mermaid"])

        assert result.exit_code == 0

    def test_report_json_produces_valid_json(self, seeded_db):
        """report --json produces valid JSON."""
        from claude_mem_lite.cli.main import app

        config = _make_config(seeded_db)
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["report", "--days", "30", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert "sessions" in parsed


# -----------------------------------------------------------------------
# Edge cases (3 tests)
# -----------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_db_file_error_exit(self, tmp_path):
        """Missing DB file produces error message and exit 1."""
        from claude_mem_lite.cli.main import app
        from claude_mem_lite.config import Config

        config = Config(base_dir=tmp_path / "nonexistent")
        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["report"])

        assert result.exit_code == 1
        assert "Database not found" in result.stdout

    def test_fts5_migration_creates_table(self, empty_db):
        """FTS5 migration creates the observations_fts table."""
        conn = sqlite3.connect(empty_db)
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='observations_fts'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "observations_fts"

    def test_empty_database_all_commands_graceful(self, empty_db):
        """All commands handle empty database gracefully."""
        from claude_mem_lite.cli.main import app

        config = _make_config(empty_db)

        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["report", "--days", "30"])
        assert result.exit_code == 0

        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["search", "anything"])
        assert result.exit_code == 0

        with patch("claude_mem_lite.config.Config", return_value=config):
            result = runner.invoke(app, ["mermaid"])
        assert result.exit_code == 0


# -----------------------------------------------------------------------
# FTS5 migration (3 tests)
# -----------------------------------------------------------------------


class TestFTS5Migration:
    """Tests for FTS5 migration and trigger functionality."""

    def test_migration_creates_fts_table(self, tmp_path):
        """Migration v5 creates observations_fts virtual table."""
        db_path = tmp_path / "fts_test.db"
        conn = sqlite3.connect(str(db_path))
        migrate(conn)

        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='observations_fts'"
        ).fetchone()
        assert row is not None
        conn.close()

    def test_fts_triggers_sync_on_insert(self, tmp_path):
        """FTS triggers sync data when observations are inserted."""
        db_path = tmp_path / "fts_trigger.db"
        conn = sqlite3.connect(str(db_path))
        migrate(conn)

        conn.execute(
            "INSERT INTO sessions (id, project_dir, started_at) "
            "VALUES ('s1', '/tmp', datetime('now'))"
        )
        conn.execute(
            "INSERT INTO observations (id, session_id, tool_name, title, summary) "
            "VALUES ('o1', 's1', 'Write', 'Auth middleware', 'Added JWT auth')"
        )
        conn.commit()

        results = conn.execute(
            "SELECT * FROM observations_fts WHERE observations_fts MATCH ?",
            ("JWT",),
        ).fetchall()

        assert len(results) == 1
        conn.close()

    def test_existing_observations_backfilled(self, tmp_path):
        """Existing observations are backfilled into FTS on migration."""
        db_path = tmp_path / "fts_backfill.db"
        conn = sqlite3.connect(str(db_path))

        # Apply only migrations 1-4
        from claude_mem_lite.storage.migrations import MIGRATIONS, _split_sql

        conn.execute("BEGIN EXCLUSIVE")
        for version, sql in MIGRATIONS:
            if version <= 4:
                for stmt in _split_sql(sql):
                    conn.execute(stmt)
                conn.execute(f"PRAGMA user_version = {version}")
        conn.execute("COMMIT")

        # Insert observation before FTS migration
        conn.execute(
            "INSERT INTO sessions (id, project_dir, started_at) "
            "VALUES ('s1', '/tmp', datetime('now'))"
        )
        conn.execute(
            "INSERT INTO observations (id, session_id, tool_name, title, summary) "
            "VALUES ('pre-fts-obs', 's1', 'Write', 'Pre-migration title', 'Pre-migration summary')"
        )
        conn.commit()

        # Now apply migration 5
        migrate(conn)

        # The pre-existing observation should be in FTS
        # Use a simple word (FTS5 interprets hyphens as operators)
        results = conn.execute(
            "SELECT * FROM observations_fts WHERE observations_fts MATCH ?",
            ('"Pre-migration"',),
        ).fetchall()
        assert len(results) == 1
        conn.close()
