"""Tests for Phase 7: Eval Framework."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from claude_mem_lite.storage.migrations import migrate

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_observation(
    *,
    title: str = "Add JWT auth middleware",
    summary: str = "Added JWT authentication middleware for stateless auth.",
    detail: str | None = "Uses pyjwt library with RS256.",
    files_touched: str = '["src/auth/middleware.py"]',
    functions_changed: str = "[]",
    tokens_raw: int = 5000,
    tokens_compressed: int = 120,
):
    """Build a minimal Observation model for testing."""
    from claude_mem_lite.storage.models import Observation

    return Observation(
        id=str(uuid.uuid4()),
        session_id="test-session",
        tool_name="Write",
        title=title,
        summary=summary,
        detail=detail,
        files_touched=files_touched,
        functions_changed=functions_changed,
        tokens_raw=tokens_raw,
        tokens_compressed=tokens_compressed,
        created_at=datetime.now(UTC).isoformat(),
    )


def _seed_pending_queue(conn: sqlite3.Connection, count: int = 5) -> list[str]:
    """Seed pending_queue with done items that have raw_output."""
    ids = []
    for i in range(count):
        item_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO pending_queue "
            "(id, session_id, tool_name, raw_output, files_touched, status, attempts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                item_id,
                "test-session",
                f"tool_{i}",
                f"raw output content {i} " * 100,
                '["file.py"]',
                "done",
                1,
            ),
        )
        ids.append(item_id)
    conn.commit()
    return ids


def _seed_event_log(conn: sqlite3.Connection) -> None:
    """Seed event_log with entries for SQL query testing."""
    now = datetime.now(UTC).isoformat()

    events = [
        ("compress.done", '{"ratio": 42.0, "model": "claude-haiku-4-5"}', 87, 5000, 120),
        ("compress.done", '{"ratio": 35.0, "model": "claude-haiku-4-5"}', 92, 4800, 115),
        ("compress.error", '{"error_type": "rate_limit"}', None, None, None),
        (
            "search.hybrid",
            '{"query": "test", "result_count": 3, "top_score": 0.85}',
            12,
            None,
            None,
        ),
        ("search.hybrid", '{"query": "auth", "result_count": 0, "top_score": 0.0}', 8, None, None),
        (
            "hook.context_inject",
            '{"total_tokens": 1847, "budget": 2000, "layers_included": "session_index,function_map,learnings"}',
            42,
            None,
            None,
        ),
    ]

    for event_type, data, duration_ms, tokens_in, tokens_out in events:
        conn.execute(
            "INSERT INTO event_log "
            "(id, session_id, event_type, data, duration_ms, tokens_in, tokens_out, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                "test-session",
                event_type,
                data,
                duration_ms,
                tokens_in,
                tokens_out,
                now,
            ),
        )
    conn.commit()


def _mock_qag_response(
    *,
    questions: list[dict] | None = None,
    rationale_present: bool = True,
    rationale_note: str = "Summary explains JWT was chosen for stateless auth.",
) -> str:
    """Build a mock QAG JSON response."""
    if questions is None:
        questions = [
            {
                "question": "What architectural pattern was adopted?",
                "answerable": True,
                "evidence": "middleware chain",
            },
            {"question": "What new dependency was added?", "answerable": True, "evidence": "pyjwt"},
            {
                "question": "Why was JWT chosen over session tokens?",
                "answerable": True,
                "evidence": "stateless auth",
            },
            {
                "question": "What breaking API change was introduced?",
                "answerable": False,
                "evidence": None,
            },
            {"question": "What error handling was added?", "answerable": False, "evidence": None},
        ]
    return json.dumps(
        {
            "questions": questions,
            "decision_rationale_present": rationale_present,
            "rationale_note": rationale_note,
        }
    )


def _make_mock_client(response_text: str) -> AsyncMock:
    """Create a mock Anthropic client that returns the given text."""
    client = AsyncMock()
    text_block = MagicMock()
    text_block.text = response_text
    response = MagicMock()
    response.content = [text_block]
    response.usage = MagicMock(input_tokens=2000, output_tokens=300)
    client.messages.create = AsyncMock(return_value=response)
    return client


# -----------------------------------------------------------------------
# Deterministic scoring (5 tests)
# -----------------------------------------------------------------------


class TestDeterministicScoring:
    """Tests for deterministic observation scoring."""

    def test_valid_observation_structural_validity_one(self, tmp_config):
        """Valid observation with title and summary gets structural_validity=1.0."""
        from claude_mem_lite.eval.evaluator import score_deterministic

        obs = _make_observation()
        score = score_deterministic(obs, "raw output", 100, "claude-haiku-4-5-20251001", tmp_config)

        assert score.structural_validity == 1.0

    def test_empty_title_structural_validity_zero(self, tmp_config):
        """Observation with empty title gets structural_validity=0.0."""
        from claude_mem_lite.eval.evaluator import score_deterministic

        obs = _make_observation(title="")
        score = score_deterministic(obs, "raw output", 100, "claude-haiku-4-5-20251001", tmp_config)

        assert score.structural_validity == 0.0

    def test_good_title_quality_one(self, tmp_config):
        """Good imperative title gets title_quality=1.0."""
        from claude_mem_lite.eval.evaluator import score_deterministic

        obs = _make_observation(title="Add JWT auth middleware")
        score = score_deterministic(obs, "raw output", 100, "claude-haiku-4-5-20251001", tmp_config)

        assert score.title_quality == 1.0

    def test_bad_title_quality_below_threshold(self, tmp_config):
        """Title with weak start, short length, and trailing period scores below 0.6."""
        from claude_mem_lite.eval.evaluator import score_deterministic

        # "The thing." = 2 words (< 3 penalty -0.3) + period (-0.2) + weak start (-0.2) = 0.3
        obs = _make_observation(title="The thing.")
        score = score_deterministic(obs, "raw output", 100, "claude-haiku-4-5-20251001", tmp_config)

        assert score.title_quality < 0.6

    def test_zero_tokens_no_division_error(self, tmp_config):
        """Zero tokens_compressed produces ratio=0, no division error."""
        from claude_mem_lite.eval.evaluator import score_deterministic

        obs = _make_observation(tokens_compressed=0)
        score = score_deterministic(obs, "raw output", 100, "claude-haiku-4-5-20251001", tmp_config)

        assert score.compression_ratio == 0.0


# -----------------------------------------------------------------------
# QAG prompt (3 tests)
# -----------------------------------------------------------------------


class TestQAGPrompt:
    """Tests for QAG prompt parsing and validation."""

    async def test_valid_json_parsed_correctly(self, tmp_config):
        """Valid QAG JSON response is parsed into correct scores."""
        from claude_mem_lite.eval.evaluator import score_info_preservation

        obs = _make_observation()
        response_text = _mock_qag_response()
        client = _make_mock_client(response_text)

        info_score, rationale_score = await score_info_preservation(
            "raw output content", obs, client
        )

        assert info_score == pytest.approx(3 / 5)
        assert rationale_score == 1.0

    async def test_malformed_response_handled_gracefully(self, tmp_config):
        """Malformed JSON response returns (0.0, 0.0)."""
        from claude_mem_lite.eval.evaluator import score_info_preservation

        obs = _make_observation()
        client = _make_mock_client("this is not json at all {broken")

        info_score, rationale_score = await score_info_preservation(
            "raw output content", obs, client
        )

        assert info_score == 0.0
        assert rationale_score == 0.0

    async def test_question_count_validation(self, tmp_config):
        """<2 questions returns 0.0; >5 questions truncated to 5."""
        from claude_mem_lite.eval.evaluator import score_info_preservation

        obs = _make_observation()

        # Under-minimum: 1 question -> 0.0
        one_q = _mock_qag_response(
            questions=[{"question": "What?", "answerable": True, "evidence": "x"}],
        )
        client_under = _make_mock_client(one_q)
        info_under, _ = await score_info_preservation("raw", obs, client_under)
        assert info_under == 0.0

        # Over-maximum: 7 questions -> truncated to 5
        seven_qs = [
            {"question": f"Q{i}?", "answerable": True, "evidence": f"e{i}"} for i in range(7)
        ]
        over_q = _mock_qag_response(questions=seven_qs)
        client_over = _make_mock_client(over_q)
        info_over, _ = await score_info_preservation("raw", obs, client_over)
        # 5/5 after truncation (all answerable)
        assert info_over == pytest.approx(1.0)


# -----------------------------------------------------------------------
# QAG scoring (3 tests)
# -----------------------------------------------------------------------


class TestQAGScoring:
    """Tests for QAG information preservation scoring."""

    async def test_info_preservation_fraction_correct(self, tmp_config):
        """Info preservation is fraction of answerable questions."""
        from claude_mem_lite.eval.evaluator import score_info_preservation

        obs = _make_observation()
        # 3 out of 5 answerable
        response_text = _mock_qag_response()
        client = _make_mock_client(response_text)

        info_score, _ = await score_info_preservation("raw output", obs, client)
        assert info_score == pytest.approx(0.6)

    async def test_decision_rationale_binary(self, tmp_config):
        """Decision rationale is 1.0 if present, 0.0 if absent."""
        from claude_mem_lite.eval.evaluator import score_info_preservation

        obs = _make_observation()

        # Present
        present = _mock_qag_response(rationale_present=True)
        client_yes = _make_mock_client(present)
        _, rat_yes = await score_info_preservation("raw", obs, client_yes)
        assert rat_yes == 1.0

        # Absent
        absent = _mock_qag_response(rationale_present=False)
        client_no = _make_mock_client(absent)
        _, rat_no = await score_info_preservation("raw", obs, client_no)
        assert rat_no == 0.0

    async def test_variable_question_count_two_answerable(self, tmp_config):
        """2 questions, both answerable -> info_preservation=1.0."""
        from claude_mem_lite.eval.evaluator import score_info_preservation

        obs = _make_observation()
        two_qs = _mock_qag_response(
            questions=[
                {"question": "What bug was fixed?", "answerable": True, "evidence": "off-by-one"},
                {
                    "question": "What was the root cause?",
                    "answerable": True,
                    "evidence": "zero-indexed",
                },
            ],
        )
        client = _make_mock_client(two_qs)
        info_score, _ = await score_info_preservation("raw", obs, client)
        assert info_score == pytest.approx(1.0)


# -----------------------------------------------------------------------
# Benchmark runner (4 tests)
# -----------------------------------------------------------------------


class TestBenchmarkRunner:
    """Tests for the offline A/B benchmark runner."""

    async def test_samples_from_pending_queue(self, tmp_config):
        """Benchmark correctly samples from pending_queue."""
        from claude_mem_lite.eval.benchmark import BenchmarkRunner

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)
        item_ids = _seed_pending_queue(conn, count=5)
        conn.close()

        import aiosqlite

        db = await aiosqlite.connect(str(tmp_config.db_path))
        db.row_factory = aiosqlite.Row

        runner = BenchmarkRunner(db=db, client=AsyncMock(), config=tmp_config)
        samples = await runner._sample_raw_outputs(3)

        assert len(samples) == 3
        for s in samples:
            assert s.id in item_ids
            assert s.raw_output
        await db.close()

    async def test_paired_results_produced(self, tmp_config):
        """Benchmark produces PairedResult for each sample."""
        from claude_mem_lite.eval.benchmark import BenchmarkRunner

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)
        _seed_pending_queue(conn, count=2)
        conn.close()

        import aiosqlite

        db = await aiosqlite.connect(str(tmp_config.db_path))
        db.row_factory = aiosqlite.Row

        # Mock the compression and QAG to return valid results
        qag_response = _mock_qag_response()
        compress_response_text = json.dumps(
            {
                "title": "Test observation",
                "summary": "A test summary of changes.",
                "detail": None,
                "files_touched": ["file.py"],
                "functions_changed": [],
            }
        )
        client = _make_mock_client(qag_response)
        # Set up a second return for compression calls
        compress_block = MagicMock()
        compress_block.text = compress_response_text
        compress_response = MagicMock()
        compress_response.content = [compress_block]
        compress_response.usage = MagicMock(input_tokens=5000, output_tokens=120)

        qag_block = MagicMock()
        qag_block.text = qag_response
        qag_resp = MagicMock()
        qag_resp.content = [qag_block]
        qag_resp.usage = MagicMock(input_tokens=2000, output_tokens=300)

        # Alternate between compression and QAG responses
        client.messages.create = AsyncMock(
            side_effect=[
                compress_response,
                compress_response,  # model_a and model_b for sample 1
                qag_resp,
                qag_resp,  # QAG for sample 1
                compress_response,
                compress_response,  # model_a and model_b for sample 2
                qag_resp,
                qag_resp,  # QAG for sample 2
            ]
        )

        runner = BenchmarkRunner(db=db, client=client, config=tmp_config)
        report = await runner.run(sample_size=2)

        assert report.sample_size == 2
        assert len(report.pairs) == 2
        await db.close()

    async def test_empty_pending_queue_graceful(self, tmp_config):
        """Empty pending_queue produces BenchmarkReport with sample_size=0."""
        from claude_mem_lite.eval.benchmark import BenchmarkRunner

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)
        conn.close()

        import aiosqlite

        db = await aiosqlite.connect(str(tmp_config.db_path))
        db.row_factory = aiosqlite.Row

        runner = BenchmarkRunner(db=db, client=AsyncMock(), config=tmp_config)
        report = await runner.run(sample_size=10)

        assert report.sample_size == 0
        assert len(report.pairs) == 0
        assert report.avg_quality_a == 0.0
        assert report.avg_quality_b == 0.0
        await db.close()

    async def test_report_quality_per_dollar(self, tmp_config):
        """Report quality/dollar calculation is correct."""
        from claude_mem_lite.eval.benchmark import BenchmarkRunner

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)
        _seed_pending_queue(conn, count=1)
        conn.close()

        import aiosqlite

        db = await aiosqlite.connect(str(tmp_config.db_path))
        db.row_factory = aiosqlite.Row

        qag_response = _mock_qag_response()
        compress_text = json.dumps(
            {
                "title": "Test observation title here",
                "summary": "A test summary of changes made.",
                "detail": None,
                "files_touched": ["file.py"],
                "functions_changed": [],
            }
        )

        compress_block = MagicMock()
        compress_block.text = compress_text
        compress_resp = MagicMock()
        compress_resp.content = [compress_block]
        compress_resp.usage = MagicMock(input_tokens=5000, output_tokens=120)

        qag_block = MagicMock()
        qag_block.text = qag_response
        qag_resp = MagicMock()
        qag_resp.content = [qag_block]
        qag_resp.usage = MagicMock(input_tokens=2000, output_tokens=300)

        client = AsyncMock()
        client.messages.create = AsyncMock(
            side_effect=[compress_resp, compress_resp, qag_resp, qag_resp]
        )

        runner = BenchmarkRunner(db=db, client=client, config=tmp_config)
        report = await runner.run(
            model_a="claude-haiku-4-5-20251001",
            model_b="claude-sonnet-4-5-20250929",
            sample_size=1,
        )

        # Quality per dollar should be quality / cost
        if report.avg_cost_a > 0:
            expected_qpd_a = report.avg_quality_a / report.avg_cost_a
            assert report.quality_per_dollar_a == pytest.approx(expected_qpd_a)
        if report.avg_cost_b > 0:
            expected_qpd_b = report.avg_quality_b / report.avg_cost_b
            assert report.quality_per_dollar_b == pytest.approx(expected_qpd_b)
        await db.close()


# -----------------------------------------------------------------------
# CLI (4 tests)
# -----------------------------------------------------------------------


class TestCLI:
    """Tests for eval CLI commands."""

    def test_eval_compression_produces_output(self, tmp_config):
        """eval_compression with seeded data produces output dict."""
        from claude_mem_lite.cli.eval_cmd import eval_compression

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)

        # Seed an observation
        now = datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
            ("test-session", "/tmp/project", now, "active"),
        )
        conn.execute(
            "INSERT INTO observations "
            "(id, session_id, tool_name, title, summary, tokens_raw, tokens_compressed, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "obs-1",
                "test-session",
                "Write",
                "Add auth",
                "Added auth middleware.",
                5000,
                120,
                now,
            ),
        )
        # Seed a corresponding pending_queue item with raw_output
        conn.execute(
            "INSERT INTO pending_queue "
            "(id, session_id, tool_name, raw_output, status, attempts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("obs-1", "test-session", "Write", "raw output text here", "done", 1),
        )
        conn.commit()

        result = eval_compression(
            conn=conn,
            config=tmp_config,
            limit=5,
            _with_qag=False,
            as_json=False,
            since=None,
        )

        assert isinstance(result, list)
        assert len(result) >= 1
        assert "observation_id" in result[0]
        assert "deterministic" in result[0]
        conn.close()

    async def test_eval_benchmark_produces_report(self, tmp_config):
        """eval_benchmark with mocked compressor produces report."""
        from claude_mem_lite.cli.eval_cmd import eval_benchmark

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)
        _seed_pending_queue(conn, count=2)
        conn.close()

        import aiosqlite

        db = await aiosqlite.connect(str(tmp_config.db_path))
        db.row_factory = aiosqlite.Row

        compress_text = json.dumps(
            {
                "title": "Test observation title here",
                "summary": "A test summary.",
                "detail": None,
                "files_touched": ["file.py"],
                "functions_changed": [],
            }
        )
        qag_text = _mock_qag_response()

        compress_block = MagicMock()
        compress_block.text = compress_text
        compress_resp = MagicMock()
        compress_resp.content = [compress_block]
        compress_resp.usage = MagicMock(input_tokens=5000, output_tokens=120)

        qag_block = MagicMock()
        qag_block.text = qag_text
        qag_resp = MagicMock()
        qag_resp.content = [qag_block]
        qag_resp.usage = MagicMock(input_tokens=2000, output_tokens=300)

        client = AsyncMock()
        client.messages.create = AsyncMock(
            side_effect=[
                compress_resp,
                compress_resp,
                qag_resp,
                qag_resp,
                compress_resp,
                compress_resp,
                qag_resp,
                qag_resp,
            ]
        )

        report = await eval_benchmark(
            db=db,
            client=client,
            config=tmp_config,
            samples=2,
            as_json=False,
        )

        assert report.sample_size == 2
        await db.close()

    def test_eval_health_produces_dashboard(self, tmp_config):
        """eval_health with seeded event_log produces dashboard data."""
        from claude_mem_lite.cli.eval_cmd import eval_health

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)
        _seed_event_log(conn)

        result = eval_health(conn=conn, days=7, as_json=False)

        assert isinstance(result, dict)
        assert "compression" in result
        assert "health" in result
        conn.close()

    def test_json_output_is_valid(self, tmp_config):
        """JSON output from eval_compression is valid JSON."""
        from claude_mem_lite.cli.eval_cmd import eval_compression

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)

        now = datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
            ("test-session", "/tmp/project", now, "active"),
        )
        conn.execute(
            "INSERT INTO observations "
            "(id, session_id, tool_name, title, summary, tokens_raw, tokens_compressed, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("obs-1", "test-session", "Write", "Add auth", "Added auth.", 5000, 120, now),
        )
        conn.execute(
            "INSERT INTO pending_queue "
            "(id, session_id, tool_name, raw_output, status, attempts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("obs-1", "test-session", "Write", "raw output", "done", 1),
        )
        conn.commit()

        result = eval_compression(
            conn=conn,
            config=tmp_config,
            limit=5,
            _with_qag=False,
            as_json=True,
            since=None,
        )

        # as_json=True should return a JSON string
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        conn.close()


# -----------------------------------------------------------------------
# SQL queries (3 tests)
# -----------------------------------------------------------------------


class TestSQLQueries:
    """Tests for SQL query string constants."""

    def test_compression_query_runs(self, tmp_config):
        """Compression monitoring query runs without error against seeded DB."""
        from claude_mem_lite.eval.queries import COMPRESSION_MONITORING

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)
        _seed_event_log(conn)

        cursor = conn.execute(COMPRESSION_MONITORING)
        rows = cursor.fetchall()
        assert isinstance(rows, list)
        # We seeded 2 compress.done events
        assert len(rows) >= 1
        conn.close()

    def test_health_dashboard_query_runs(self, tmp_config):
        """System health daily query runs without error."""
        from claude_mem_lite.eval.queries import SYSTEM_HEALTH_DAILY

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)
        _seed_event_log(conn)

        cursor = conn.execute(SYSTEM_HEALTH_DAILY)
        rows = cursor.fetchall()
        assert isinstance(rows, list)
        conn.close()

    def test_search_query_runs(self, tmp_config):
        """Search quality query runs without error."""
        from claude_mem_lite.eval.queries import SEARCH_QUALITY

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)
        _seed_event_log(conn)

        cursor = conn.execute(SEARCH_QUALITY)
        rows = cursor.fetchall()
        assert isinstance(rows, list)
        conn.close()


# -----------------------------------------------------------------------
# Gap fix tests (5 tests)
# -----------------------------------------------------------------------


class TestGapFixes:
    """Tests for confirmed gap fixes."""

    def test_eval_compression_logs_event(self, tmp_config):
        """eval_compression logs an eval.compression event to event_log."""
        from claude_mem_lite.cli.eval_cmd import eval_compression

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)

        now = datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
            ("test-session", "/tmp/project", now, "active"),
        )
        conn.execute(
            "INSERT INTO observations "
            "(id, session_id, tool_name, title, summary, tokens_raw, tokens_compressed, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("obs-1", "test-session", "Write", "Add auth", "Added auth.", 5000, 120, now),
        )
        conn.execute(
            "INSERT INTO pending_queue "
            "(id, session_id, tool_name, raw_output, status, attempts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("obs-1", "test-session", "Write", "raw output", "done", 1),
        )
        conn.commit()

        eval_compression(conn=conn, config=tmp_config, limit=5, as_json=False, since=None)

        events = conn.execute(
            "SELECT * FROM event_log WHERE event_type = 'eval.compression'"
        ).fetchall()
        assert len(events) == 1
        data = json.loads(events[0][3])  # data column
        assert data["count"] == 1
        conn.close()

    async def test_benchmark_logs_start_and_done_events(self, tmp_config):
        """BenchmarkRunner logs eval.benchmark.start and eval.benchmark.done events."""
        from claude_mem_lite.eval.benchmark import BenchmarkRunner

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)
        conn.close()

        import aiosqlite

        db = await aiosqlite.connect(str(tmp_config.db_path))
        db.row_factory = aiosqlite.Row

        runner = BenchmarkRunner(db=db, client=AsyncMock(), config=tmp_config)
        await runner.run(sample_size=0)

        cursor = await db.execute("SELECT event_type FROM event_log ORDER BY created_at")
        rows = await cursor.fetchall()
        event_types = [row["event_type"] for row in rows]

        assert "eval.benchmark.start" in event_types
        assert "eval.benchmark.done" in event_types
        await db.close()

    def test_benchmark_typer_command_registered(self):
        """The 'benchmark' command is registered in eval_app."""
        from claude_mem_lite.cli.eval_cmd import eval_app

        command_names = [cmd.name for cmd in eval_app.registered_commands]
        assert "benchmark" in command_names
