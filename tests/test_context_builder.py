"""Tests for Phase 5: Context Injection -- 27 tests."""

from __future__ import annotations

import asyncio
import socket
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import httpx
import pytest

from claude_mem_lite.worker.processor import IdleTracker
from claude_mem_lite.worker.server import app

if TYPE_CHECKING:
    import aiosqlite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_test_data(db: aiosqlite.Connection, project_dir: str = "/test/project"):
    """Seed typical test data: sessions, observations, function_map, learnings."""
    now = datetime.now(UTC)
    for i, delta in enumerate([timedelta(hours=2), timedelta(days=1), timedelta(days=2)]):
        ts = (now - delta).isoformat()
        await db.execute(
            "INSERT INTO sessions (id, project_dir, started_at, summary, status) "
            "VALUES (?, ?, ?, ?, 'closed')",
            (f"s{i + 1}", project_dir, ts, f"Session {i + 1} summary text here"),
        )
    for i in range(3):
        await db.execute(
            "INSERT INTO observations "
            "(id, session_id, tool_name, title, summary, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"obs{i + 1}",
                f"s{i + 1}",
                "Write",
                f"Observation {i + 1}",
                f"Summary of observation {i + 1}",
                (now - timedelta(hours=i + 1)).isoformat(),
            ),
        )
    await db.execute(
        "INSERT INTO function_map "
        "(id, session_id, file_path, qualified_name, kind, signature, body_hash, change_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "fm1",
            "s1",
            "/test/project/src/auth.py",
            "authenticate",
            "function",
            "authenticate(email, password) -> Token",
            "abc123",
            "modified",
        ),
    )
    await db.execute(
        "INSERT INTO learnings "
        "(id, category, content, confidence, source_session_id, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("l1", "architecture", "FastAPI + SQLAlchemy ORM", 0.85, "s1", 1),
    )
    await db.commit()


# ---------------------------------------------------------------------------
# 1. TestTokenEstimator -- 3 tests
# ---------------------------------------------------------------------------


class TestTokenEstimator:
    """Tests for the token estimator utility."""

    def test_empty_string(self):
        """estimate_tokens('') returns minimum of 1."""
        from claude_mem_lite.context.estimator import estimate_tokens

        assert estimate_tokens("") == 1

    def test_english_prose(self):
        """English prose produces a reasonable token estimate."""
        from claude_mem_lite.context.estimator import estimate_tokens

        text = "The quick brown fox jumps over the lazy dog near the river bank"
        tokens = estimate_tokens(text)
        # ~13 words => roughly 10-20 tokens depending on estimator
        assert 5 <= tokens <= 25

    def test_code_snippet(self):
        """Python function signature produces a reasonable token count."""
        from claude_mem_lite.context.estimator import estimate_tokens

        code = "def authenticate(email: str, password: str) -> Token:\n    pass"
        tokens = estimate_tokens(code)
        assert tokens >= 5
        assert tokens <= 50


# ---------------------------------------------------------------------------
# 2. TestContextBuilderLayers -- 5 tests (async)
# ---------------------------------------------------------------------------


class TestContextBuilderLayers:
    """Each layer builds correctly from fixture data."""

    async def test_session_index_layer(self, async_db):
        """Session index layer includes '## Recent Sessions' header and session entries."""
        from claude_mem_lite.context.builder import ContextBuilder

        await _seed_test_data(async_db)
        builder = ContextBuilder(async_db, lance_store=None, budget=2000)
        layer = await builder._build_session_index()

        assert layer is not None
        assert "## Recent Sessions" in layer.text
        assert "Session 1 summary" in layer.text
        assert "Session 2 summary" in layer.text

    async def test_function_map_layer(self, async_db):
        """Function map layer includes '## Recently Changed Code' header."""
        from claude_mem_lite.context.builder import ContextBuilder

        await _seed_test_data(async_db)
        builder = ContextBuilder(async_db, lance_store=None, budget=2000)
        layer = await builder._build_function_map()

        assert layer is not None
        assert "## Recently Changed Code" in layer.text
        assert "authenticate" in layer.text

    async def test_learnings_layer(self, async_db):
        """Learnings layer includes '## Project Knowledge' header."""
        from claude_mem_lite.context.builder import ContextBuilder

        await _seed_test_data(async_db)
        builder = ContextBuilder(async_db, lance_store=None, budget=2000)
        layer = await builder._build_learnings()

        assert layer is not None
        assert "## Project Knowledge" in layer.text
        assert "FastAPI + SQLAlchemy ORM" in layer.text

    async def test_observations_layer_recency(self, async_db):
        """Observations layer uses recency (NOT semantic search) per Amendment 1."""
        from claude_mem_lite.context.builder import ContextBuilder

        await _seed_test_data(async_db)
        builder = ContextBuilder(async_db, lance_store=None, budget=2000)
        layer = await builder._build_observations()

        assert layer is not None
        assert "## Recent Observations" in layer.text
        assert "Observation 1" in layer.text

    async def test_empty_table_returns_none(self, async_db):
        """Each layer returns None when its table is empty."""
        from claude_mem_lite.context.builder import ContextBuilder

        builder = ContextBuilder(async_db, lance_store=None, budget=2000)

        assert await builder._build_session_index() is None
        assert await builder._build_function_map() is None
        assert await builder._build_learnings() is None
        assert await builder._build_observations() is None


# ---------------------------------------------------------------------------
# 3. TestContextBuilderAssembly -- 5 tests (async)
# ---------------------------------------------------------------------------


class TestContextBuilderAssembly:
    """Full build assembly with budget enforcement."""

    async def test_full_build_within_budget(self, async_db):
        """Build with populated DB stays within budget, includes session_index."""
        from claude_mem_lite.context.builder import ContextBuilder

        await _seed_test_data(async_db)
        builder = ContextBuilder(async_db, lance_store=None, budget=2000)
        result = await builder.build()

        assert result.total_tokens <= 2000
        assert "session_index" in result.layers_included

    async def test_budget_enforcement(self, async_db):
        """Build with tiny budget (100 tokens) skips some layers."""
        from claude_mem_lite.context.builder import ContextBuilder

        await _seed_test_data(async_db)
        builder = ContextBuilder(async_db, lance_store=None, budget=100)
        result = await builder.build()

        assert result.total_tokens <= 100
        assert len(result.layers_skipped) > 0

    async def test_empty_db_returns_empty(self, async_db):
        """Build with empty DB returns empty text and zero tokens."""
        from claude_mem_lite.context.builder import ContextBuilder

        builder = ContextBuilder(async_db, lance_store=None, budget=2000)
        result = await builder.build()

        assert result.text == ""
        assert result.total_tokens == 0

    async def test_layer_exception_isolated(self, async_db):
        """If one layer fails (table dropped), other layers still build."""
        from claude_mem_lite.context.builder import ContextBuilder

        await _seed_test_data(async_db)

        # Drop function_map table to force an error in that layer
        await async_db.execute("DROP TABLE function_map")
        await async_db.commit()

        builder = ContextBuilder(async_db, lance_store=None, budget=2000)
        result = await builder.build()

        # Other layers should still be present
        assert "session_index" in result.layers_included or result.total_tokens > 0

    async def test_observations_no_semantic_search(self, async_db):
        """Observations layer does NOT call lance_store.search_observations (Amendment 1)."""
        from claude_mem_lite.context.builder import ContextBuilder

        await _seed_test_data(async_db)

        mock_lance = MagicMock()
        mock_lance.search_observations = MagicMock(return_value=[])

        builder = ContextBuilder(async_db, lance_store=mock_lance, budget=2000)
        await builder.build()

        mock_lance.search_observations.assert_not_called()


# ---------------------------------------------------------------------------
# 4. TestContextBuilderFormat -- 3 tests
# ---------------------------------------------------------------------------


class TestContextBuilderFormat:
    """Tests for formatting helpers."""

    def test_relative_time_recent(self):
        """2 hours ago renders as '2h' or similar."""
        from claude_mem_lite.context.builder import _relative_time

        ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        result = _relative_time(ts)
        assert "2h" in result or "2 hour" in result

    def test_relative_time_yesterday(self):
        """25 hours ago renders as 'yesterday' or '1d'."""
        from claude_mem_lite.context.builder import _relative_time

        ts = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
        result = _relative_time(ts)
        assert "yesterday" in result.lower() or "1d" in result

    def test_relative_time_utc_naive(self):
        """Naive timestamp (no tz suffix, as SQLite stores) shows 'ago', not future."""
        from claude_mem_lite.context.builder import _relative_time

        # SQLite stores naive UTC: "2026-02-15 10:00:00"
        naive_ts = (datetime.now(UTC) - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
        result = _relative_time(naive_ts)
        assert "ago" in result or "h" in result
        assert "in the future" not in result.lower()


# ---------------------------------------------------------------------------
# 5. TestContextEndpoint -- 4 tests (async, httpx)
# ---------------------------------------------------------------------------


@pytest.fixture
async def context_app(tmp_config, store, async_db):
    """Set up app.state for context endpoint testing."""
    mock_lance = MagicMock()
    mock_lance.search_observations = MagicMock(return_value=[])

    idle_tracker = IdleTracker(timeout_minutes=999)

    app.state.config = tmp_config
    app.state.db = async_db
    app.state.lance_store = mock_lance
    app.state.idle_tracker = idle_tracker

    return app


@pytest.fixture
async def context_client(context_app):
    """Async HTTP client for context endpoint tests."""
    transport = httpx.ASGITransport(app=context_app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestContextEndpoint:
    """Tests for the GET /api/context endpoint."""

    async def test_get_context_returns_json(self, context_client):
        """GET /api/context?project_path=/proj returns 200 with expected keys."""
        response = await context_client.get("/api/context", params={"project_path": "/proj"})

        assert response.status_code == 200
        data = response.json()
        assert "context" in data
        assert "tokens" in data
        assert "layers" in data
        assert "build_ms" in data

    async def test_get_context_empty_db(self, context_client):
        """Empty DB returns context='' with zero tokens."""
        response = await context_client.get("/api/context", params={"project_path": "/proj"})

        assert response.status_code == 200
        data = response.json()
        assert data["context"] == ""
        assert data["tokens"] == 0

    async def test_get_context_with_sessions(self, async_db, context_client):
        """Seeded sessions appear in the context response."""
        await _seed_test_data(async_db)

        response = await context_client.get(
            "/api/context", params={"project_path": "/test/project"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "Session 1 summary" in data["context"] or data["tokens"] > 0

    async def test_get_context_touches_idle_tracker(self, context_app, context_client):
        """GET /api/context calls idle_tracker.touch()."""
        mock_tracker = MagicMock(spec=IdleTracker)
        context_app.state.idle_tracker = mock_tracker

        await context_client.get("/api/context", params={"project_path": "/proj"})

        mock_tracker.touch.assert_called()


# ---------------------------------------------------------------------------
# 6. TestContextHook -- 4 tests (sync)
# ---------------------------------------------------------------------------


class TestContextHook:
    """Tests for updated hooks/context.py functions."""

    def test_process_event_creates_session(self, store):
        """_process_event creates a session in the DB."""
        from claude_mem_lite.hooks.context import _process_event

        event = {
            "session_id": "hook-sess-001",
            "hook_event_name": "SessionStart",
            "cwd": "/test/project",
            "source": "startup",
            "transcript_path": "/tmp/transcript.jsonl",
            "permission_mode": "default",
        }
        _process_event(event, store)

        session = store.get_session("hook-sess-001")
        assert session is not None
        assert session.project_dir != ""

    def test_basic_context_fallback(self, store):
        """_get_basic_context returns session list when worker unavailable."""
        from claude_mem_lite.hooks.context import _get_basic_context

        store.create_session(project_dir="/proj/a")
        store.create_session(project_dir="/proj/b")

        result = _get_basic_context(store)
        assert "Recent Sessions" in result

    def test_basic_context_empty_db(self, store):
        """_get_basic_context returns '' when no sessions exist."""
        from claude_mem_lite.hooks.context import _get_basic_context

        result = _get_basic_context(store)
        assert result == ""

    def test_af_unix_guard(self):
        """When socket.AF_UNIX is not available, _get_worker_context returns ''."""
        from claude_mem_lite.hooks.context import _get_worker_context

        with (
            patch.object(socket, "AF_UNIX", create=False, new=None),
            patch("claude_mem_lite.hooks.context.hasattr", return_value=False),
        ):
            result = _get_worker_context("/fake/socket.sock", "/test/project")
        assert result == ""


# ---------------------------------------------------------------------------
# 7. TestIntegration -- 3 tests (async)
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests for context building."""

    @pytest.mark.integration
    async def test_full_pipeline(self, async_db):
        """Insert all data types, build context, verify all layers present and within budget."""
        from claude_mem_lite.context.builder import ContextBuilder

        await _seed_test_data(async_db)

        # Add extra function_map entries to ensure that layer has content
        await async_db.execute(
            "INSERT INTO function_map "
            "(id, session_id, file_path, qualified_name, kind, signature, body_hash, change_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "fm2",
                "s2",
                "/test/project/src/models.py",
                "UserModel",
                "class",
                "class UserModel(BaseModel)",
                "def456",
                "new",
            ),
        )
        await async_db.commit()

        builder = ContextBuilder(async_db, lance_store=None, budget=2000)
        result = await builder.build()

        assert result.total_tokens <= 2000
        assert result.total_tokens > 0
        assert len(result.layers_included) >= 3  # at least sessions, observations, learnings

        # Verify content from each data type appears
        assert "Session" in result.text
        assert "authenticate" in result.text or "UserModel" in result.text

    @pytest.mark.integration
    async def test_token_budget_respected_e2e(self, async_db):
        """Build with budget=500, verify total_tokens <= 500."""
        from claude_mem_lite.context.builder import ContextBuilder

        await _seed_test_data(async_db)
        builder = ContextBuilder(async_db, lance_store=None, budget=500)
        result = await builder.build()

        assert result.total_tokens <= 500

    @pytest.mark.integration
    async def test_concurrent_context_builds(self, async_db):
        """3 concurrent builds all succeed without errors."""
        from claude_mem_lite.context.builder import ContextBuilder

        await _seed_test_data(async_db)

        async def do_build():
            builder = ContextBuilder(async_db, lance_store=None, budget=2000)
            return await builder.build()

        results = await asyncio.gather(do_build(), do_build(), do_build())

        assert len(results) == 3
        for result in results:
            assert result.total_tokens >= 0
            assert isinstance(result.text, str)
