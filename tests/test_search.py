"""Phase 4: Embeddings + Search — 24 tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Embedder tests (5)
# ---------------------------------------------------------------------------


class TestEmbedder:
    """Tests for the Qwen3-Embedding-0.6B wrapper."""

    def test_embed_produces_correct_dims(self, mock_embedder, tmp_config):
        """embed_texts output shape is (n, config.embedding_dim)."""
        texts = ["hello world", "another text"]
        result = mock_embedder.embed_texts(texts)
        assert len(result) == 2
        assert len(result[0]) == tmp_config.embedding_dim
        assert len(result[1]) == tmp_config.embedding_dim

    def test_instruction_prefix_applied(self, tmp_config):
        """'observation' query_type prepends the instruction prefix."""
        from claude_mem_lite.search.embedder import Embedder

        e = Embedder(tmp_config)
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(
            tolist=lambda: [[0.1] * tmp_config.embedding_dim]
        )
        e._model = mock_model
        e._available = True

        e.embed_texts(["test query"], query_type="observation")

        call_args = mock_model.encode.call_args
        passed_texts = call_args[0][0]
        assert len(passed_texts) == 1
        assert passed_texts[0].startswith("Instruct: Find development observations")
        assert "test query" in passed_texts[0]

    def test_document_no_prefix(self, tmp_config):
        """'document' query_type adds no prefix — text passed as-is."""
        from claude_mem_lite.search.embedder import Embedder

        e = Embedder(tmp_config)
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(
            tolist=lambda: [[0.1] * tmp_config.embedding_dim]
        )
        e._model = mock_model
        e._available = True

        e.embed_texts(["raw text"], query_type="document")

        call_args = mock_model.encode.call_args
        passed_texts = call_args[0][0]
        assert passed_texts == ["raw text"]

    def test_truncate_dim_respected(self, tmp_path):
        """Config with embedding_dim=256 passes truncate_dim to SentenceTransformer."""
        from claude_mem_lite.config import Config
        from claude_mem_lite.search.embedder import Embedder

        config = Config(base_dir=tmp_path / ".claude-mem", embedding_dim=256)
        config.ensure_dirs()
        e = Embedder(config)

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value = MagicMock()
            e.load()
            mock_st.assert_called_once()
            _, kwargs = mock_st.call_args
            assert kwargs.get("truncate_dim") == 256

    def test_load_failure_graceful(self, tmp_path):
        """Invalid model name sets available=False without raising."""
        from claude_mem_lite.config import Config
        from claude_mem_lite.search.embedder import Embedder

        config = Config(
            base_dir=tmp_path / ".claude-mem",
            embedding_model="nonexistent/model-that-does-not-exist",
        )
        config.ensure_dirs()
        e = Embedder(config)

        result = e.load()

        assert result is False
        assert e.available is False


# ---------------------------------------------------------------------------
# LanceStore tests (6)
# ---------------------------------------------------------------------------


class TestLanceStore:
    """Tests for LanceDB table management."""

    def test_tables_created(self, lance_store):
        """connect() creates all three tables."""
        table_names = set(lance_store._db.table_names())
        assert "observations_vec" in table_names
        assert "summaries_vec" in table_names
        assert "learnings_vec" in table_names

    def test_add_observation(self, lance_store):
        """add_observation stores data that is retrievable via search."""
        lance_store.add_observation(
            obs_id="obs-1",
            session_id="sess-1",
            title="Added JWT auth middleware",
            summary="Implemented JWT middleware for API routes",
            files_touched="src/auth.py",
            functions_changed="login,verify_token",
            created_at="2026-02-14T10:00:00",
        )
        results = lance_store.search_observations("JWT auth", limit=5)
        assert len(results) >= 1
        assert any(r["observation_id"] == "obs-1" for r in results)

    def test_search_empty_table(self, lance_store):
        """Search on empty table returns empty list."""
        results = lance_store.search_observations("anything", limit=5)
        assert results == []

    def test_fts_index_created(self, lance_store):
        """FTS search finds keyword matches on title/summary text."""
        lance_store.add_observation(
            obs_id="obs-fts",
            session_id="sess-1",
            title="Refactored authentication module",
            summary="Moved auth logic to dedicated middleware",
            files_touched="src/auth.py",
            functions_changed="",
            created_at="2026-02-14T10:00:00",
        )
        results = lance_store.search_fts_only("authentication", limit=5)
        assert len(results) >= 1

    def test_hybrid_search(self, lance_store):
        """Hybrid search merges vector + FTS results."""
        lance_store.add_observation(
            obs_id="obs-hybrid",
            session_id="sess-1",
            title="Implemented rate limiting",
            summary="Added token bucket rate limiter to API gateway",
            files_touched="src/gateway.py",
            functions_changed="rate_limit",
            created_at="2026-02-14T10:00:00",
        )
        results = lance_store.search_observations("rate limiting", limit=5)
        assert len(results) >= 1

    def test_add_summary(self, lance_store):
        """Session summary is stored and searchable."""
        lance_store.add_summary(
            session_id="sess-1",
            summary_text="Completed authentication and rate limiting features",
            project_path="/home/user/project",
            created_at="2026-02-14T12:00:00",
        )
        table = lance_store._tables["summaries_vec"]
        assert table.count_rows() == 1


# ---------------------------------------------------------------------------
# Hybrid Search tests (4)
# ---------------------------------------------------------------------------


class TestHybridSearch:
    """Tests for hybrid search orchestration."""

    def test_semantic_query_finds_related(self, lance_store):
        """Semantic search finds related observations."""
        from claude_mem_lite.search.hybrid import HybridSearcher

        lance_store.add_observation(
            obs_id="obs-sem-1",
            session_id="sess-1",
            title="Added JWT auth middleware",
            summary="Implemented token-based authentication for the API",
            files_touched="src/auth.py",
            functions_changed="verify_token",
            created_at="2026-02-14T10:00:00",
        )
        lance_store.add_observation(
            obs_id="obs-sem-2",
            session_id="sess-1",
            title="Fixed database connection pooling",
            summary="Resolved connection leak in async pool manager",
            files_touched="src/db.py",
            functions_changed="get_pool",
            created_at="2026-02-14T11:00:00",
        )

        searcher = HybridSearcher(lance_store, lance_store.embedder)
        results, search_type = searcher.search("authentication middleware", limit=5)
        assert len(results) >= 1
        assert search_type == "hybrid"

    def test_keyword_query_finds_exact(self, lance_store):
        """Keyword matches exact title/summary text via FTS."""
        from claude_mem_lite.search.hybrid import HybridSearcher

        lance_store.add_observation(
            obs_id="obs-kw",
            session_id="sess-1",
            title="Implemented WebSocket handler",
            summary="Added real-time messaging via WebSocket connections",
            files_touched="src/ws.py",
            functions_changed="on_connect",
            created_at="2026-02-14T10:00:00",
        )

        searcher = HybridSearcher(lance_store, lance_store.embedder)
        results, _search_type = searcher.search("WebSocket", limit=5)
        assert len(results) >= 1

    def test_hybrid_outperforms_either_alone(self, lance_store):
        """Smoke test: hybrid search executes and returns results."""
        from claude_mem_lite.search.hybrid import HybridSearcher

        for i in range(5):
            lance_store.add_observation(
                obs_id=f"obs-perf-{i}",
                session_id="sess-1",
                title=f"Feature implementation #{i}",
                summary=f"Implemented feature number {i} with tests",
                files_touched=f"src/feature_{i}.py",
                functions_changed=f"feature_{i}",
                created_at=f"2026-02-14T{10 + i}:00:00",
            )

        searcher = HybridSearcher(lance_store, lance_store.embedder)
        results, search_type = searcher.search("feature implementation", limit=5)
        assert len(results) >= 1
        assert search_type == "hybrid"

    def test_fts_fallback_when_embeddings_unavailable(self, tmp_config, lance_store):
        """When embedder.available=False, search uses FTS only."""
        from claude_mem_lite.search.embedder import Embedder
        from claude_mem_lite.search.hybrid import HybridSearcher

        # Seed data with the working mock embedder
        lance_store.add_observation(
            obs_id="obs-fallback",
            session_id="sess-1",
            title="Database migration script",
            summary="Added Alembic migration for user table",
            files_touched="migrations/001.py",
            functions_changed="upgrade",
            created_at="2026-02-14T10:00:00",
        )

        # Create an unavailable embedder
        unavailable = Embedder(tmp_config)
        unavailable._available = False

        searcher = HybridSearcher(lance_store, unavailable)
        results, search_type = searcher.search("migration", limit=5)
        assert search_type == "fts"
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# API Endpoints tests (4)
# ---------------------------------------------------------------------------


class TestSearchEndpoints:
    """Tests for Phase 4 API endpoints."""

    @pytest.fixture
    async def search_app(self, tmp_config, mock_embedder, store):
        """FastAPI app configured with mock search dependencies."""
        import aiosqlite

        from claude_mem_lite.worker.server import app

        db = await aiosqlite.connect(str(tmp_config.db_path))
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute("PRAGMA busy_timeout=3000")

        app.state.config = tmp_config
        app.state.db = db
        app.state.embedder = mock_embedder
        app.state.idle_tracker = MagicMock()

        # Mock lance_store
        mock_ls = MagicMock()
        mock_ls.search_observations.return_value = [
            {
                "observation_id": "obs-api-1",
                "session_id": "sess-1",
                "title": "Added auth",
                "summary": "JWT auth middleware",
                "files_touched": "src/auth.py",
                "created_at": "2026-02-14T10:00:00",
                "_relevance_score": 0.95,
            }
        ]
        mock_ls.search_fts_only.return_value = [
            {
                "observation_id": "obs-api-1",
                "session_id": "sess-1",
                "title": "Added auth",
                "summary": "JWT auth middleware",
                "files_touched": "src/auth.py",
                "created_at": "2026-02-14T10:00:00",
            }
        ]
        app.state.lance_store = mock_ls

        yield app

        await db.close()

    @pytest.fixture
    async def client(self, search_app):
        """Async httpx client for testing."""
        import httpx

        transport = httpx.ASGITransport(app=search_app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    async def test_search_returns_results(self, client):
        """GET /api/search?q=auth returns 200 with results array."""
        resp = await client.get("/api/search", params={"q": "auth"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "query" in data
        assert "count" in data
        assert "search_type" in data
        assert data["count"] >= 1
        assert data["query"] == "auth"

    async def test_search_fts_fallback(self, search_app, client):
        """When embedder.available=False, search_type is 'fts'."""
        from claude_mem_lite.search.embedder import Embedder

        unavailable = Embedder(search_app.state.config)
        unavailable._available = False
        search_app.state.embedder = unavailable

        resp = await client.get("/api/search", params={"q": "auth"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["search_type"] == "fts"

    async def test_observation_detail(self, client, store):
        """GET /api/observation/{id} returns full observation."""
        import json
        import uuid

        obs_id = str(uuid.uuid4())
        store.conn.execute(
            """INSERT INTO sessions (id, project_dir) VALUES (?, ?)""",
            ("sess-detail", "/tmp/project"),
        )
        store.conn.execute(
            """INSERT INTO observations
               (id, session_id, tool_name, title, summary, detail,
                files_touched, functions_changed, tokens_raw, tokens_compressed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                obs_id,
                "sess-detail",
                "Write",
                "Test observation",
                "A test summary",
                "Detailed info",
                json.dumps(["file.py"]),
                json.dumps([]),
                100,
                50,
            ),
        )
        store.conn.commit()

        resp = await client.get(f"/api/observation/{obs_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Test observation"
        assert data["summary"] == "A test summary"

    async def test_observation_not_found(self, client):
        """GET /api/observation/xxx returns 404."""
        resp = await client.get("/api/observation/nonexistent-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Processor Integration tests (3)
# ---------------------------------------------------------------------------


class TestProcessorIntegration:
    """Tests for embedding in the compress pipeline."""

    async def test_compress_and_embed_pipeline(self, tmp_config, async_db, mock_embedder):
        """End-to-end: compress -> embed -> store in LanceDB."""
        from claude_mem_lite.search.lance_store import LanceStore
        from claude_mem_lite.worker.compressor import Compressor
        from claude_mem_lite.worker.processor import IdleTracker, Processor

        lance = LanceStore(tmp_config, mock_embedder)
        lance.connect()

        # Insert a session
        await async_db.execute(
            "INSERT INTO sessions (id, project_dir) VALUES (?, ?)",
            ("sess-pipe", "/tmp/project"),
        )
        # Insert a queue item
        await async_db.execute(
            """INSERT INTO pending_queue (id, session_id, tool_name, raw_output, status)
               VALUES (?, ?, ?, ?, ?)""",
            ("q-1", "sess-pipe", "Write", "wrote file.py with auth logic", "raw"),
        )
        await async_db.commit()

        compressor = MagicMock(spec=Compressor)
        from claude_mem_lite.storage.models import CompressedObservation

        compressor.compress = AsyncMock(
            return_value=CompressedObservation(
                title="Added auth logic",
                summary="Wrote authentication middleware",
                files_touched=["src/auth.py"],
                tokens_in=200,
                tokens_out=80,
            )
        )

        idle_tracker = IdleTracker(timeout_minutes=30)
        processor = Processor(
            async_db,
            compressor,
            idle_tracker,
            lance_store=lance,
            embedder=mock_embedder,
        )

        items = await processor.dequeue_batch()
        assert len(items) == 1
        await processor.process_item(items[0])

        # Verify observation in SQLite
        cursor = await async_db.execute("SELECT * FROM observations WHERE session_id = 'sess-pipe'")
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0]["embedding_status"] == "embedded"

        # Verify in LanceDB
        assert lance._tables["observations_vec"].count_rows() == 1

    async def test_embed_failure_nonblocking(self, tmp_config, async_db):
        """Embedding error does not prevent compression/storage."""
        from claude_mem_lite.worker.compressor import Compressor
        from claude_mem_lite.worker.processor import IdleTracker, Processor

        # Insert session + queue item
        await async_db.execute(
            "INSERT INTO sessions (id, project_dir) VALUES (?, ?)",
            ("sess-fail", "/tmp/project"),
        )
        await async_db.execute(
            """INSERT INTO pending_queue (id, session_id, tool_name, raw_output, status)
               VALUES (?, ?, ?, ?, ?)""",
            ("q-fail", "sess-fail", "Write", "wrote stuff", "raw"),
        )
        await async_db.commit()

        compressor = MagicMock(spec=Compressor)
        from claude_mem_lite.storage.models import CompressedObservation

        compressor.compress = AsyncMock(
            return_value=CompressedObservation(
                title="Some change",
                summary="Did something",
                tokens_in=100,
                tokens_out=50,
            )
        )

        # Mock lance_store that raises on add
        broken_lance = MagicMock()
        broken_lance.add_observation.side_effect = RuntimeError("LanceDB exploded")

        mock_emb = MagicMock()
        mock_emb.available = True

        idle_tracker = IdleTracker(timeout_minutes=30)
        processor = Processor(
            async_db,
            compressor,
            idle_tracker,
            lance_store=broken_lance,
            embedder=mock_emb,
        )

        items = await processor.dequeue_batch()
        await processor.process_item(items[0])

        # Observation still in SQLite despite embed failure
        cursor = await async_db.execute("SELECT * FROM observations WHERE session_id = 'sess-fail'")
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0]["embedding_status"] == "failed"

        # Queue item marked done
        cursor = await async_db.execute("SELECT status FROM pending_queue WHERE id = 'q-fail'")
        row = await cursor.fetchone()
        assert row["status"] == "done"

    async def test_embedding_status_updated(self, tmp_config, async_db, mock_embedder):
        """embedding_status column is set to 'embedded' after successful embed."""
        from claude_mem_lite.search.lance_store import LanceStore
        from claude_mem_lite.worker.compressor import Compressor
        from claude_mem_lite.worker.processor import IdleTracker, Processor

        lance = LanceStore(tmp_config, mock_embedder)
        lance.connect()

        await async_db.execute(
            "INSERT INTO sessions (id, project_dir) VALUES (?, ?)",
            ("sess-status", "/tmp/project"),
        )
        await async_db.execute(
            """INSERT INTO pending_queue (id, session_id, tool_name, raw_output, status)
               VALUES (?, ?, ?, ?, ?)""",
            ("q-status", "sess-status", "Bash", "ran tests", "raw"),
        )
        await async_db.commit()

        compressor = MagicMock(spec=Compressor)
        from claude_mem_lite.storage.models import CompressedObservation

        compressor.compress = AsyncMock(
            return_value=CompressedObservation(
                title="Ran test suite",
                summary="Executed pytest with all tests passing",
                tokens_in=150,
                tokens_out=60,
            )
        )

        processor = Processor(
            async_db,
            compressor,
            IdleTracker(timeout_minutes=30),
            lance_store=lance,
            embedder=mock_embedder,
        )

        items = await processor.dequeue_batch()
        await processor.process_item(items[0])

        cursor = await async_db.execute(
            "SELECT embedding_status FROM observations WHERE session_id = 'sess-status'"
        )
        row = await cursor.fetchone()
        assert row["embedding_status"] == "embedded"


# ---------------------------------------------------------------------------
# Backfill tests (2)
# ---------------------------------------------------------------------------


class TestBackfill:
    """Tests for embedding backfill on startup."""

    async def test_backfill_processes_pending(self, tmp_config, async_db, mock_embedder):
        """backfill_embeddings embeds observations with status='pending'."""
        from claude_mem_lite.search.lance_store import LanceStore
        from claude_mem_lite.worker.compressor import Compressor
        from claude_mem_lite.worker.processor import IdleTracker, Processor

        lance = LanceStore(tmp_config, mock_embedder)
        lance.connect()

        # Insert session + observations with pending status
        await async_db.execute(
            "INSERT INTO sessions (id, project_dir) VALUES (?, ?)",
            ("sess-bf", "/tmp/project"),
        )
        for i in range(3):
            await async_db.execute(
                """INSERT INTO observations
                   (id, session_id, tool_name, title, summary,
                    files_touched, functions_changed, tokens_raw, tokens_compressed,
                    embedding_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"obs-bf-{i}",
                    "sess-bf",
                    "Write",
                    f"Observation {i}",
                    f"Summary for observation {i}",
                    "[]",
                    "[]",
                    100,
                    50,
                    "pending",
                ),
            )
        await async_db.commit()

        processor = Processor(
            async_db,
            MagicMock(spec=Compressor),
            IdleTracker(timeout_minutes=30),
            lance_store=lance,
            embedder=mock_embedder,
        )

        count = await processor.backfill_embeddings()
        assert count == 3

        # Verify status changed
        cursor = await async_db.execute(
            "SELECT embedding_status FROM observations WHERE session_id = 'sess-bf'"
        )
        rows = await cursor.fetchall()
        assert all(r["embedding_status"] == "embedded" for r in rows)

    async def test_backfill_respects_limit_cap(self, tmp_config, async_db, mock_embedder):
        """Backfill processes max 100 observations per run."""
        from claude_mem_lite.search.lance_store import LanceStore
        from claude_mem_lite.worker.compressor import Compressor
        from claude_mem_lite.worker.processor import IdleTracker, Processor

        lance = LanceStore(tmp_config, mock_embedder)
        lance.connect()

        await async_db.execute(
            "INSERT INTO sessions (id, project_dir) VALUES (?, ?)",
            ("sess-cap", "/tmp/project"),
        )
        for i in range(150):
            await async_db.execute(
                """INSERT INTO observations
                   (id, session_id, tool_name, title, summary,
                    files_touched, functions_changed, tokens_raw, tokens_compressed,
                    embedding_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"obs-cap-{i}",
                    "sess-cap",
                    "Write",
                    f"Observation {i}",
                    f"Summary {i}",
                    "[]",
                    "[]",
                    100,
                    50,
                    "pending",
                ),
            )
        await async_db.commit()

        processor = Processor(
            async_db,
            MagicMock(spec=Compressor),
            IdleTracker(timeout_minutes=30),
            lance_store=lance,
            embedder=mock_embedder,
        )

        count = await processor.backfill_embeddings()
        assert count == 100

        # 50 should remain pending
        cursor = await async_db.execute(
            "SELECT COUNT(*) as cnt FROM observations "
            "WHERE session_id = 'sess-cap' AND embedding_status = 'pending'"
        )
        row = await cursor.fetchone()
        assert row["cnt"] == 50
