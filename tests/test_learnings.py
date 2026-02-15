"""Tests for Phase 6: Learnings Engine."""

from __future__ import annotations

import json
import sqlite3

import pytest

from claude_mem_lite.config import Config
from claude_mem_lite.learnings.healer import CallGraphHealer
from claude_mem_lite.storage.migrations import LATEST_VERSION, MIGRATIONS, migrate

# -----------------------------------------------------------------------
# Migration tests
# -----------------------------------------------------------------------


class TestMigration:
    """Schema migration v4: learnings table columns."""

    def test_migration_applies_cleanly(self, tmp_config):
        """Fresh DB has all Phase 6 columns after migration."""
        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)

        cursor = conn.execute("PRAGMA table_info(learnings)")
        columns = {row[1] for row in cursor.fetchall()}

        assert "times_seen" in columns
        assert "source_sessions" in columns
        assert "is_manual" in columns
        assert LATEST_VERSION >= 4
        conn.close()

    def test_data_migrates_from_source_session_id(self, tmp_config):
        """Existing learnings get source_session_id copied into source_sessions."""
        conn = sqlite3.connect(str(tmp_config.db_path))
        # Apply up to v3 only
        conn.execute("BEGIN EXCLUSIVE")
        for version, sql in MIGRATIONS:
            if version <= 3:
                for raw_stmt in sql.strip().split(";"):
                    stmt = raw_stmt.strip()
                    if stmt:
                        conn.execute(stmt)
                conn.execute(f"PRAGMA user_version = {version}")
        conn.execute("COMMIT")

        # Insert a learning with old schema
        conn.execute(
            "INSERT INTO learnings "
            "(id, category, content, confidence, source_session_id, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("L1", "convention", "Use pytest", 0.7, "session-abc", 1),
        )
        conn.commit()

        # Now apply remaining migrations (v4)
        migrate(conn)

        # Verify data migrated
        row = conn.execute(
            "SELECT source_sessions, times_seen, is_manual FROM learnings WHERE id = 'L1'"
        ).fetchone()
        assert row is not None
        assert json.loads(row[0]) == ["session-abc"]
        assert row[1] == 1  # times_seen default
        assert row[2] == 0  # is_manual default
        conn.close()

    def test_existing_queries_still_work(self, tmp_config):
        """Phase 5 context builder query still works after migration."""
        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)

        # This is the query Phase 5 context builder uses
        cursor = conn.execute(
            "SELECT category, content, confidence FROM learnings "
            "WHERE is_active = 1 AND confidence >= 0.5 "
            "ORDER BY confidence DESC LIMIT 10"
        )
        rows = cursor.fetchall()
        assert rows == []  # Empty but doesn't crash
        conn.close()


# -----------------------------------------------------------------------
# Config tests
# -----------------------------------------------------------------------


class TestConfig:
    """Config has Phase 6 fields."""

    def test_learning_dedup_threshold_default(self):
        """Config has learning_dedup_threshold with default 0.90."""
        config = Config()
        assert config.learning_dedup_threshold == 0.90

    def test_learning_dedup_threshold_override(self, tmp_path):
        """learning_dedup_threshold can be overridden."""
        config = Config(base_dir=tmp_path, learning_dedup_threshold=0.85)
        assert config.learning_dedup_threshold == 0.85


# -----------------------------------------------------------------------
# Model tests
# -----------------------------------------------------------------------


class TestLearningModel:
    """Learning Pydantic model has Phase 6 fields."""

    def test_new_fields_present(self):
        """Learning model has times_seen, source_sessions, is_manual."""
        from claude_mem_lite.storage.models import Learning

        learning = Learning(
            id="L1",
            category="convention",
            content="Use pytest",
            confidence=0.7,
            source_sessions='["session-abc"]',
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
        )
        assert learning.times_seen == 1
        assert learning.source_sessions == '["session-abc"]'
        assert learning.is_manual is False
        assert learning.is_active is True

    def test_source_session_id_removed(self):
        """Learning model no longer has source_session_id field."""
        from claude_mem_lite.storage.models import Learning

        assert "source_session_id" not in Learning.model_fields


# -----------------------------------------------------------------------
# Prompt tests
# -----------------------------------------------------------------------


class TestPrompts:
    """Learning extraction prompt construction."""

    def test_build_extraction_prompt_minimal(self):
        """Prompt with only summary, no observations or existing learnings."""
        from claude_mem_lite.learnings.prompts import build_extraction_prompt

        result = build_extraction_prompt("Did some refactoring", [], [])
        assert "## Session Summary" in result
        assert "Did some refactoring" in result
        assert "## Key Observations" not in result
        assert "## Existing Project Learnings" not in result

    def test_build_extraction_prompt_with_observations(self):
        """Prompt includes observations section when provided."""
        from claude_mem_lite.learnings.prompts import build_extraction_prompt

        observations = [
            {"title": "Fix auth", "summary": "Fixed JWT token refresh"},
            {"title": "Add tests", "summary": "Added unit tests for auth module"},
        ]
        result = build_extraction_prompt("Session work", observations, [])
        assert "## Key Observations" in result
        assert "Fix auth: Fixed JWT token refresh" in result
        assert "Add tests: Added unit tests for auth module" in result

    def test_build_extraction_prompt_with_existing_learnings(self):
        """Prompt includes existing learnings with low-confidence markers."""
        from claude_mem_lite.learnings.prompts import build_extraction_prompt

        existing = [
            {"category": "convention", "content": "Use pytest", "confidence": 0.7},
            {"category": "gotcha", "content": "SQLite needs WAL", "confidence": 0.3},
        ]
        result = build_extraction_prompt("Session work", [], existing)
        assert "## Existing Project Learnings" in result
        assert "[convention] Use pytest" in result
        assert "[low-confidence] [gotcha] SQLite needs WAL" in result

    def test_build_extraction_prompt_caps_observations(self):
        """Prompt limits observations to 8."""
        from claude_mem_lite.learnings.prompts import build_extraction_prompt

        observations = [{"title": f"Obs {i}", "summary": f"Summary {i}"} for i in range(15)]
        result = build_extraction_prompt("Session work", observations, [])
        assert "Obs 7" in result
        assert "Obs 8" not in result

    def test_build_extraction_prompt_caps_learnings(self):
        """Prompt limits existing learnings to 30."""
        from claude_mem_lite.learnings.prompts import build_extraction_prompt

        existing = [
            {"category": "pattern", "content": f"Learning {i}", "confidence": 0.6}
            for i in range(40)
        ]
        result = build_extraction_prompt("Session work", [], existing)
        assert "Learning 29" in result
        assert "Learning 30" not in result

    def test_schema_has_required_fields(self):
        """LEARNING_EXTRACTION_SCHEMA has correct structure."""
        from claude_mem_lite.learnings.prompts import LEARNING_EXTRACTION_SCHEMA

        assert LEARNING_EXTRACTION_SCHEMA["type"] == "object"
        assert "learnings" in LEARNING_EXTRACTION_SCHEMA["properties"]
        item_props = LEARNING_EXTRACTION_SCHEMA["properties"]["learnings"]["items"]["properties"]
        assert "category" in item_props
        assert "content" in item_props
        assert "confidence" in item_props
        assert "contradicts" in item_props


# -----------------------------------------------------------------------
# LanceDB learning methods tests
# -----------------------------------------------------------------------


class TestLanceDBLearnings:
    """LanceDB add_learning and search_learnings methods."""

    def test_add_learning_embeds(self, lance_store):
        """add_learning stores a learning that can be found via search."""
        lance_store.add_learning(
            learning_id="L1",
            category="convention",
            content="Use pytest fixtures with tmp_path",
        )

        results = lance_store.search_learnings(
            query="pytest fixtures",
            limit=5,
        )
        assert len(results) >= 1
        assert results[0]["learning_id"] == "L1"
        assert results[0]["content"] == "Use pytest fixtures with tmp_path"
        assert "score" in results[0]

    def test_search_returns_scores(self, lance_store):
        """search_learnings returns results with similarity scores."""
        lance_store.add_learning("L1", "convention", "Use ruff for formatting")
        lance_store.add_learning("L2", "gotcha", "ORM silently swallows errors")

        results = lance_store.search_learnings("ruff formatting", limit=5)
        assert len(results) >= 1
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_category_filter(self, lance_store):
        """search_learnings can filter by category."""
        lance_store.add_learning("L1", "convention", "Use ruff for formatting")
        lance_store.add_learning("L2", "gotcha", "ORM silently swallows errors")

        results = lance_store.search_learnings(
            query="formatting",
            limit=5,
            category="gotcha",
        )
        # Filter should only return the gotcha category result
        for r in results:
            assert r["category"] == "gotcha"

    def test_search_empty_table_returns_empty(self, lance_store):
        """search_learnings on empty table returns empty list."""
        results = lance_store.search_learnings("anything", limit=5)
        assert results == []


# -----------------------------------------------------------------------
# Engine helper: mock Anthropic client
# -----------------------------------------------------------------------


def _mock_client(response_text: str):
    """Create mock AsyncAnthropic that returns given text."""
    from unittest.mock import AsyncMock, MagicMock

    client = AsyncMock()
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = response_text
    mock_response.content = [mock_content]
    mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
    client.messages.create = AsyncMock(return_value=mock_response)
    return client


async def _seed_learning(
    db,
    learning_id,
    category,
    content,
    confidence=0.5,
    session_id="s1",
):
    """Insert a learning row into the database."""
    await db.execute(
        "INSERT INTO learnings "
        "(id, category, content, confidence, source_session_id, source_sessions, "
        "times_seen, is_manual, is_active, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))",
        (
            learning_id,
            category,
            content,
            confidence,
            session_id,
            json.dumps([session_id]),
            1,
            0,
            1,
        ),
    )
    await db.commit()


# -----------------------------------------------------------------------
# Extraction tests
# -----------------------------------------------------------------------


class TestExtraction:
    """LearningsEngine extraction from session summaries."""

    async def test_valid_response_parsed(self, async_db, tmp_config):
        """Valid API response produces inserted learnings."""
        from claude_mem_lite.learnings.engine import LearningsEngine

        response = json.dumps(
            {
                "learnings": [
                    {
                        "category": "convention",
                        "content": "Always use type hints",
                        "confidence": 0.6,
                        "contradicts": None,
                    },
                ]
            }
        )
        client = _mock_client(response)
        engine = LearningsEngine(async_db, client, None, tmp_config)

        results = await engine.extract_from_session(
            session_id="session-1",
            summary="Added type hints everywhere",
            observations=[],
            project_path="/test/project",
        )

        assert len(results) == 1
        assert results[0]["content"] == "Always use type hints"
        assert results[0]["action"] == "inserted"

        # Verify in DB
        row = await async_db.execute(
            "SELECT content, confidence, category FROM learnings WHERE content = ?",
            ("Always use type hints",),
        )
        row = await row.fetchone()
        assert row is not None
        assert row["category"] == "convention"

    async def test_empty_response_no_crash(self, async_db, tmp_config):
        """Empty learnings array returns empty results."""
        from claude_mem_lite.learnings.engine import LearningsEngine

        response = json.dumps({"learnings": []})
        client = _mock_client(response)
        engine = LearningsEngine(async_db, client, None, tmp_config)

        results = await engine.extract_from_session(
            session_id="session-1",
            summary="Nothing notable happened",
            observations=[],
            project_path="/test/project",
        )

        assert results == []

    async def test_malformed_json_returns_empty(self, async_db, tmp_config):
        """Malformed JSON is handled gracefully."""
        from claude_mem_lite.learnings.engine import LearningsEngine

        client = _mock_client("not valid json at all {{{")
        engine = LearningsEngine(async_db, client, None, tmp_config)

        results = await engine.extract_from_session(
            session_id="session-1",
            summary="Some work",
            observations=[],
            project_path="/test/project",
        )

        assert results == []

    async def test_api_error_non_fatal(self, async_db, tmp_config):
        """API errors don't crash, return empty results."""
        from unittest.mock import AsyncMock

        from claude_mem_lite.learnings.engine import LearningsEngine

        client = AsyncMock()
        client.messages.create = AsyncMock(side_effect=Exception("API down"))
        engine = LearningsEngine(async_db, client, None, tmp_config)

        results = await engine.extract_from_session(
            session_id="session-1",
            summary="Some work",
            observations=[],
            project_path="/test/project",
        )

        assert results == []

    async def test_low_confidence_candidate_skipped(self, async_db, tmp_config):
        """Candidates with confidence < 0.3 are skipped."""
        from claude_mem_lite.learnings.engine import LearningsEngine

        response = json.dumps(
            {
                "learnings": [
                    {
                        "category": "pattern",
                        "content": "Might use decorators sometimes",
                        "confidence": 0.2,
                        "contradicts": None,
                    },
                ]
            }
        )
        client = _mock_client(response)
        engine = LearningsEngine(async_db, client, None, tmp_config)

        results = await engine.extract_from_session(
            session_id="session-1",
            summary="Some work",
            observations=[],
            project_path="/test/project",
        )

        assert results == []

        # Verify NOT in DB
        row = await async_db.execute("SELECT COUNT(*) FROM learnings")
        count = (await row.fetchone())[0]
        assert count == 0

    async def test_multiple_learnings_extracted(self, async_db, tmp_config):
        """Multiple valid learnings in one response are all processed."""
        from claude_mem_lite.learnings.engine import LearningsEngine

        response = json.dumps(
            {
                "learnings": [
                    {
                        "category": "convention",
                        "content": "Use Google-style docstrings",
                        "confidence": 0.6,
                        "contradicts": None,
                    },
                    {
                        "category": "gotcha",
                        "content": "SQLite WAL mode needed for concurrency",
                        "confidence": 0.5,
                        "contradicts": None,
                    },
                ]
            }
        )
        client = _mock_client(response)
        engine = LearningsEngine(async_db, client, None, tmp_config)

        results = await engine.extract_from_session(
            session_id="session-1",
            summary="Set up project conventions",
            observations=[],
            project_path="/test/project",
        )

        assert len(results) == 2
        contents = {r["content"] for r in results}
        assert "Use Google-style docstrings" in contents
        assert "SQLite WAL mode needed for concurrency" in contents


# -----------------------------------------------------------------------
# Dedup tests
# -----------------------------------------------------------------------


class TestDedup:
    """Deduplication and merging of learnings."""

    async def test_new_learning_inserted(self, async_db, tmp_config):
        """New learning with no duplicate is inserted."""
        from claude_mem_lite.learnings.engine import LearningsEngine

        client = _mock_client("{}")  # Not used directly
        engine = LearningsEngine(async_db, client, None, tmp_config)

        candidate = {
            "category": "architecture",
            "content": "Use repository pattern for DB access",
            "confidence": 0.6,
            "contradicts": None,
        }
        result = await engine._process_candidate(candidate, "session-1", "/test/project")

        assert result["action"] == "inserted"
        assert result["content"] == "Use repository pattern for DB access"

        # Verify in DB
        row = await async_db.execute(
            "SELECT confidence, times_seen, source_sessions FROM learnings WHERE content = ?",
            ("Use repository pattern for DB access",),
        )
        row = await row.fetchone()
        assert row is not None
        assert row["confidence"] == 0.5  # INITIAL_CONFIDENCE
        assert row["times_seen"] == 1
        assert json.loads(row["source_sessions"]) == ["session-1"]

    async def test_duplicate_merged_with_boost(self, async_db, tmp_config):
        """Duplicate learning merges and boosts confidence."""
        from claude_mem_lite.learnings.engine import LearningsEngine

        # Seed existing learning
        await _seed_learning(async_db, "L1", "convention", "Use type hints", 0.5, "s1")

        client = _mock_client("{}")
        engine = LearningsEngine(async_db, client, None, tmp_config)

        candidate = {
            "category": "convention",
            "content": "Use type hints",  # Exact substring match
            "confidence": 0.6,
            "contradicts": None,
        }
        result = await engine._process_candidate(candidate, "session-2", "/test/project")

        assert result["action"] == "merged"

        # Verify boosted confidence in DB
        row = await async_db.execute(
            "SELECT confidence, times_seen, source_sessions FROM learnings WHERE id = 'L1'"
        )
        row = await row.fetchone()
        assert row is not None
        assert row["confidence"] > 0.5  # Should be boosted
        assert row["times_seen"] == 2
        sessions = json.loads(row["source_sessions"])
        assert "session-2" in sessions

    async def test_contradiction_penalizes_and_inserts(self, async_db, tmp_config):
        """Contradiction penalizes old, inserts new as fresh entry."""
        from claude_mem_lite.learnings.engine import LearningsEngine

        # Seed existing learning
        await _seed_learning(
            async_db,
            "L1",
            "convention",
            "Use tabs for indentation",
            0.7,
            "s1",
        )

        client = _mock_client("{}")
        engine = LearningsEngine(async_db, client, None, tmp_config)

        candidate = {
            "category": "convention",
            "content": "Use spaces for indentation",
            "confidence": 0.6,
            "contradicts": "Use tabs for indentation",
        }
        result = await engine._process_candidate(candidate, "session-2", "/test/project")

        assert result["action"] == "contradiction"

        # Old learning should be penalized
        row = await async_db.execute("SELECT confidence FROM learnings WHERE id = 'L1'")
        row = await row.fetchone()
        assert row["confidence"] < 0.7  # Penalized from 0.7

        # New learning should be inserted
        row = await async_db.execute(
            "SELECT confidence, content FROM learnings WHERE content = ?",
            ("Use spaces for indentation",),
        )
        row = await row.fetchone()
        assert row is not None
        assert row["confidence"] == 0.5  # INITIAL_CONFIDENCE for new

    async def test_below_threshold_is_new(self, async_db, tmp_path, lance_store):
        """Score below dedup threshold treats as new learning."""
        from claude_mem_lite.learnings.engine import LearningsEngine

        # Use a threshold higher than any possible score (mock embedder
        # returns identical vectors -> score=1.0, so set threshold > 1.0)
        high_threshold_config = Config(
            base_dir=tmp_path / ".claude-mem-high",
            learning_dedup_threshold=1.1,
        )
        high_threshold_config.ensure_dirs()

        # Seed existing learning with embedding
        await _seed_learning(async_db, "L1", "architecture", "Use MVC pattern", 0.5, "s1")
        lance_store.add_learning("L1", "architecture", "Use MVC pattern")

        client = _mock_client("{}")
        engine = LearningsEngine(async_db, client, lance_store, high_threshold_config)

        candidate = {
            "category": "architecture",
            "content": "Use MVC pattern variation",
            "confidence": 0.5,
            "contradicts": None,
        }
        result = await engine._process_candidate(candidate, "session-2", "/test/project")

        assert result["action"] == "inserted"

        # Should have 2 learnings now
        row = await async_db.execute("SELECT COUNT(*) FROM learnings")
        count = (await row.fetchone())[0]
        assert count == 2

    async def test_fallback_substring_match(self, async_db, tmp_config):
        """Without lance_store, falls back to SQL substring match."""
        from claude_mem_lite.learnings.engine import LearningsEngine

        # Seed existing learning
        await _seed_learning(
            async_db,
            "L1",
            "convention",
            "Use type hints everywhere",
            0.5,
            "s1",
        )

        client = _mock_client("{}")
        engine = LearningsEngine(async_db, client, None, tmp_config)

        # Content that is a substring of existing
        candidate = {
            "category": "convention",
            "content": "Use type hints",
            "confidence": 0.6,
            "contradicts": None,
        }
        result = await engine._process_candidate(candidate, "session-2", "/test/project")

        assert result["action"] == "merged"


# -----------------------------------------------------------------------
# Confidence tests
# -----------------------------------------------------------------------


class TestConfidence:
    """Confidence evolution formulas."""

    def test_boost_formula(self):
        """Boost produces correct diminishing returns."""
        from claude_mem_lite.learnings.engine import LearningsEngine

        engine = LearningsEngine.__new__(LearningsEngine)
        # From 0.5: min(0.95, 0.5 + 0.2 * (0.95 - 0.5)) = 0.59
        result = engine._boost_confidence(0.5)
        expected = 0.5 + 0.2 * (0.95 - 0.5)
        assert abs(result - expected) < 1e-6

    def test_boost_near_max(self):
        """Boost near max stays below MAX_AUTO_CONFIDENCE."""
        from claude_mem_lite.learnings.engine import LearningsEngine

        engine = LearningsEngine.__new__(LearningsEngine)
        result = engine._boost_confidence(0.94)
        assert result <= 0.95
        assert result > 0.94

    def test_penalty_clamps_at_minimum(self):
        """Penalty doesn't go below MIN_CONFIDENCE."""
        from claude_mem_lite.learnings.engine import LearningsEngine

        engine = LearningsEngine.__new__(LearningsEngine)
        result = engine._penalize_confidence(0.2)
        assert result == 0.1  # MIN_CONFIDENCE

    def test_penalty_from_high(self):
        """Penalty from a high confidence value."""
        from claude_mem_lite.learnings.engine import LearningsEngine

        engine = LearningsEngine.__new__(LearningsEngine)
        result = engine._penalize_confidence(0.8)
        assert abs(result - 0.5) < 1e-6  # 0.8 - 0.3 = 0.5

    def test_multiple_boosts_converge(self):
        """Multiple boosts converge toward MAX_AUTO_CONFIDENCE."""
        from claude_mem_lite.learnings.engine import LearningsEngine

        engine = LearningsEngine.__new__(LearningsEngine)
        conf = 0.5
        for _ in range(20):
            conf = engine._boost_confidence(conf)
        assert conf > 0.90
        assert conf <= 0.95

    def test_manual_override(self, tmp_config):
        """Manual learnings get confidence=1.0 (tested via direct insert)."""
        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)

        conn.execute(
            "INSERT INTO learnings "
            "(id, category, content, confidence, source_session_id, "
            "source_sessions, times_seen, is_manual, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "L-manual",
                "convention",
                "Always run tests",
                1.0,
                "manual",
                '["manual"]',
                1,
                1,
                1,
            ),
        )
        conn.commit()

        row = conn.execute(
            "SELECT confidence, is_manual FROM learnings WHERE id = 'L-manual'"
        ).fetchone()
        assert row[0] == 1.0
        assert row[1] == 1
        conn.close()


# -----------------------------------------------------------------------
# CallGraphHealer tests
# -----------------------------------------------------------------------


class TestHealerEdgeConfirmation:
    """CallGraphHealer edge confirmation from observations."""

    @pytest.mark.asyncio
    async def test_existing_edge_confirmed(self, async_db):
        """Existing edge gets times_confirmed incremented and confidence boosted."""
        await async_db.execute(
            "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
            ("s1", "/project", "2026-01-01T00:00:00", "active"),
        )
        await async_db.execute(
            "INSERT INTO call_graph (id, caller_file, caller_function, callee_file, callee_function, "
            "resolution, confidence, times_confirmed, source, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "e1",
                "auth.py",
                "authenticate",
                "token.py",
                "create_token",
                "import",
                0.8,
                0,
                "ast",
                "s1",
            ),
        )
        await async_db.commit()

        healer = CallGraphHealer(async_db)
        obs = {"functions_changed": json.dumps(["authenticate", "create_token"])}
        result = await healer.confirm_edges_from_observation(obs, "s1")

        assert result["confirmed"] == 1
        assert result["discovered"] == 0

        cursor = await async_db.execute(
            "SELECT confidence, times_confirmed FROM call_graph WHERE id = 'e1'"
        )
        row = dict(await cursor.fetchone())
        assert row["confidence"] == pytest.approx(0.95)  # 0.8 + 0.15
        assert row["times_confirmed"] == 1

    @pytest.mark.asyncio
    async def test_new_edge_discovered(self, async_db):
        """New edge created at NEW_EDGE_CONFIDENCE when no existing edge."""
        await async_db.execute(
            "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
            ("s1", "/project", "2026-01-01T00:00:00", "active"),
        )
        await async_db.commit()

        healer = CallGraphHealer(async_db)
        obs = {"functions_changed": json.dumps(["func_a", "func_b"])}
        result = await healer.confirm_edges_from_observation(obs, "s1")

        assert result["confirmed"] == 0
        assert result["discovered"] == 1

        cursor = await async_db.execute(
            "SELECT confidence, source FROM call_graph WHERE caller_function = 'func_a'"
        )
        row = dict(await cursor.fetchone())
        assert row["confidence"] == pytest.approx(0.6)
        assert row["source"] == "observation"

    @pytest.mark.asyncio
    async def test_single_function_noop(self, async_db):
        """Observation with < 2 functions does nothing."""
        healer = CallGraphHealer(async_db)
        obs = {"functions_changed": json.dumps(["only_one"])}
        result = await healer.confirm_edges_from_observation(obs, "s1")

        assert result["confirmed"] == 0
        assert result["discovered"] == 0

    @pytest.mark.asyncio
    async def test_empty_functions_noop(self, async_db):
        """Observation with no functions_changed does nothing."""
        healer = CallGraphHealer(async_db)
        obs = {}
        result = await healer.confirm_edges_from_observation(obs, "s1")

        assert result["confirmed"] == 0
        assert result["discovered"] == 0

    @pytest.mark.asyncio
    async def test_confidence_capped_at_one(self, async_db):
        """Confirmation boost doesn't exceed 1.0."""
        await async_db.execute(
            "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
            ("s1", "/project", "2026-01-01T00:00:00", "active"),
        )
        await async_db.execute(
            "INSERT INTO call_graph (id, caller_file, caller_function, callee_file, callee_function, "
            "resolution, confidence, times_confirmed, source, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("e1", "a.py", "func_a", "b.py", "func_b", "direct", 0.95, 3, "ast", "s1"),
        )
        await async_db.commit()

        healer = CallGraphHealer(async_db)
        obs = {"functions_changed": json.dumps(["func_a", "func_b"])}
        await healer.confirm_edges_from_observation(obs, "s1")

        cursor = await async_db.execute("SELECT confidence FROM call_graph WHERE id = 'e1'")
        row = dict(await cursor.fetchone())
        assert row["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_reverse_direction_matches(self, async_db):
        """Edge lookup matches regardless of caller/callee direction."""
        await async_db.execute(
            "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
            ("s1", "/project", "2026-01-01T00:00:00", "active"),
        )
        await async_db.execute(
            "INSERT INTO call_graph (id, caller_file, caller_function, callee_file, callee_function, "
            "resolution, confidence, times_confirmed, source, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("e1", "a.py", "func_a", "b.py", "func_b", "direct", 0.7, 0, "ast", "s1"),
        )
        await async_db.commit()

        healer = CallGraphHealer(async_db)
        # Reversed order: func_b before func_a
        obs = {"functions_changed": json.dumps(["func_b", "func_a"])}
        result = await healer.confirm_edges_from_observation(obs, "s1")

        assert result["confirmed"] == 1
        assert result["discovered"] == 0


class TestHealerStaleDecay:
    """CallGraphHealer stale edge decay."""

    @pytest.mark.asyncio
    async def test_stale_observation_edges_decayed(self, async_db):
        """Observation-sourced edges older than threshold are decayed."""
        for i in range(10):
            await async_db.execute(
                "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
                (f"s{i}", "/project", f"2026-01-{i + 1:02d}T00:00:00", "closed"),
            )
        await async_db.execute(
            "INSERT INTO call_graph (id, caller_file, caller_function, callee_file, callee_function, "
            "resolution, confidence, times_confirmed, source, session_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "e1",
                "",
                "func_a",
                "",
                "func_b",
                "unresolved",
                0.6,
                0,
                "observation",
                "s0",
                "2025-12-01T00:00:00",
            ),
        )
        await async_db.commit()

        healer = CallGraphHealer(async_db)
        decayed = await healer.decay_stale_edges("/project")

        assert decayed == 1
        cursor = await async_db.execute("SELECT confidence FROM call_graph WHERE id = 'e1'")
        row = dict(await cursor.fetchone())
        assert row["confidence"] == pytest.approx(0.55)  # 0.6 - 0.05

    @pytest.mark.asyncio
    async def test_ast_edges_untouched(self, async_db):
        """AST-sourced edges are not decayed even if old."""
        for i in range(10):
            await async_db.execute(
                "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
                (f"s{i}", "/project", f"2026-01-{i + 1:02d}T00:00:00", "closed"),
            )
        await async_db.execute(
            "INSERT INTO call_graph (id, caller_file, caller_function, callee_file, callee_function, "
            "resolution, confidence, times_confirmed, source, session_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "e1",
                "a.py",
                "func_a",
                "b.py",
                "func_b",
                "direct",
                1.0,
                0,
                "ast",
                "s0",
                "2025-12-01T00:00:00",
            ),
        )
        await async_db.commit()

        healer = CallGraphHealer(async_db)
        decayed = await healer.decay_stale_edges("/project")

        assert decayed == 0
        cursor = await async_db.execute("SELECT confidence FROM call_graph WHERE id = 'e1'")
        row = dict(await cursor.fetchone())
        assert row["confidence"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_insufficient_history_no_decay(self, async_db):
        """Fewer than STALE_SESSION_THRESHOLD sessions = no decay."""
        for i in range(5):
            await async_db.execute(
                "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
                (f"s{i}", "/project", f"2026-01-{i + 1:02d}T00:00:00", "closed"),
            )
        await async_db.execute(
            "INSERT INTO call_graph (id, caller_file, caller_function, callee_file, callee_function, "
            "resolution, confidence, times_confirmed, source, session_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "e1",
                "",
                "func_a",
                "",
                "func_b",
                "unresolved",
                0.6,
                0,
                "observation",
                "s0",
                "2025-12-01T00:00:00",
            ),
        )
        await async_db.commit()

        healer = CallGraphHealer(async_db)
        decayed = await healer.decay_stale_edges("/project")

        assert decayed == 0

    @pytest.mark.asyncio
    async def test_confirmed_edges_not_decayed(self, async_db):
        """Observation edges with times_confirmed > 0 are not decayed."""
        for i in range(10):
            await async_db.execute(
                "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
                (f"s{i}", "/project", f"2026-01-{i + 1:02d}T00:00:00", "closed"),
            )
        await async_db.execute(
            "INSERT INTO call_graph (id, caller_file, caller_function, callee_file, callee_function, "
            "resolution, confidence, times_confirmed, source, session_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "e1",
                "",
                "func_a",
                "",
                "func_b",
                "unresolved",
                0.6,
                2,
                "observation",
                "s0",
                "2025-12-01T00:00:00",
            ),
        )
        await async_db.commit()

        healer = CallGraphHealer(async_db)
        decayed = await healer.decay_stale_edges("/project")

        assert decayed == 0


# -----------------------------------------------------------------------
# Worker integration tests
# -----------------------------------------------------------------------


class TestWorkerIntegration:
    """Processor pipeline integration with Phase 6 hooks."""

    async def test_post_observation_fires(self, async_db):
        """_post_observation calls healer after compression."""
        from unittest.mock import AsyncMock, MagicMock

        from claude_mem_lite.storage.models import (
            CompressedObservation,
            FunctionChangeRecord,
            PendingQueueItem,
        )
        from claude_mem_lite.worker.processor import IdleTracker, Processor

        healer = AsyncMock()
        healer.confirm_edges_from_observation = AsyncMock(
            return_value={"confirmed": 0, "discovered": 1}
        )

        processor = Processor(
            db=async_db,
            compressor=MagicMock(),
            idle_tracker=IdleTracker(),
            call_graph_healer=healer,
        )

        item = PendingQueueItem(
            id="q1",
            session_id="s1",
            tool_name="Write",
            raw_output="x",
            files_touched="[]",
            created_at="2026-01-01T00:00:00",
        )
        result = CompressedObservation(
            title="Test",
            summary="Test summary",
            functions_changed=[FunctionChangeRecord(file="a.py", name="func_a", action="new")],
        )

        await processor._post_observation(item, result)
        healer.confirm_edges_from_observation.assert_called_once()

    async def test_post_summarization_fires(self, async_db):
        """_post_summarization calls learnings engine after summary."""
        from unittest.mock import AsyncMock, MagicMock

        from claude_mem_lite.worker.processor import IdleTracker, Processor

        # Create a session
        await async_db.execute(
            "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
            ("s1", "/project", "2026-01-01T00:00:00", "active"),
        )
        await async_db.commit()

        engine = AsyncMock()
        engine.extract_from_session = AsyncMock(return_value=[])

        processor = Processor(
            db=async_db,
            compressor=MagicMock(),
            idle_tracker=IdleTracker(),
            learnings_engine=engine,
        )

        await processor._post_summarization("s1", "Session summary text")
        engine.extract_from_session.assert_called_once()

    async def test_post_hooks_nonfatal(self, async_db):
        """Exceptions in post-hooks don't break the pipeline."""
        from unittest.mock import AsyncMock, MagicMock

        from claude_mem_lite.storage.models import CompressedObservation, PendingQueueItem
        from claude_mem_lite.worker.processor import IdleTracker, Processor

        healer = AsyncMock()
        healer.confirm_edges_from_observation = AsyncMock(side_effect=Exception("Heal failed"))

        engine = AsyncMock()
        engine.extract_from_session = AsyncMock(side_effect=Exception("Extract failed"))

        processor = Processor(
            db=async_db,
            compressor=MagicMock(),
            idle_tracker=IdleTracker(),
            learnings_engine=engine,
            call_graph_healer=healer,
        )

        # _post_observation should not raise
        item = PendingQueueItem(
            id="q1",
            session_id="s1",
            tool_name="Write",
            raw_output="x",
            files_touched="[]",
            created_at="2026-01-01T00:00:00",
        )
        result = CompressedObservation(title="Test", summary="Summary")
        await processor._post_observation(item, result)  # Should not raise

        # _post_summarization should not raise
        await async_db.execute(
            "INSERT INTO sessions (id, project_dir, started_at, status) VALUES (?, ?, ?, ?)",
            ("s1", "/project", "2026-01-01T00:00:00", "active"),
        )
        await async_db.commit()
        await processor._post_summarization("s1", "Summary")  # Should not raise


# -----------------------------------------------------------------------
# CLI learnings command tests
# -----------------------------------------------------------------------


class TestCLILearnings:
    """CLI functions for managing learnings."""

    def test_list_learnings(self, tmp_config):
        """list_learnings returns active learnings."""
        from claude_mem_lite.cli.learnings_cmd import add_learning, list_learnings

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)

        add_learning(conn, "convention", "Use type hints")
        add_learning(conn, "gotcha", "SQLite needs WAL")

        results = list_learnings(conn)
        assert len(results) == 2
        contents = {r["content"] for r in results}
        assert "Use type hints" in contents
        assert "SQLite needs WAL" in contents
        conn.close()

    def test_list_learnings_filter_by_category(self, tmp_config):
        """list_learnings filters by category when provided."""
        from claude_mem_lite.cli.learnings_cmd import add_learning, list_learnings

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)

        add_learning(conn, "convention", "Use type hints")
        add_learning(conn, "gotcha", "SQLite needs WAL")

        results = list_learnings(conn, category="convention")
        assert len(results) == 1
        assert results[0]["content"] == "Use type hints"
        assert results[0]["category"] == "convention"
        conn.close()

    def test_add_manual_learning(self, tmp_config):
        """add_learning sets confidence=1.0 and is_manual=True."""
        from claude_mem_lite.cli.learnings_cmd import add_learning

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)

        learning_id = add_learning(conn, "convention", "Always run tests")
        assert learning_id  # non-empty string

        row = conn.execute(
            "SELECT confidence, is_manual, is_active FROM learnings WHERE id = ?",
            (learning_id,),
        ).fetchone()
        assert row[0] == 1.0
        assert row[1] == 1  # is_manual
        assert row[2] == 1  # is_active
        conn.close()

    def test_edit_learning(self, tmp_config):
        """edit_learning updates content."""
        from claude_mem_lite.cli.learnings_cmd import add_learning, edit_learning

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)

        learning_id = add_learning(conn, "convention", "Old content")
        result = edit_learning(conn, learning_id, "New content")
        assert result is True

        row = conn.execute("SELECT content FROM learnings WHERE id = ?", (learning_id,)).fetchone()
        assert row[0] == "New content"

        # Non-existent ID returns False
        result = edit_learning(conn, "nonexistent", "Content")
        assert result is False
        conn.close()

    def test_remove_learning(self, tmp_config):
        """remove_learning soft-deletes (sets is_active=0)."""
        from claude_mem_lite.cli.learnings_cmd import (
            add_learning,
            list_learnings,
            remove_learning,
        )

        conn = sqlite3.connect(str(tmp_config.db_path))
        migrate(conn)

        learning_id = add_learning(conn, "convention", "Test learning")
        assert len(list_learnings(conn)) == 1

        result = remove_learning(conn, learning_id)
        assert result is True
        assert len(list_learnings(conn)) == 0

        # Verify it's still in DB but inactive
        row = conn.execute(
            "SELECT is_active FROM learnings WHERE id = ?", (learning_id,)
        ).fetchone()
        assert row[0] == 0
        conn.close()
