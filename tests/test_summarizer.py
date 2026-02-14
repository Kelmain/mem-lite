"""Tests for session summarization."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from claude_mem_lite.config import Config
from claude_mem_lite.storage.models import SessionSummary
from claude_mem_lite.storage.sqlite_store import SQLiteStore
from claude_mem_lite.worker.prompts import SUMMARY_SCHEMA
from claude_mem_lite.worker.summarizer import Summarizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_config(tmp_path):
    """Config pointing to a temp directory -- fresh DB per test."""
    config = Config(base_dir=tmp_path / ".claude-mem")
    config.ensure_dirs()
    return config


@pytest.fixture
def store(tmp_config):
    """SQLiteStore with migrated DB."""
    s = SQLiteStore(tmp_config.db_path)
    yield s
    s.close()


@pytest.fixture
async def async_db(tmp_config, store):
    """Async SQLite connection. Store fixture ensures migrations run first."""
    db = await aiosqlite.connect(str(tmp_config.db_path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await db.execute("PRAGMA busy_timeout=3000")
    yield db
    await db.close()


def _make_mock_response(text: str, input_tokens: int = 50, output_tokens: int = 30):
    """Create a mock Anthropic API response."""
    content_block = MagicMock()
    content_block.text = text
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    response = MagicMock()
    response.content = [content_block]
    response.usage = usage
    return response


@pytest.fixture
def mock_compressor():
    """Mock compressor with AsyncMock client.messages.create."""
    comp = MagicMock()
    comp.client = MagicMock()
    comp.client.messages = MagicMock()
    comp.client.messages.create = AsyncMock()
    comp.model = "claude-haiku-4-5"
    return comp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSummarizer:
    """Tests for the Summarizer class."""

    async def test_summarize_session_with_observations(self, store, async_db, mock_compressor):
        """Summarizing a session with observations returns correct SessionSummary fields."""
        session = store.create_session("/project")
        store.create_observation(
            session_id=session.id,
            tool_name="Write",
            title="Add auth middleware",
            summary="Added JWT-based auth middleware.",
        )
        store.create_observation(
            session_id=session.id,
            tool_name="Read",
            title="Review config",
            summary="Checked database config settings.",
        )

        api_response = {
            "summary": "Implemented JWT auth and reviewed config.",
            "key_files": ["src/auth.py", "config.py"],
            "key_decisions": ["Use JWT over session tokens"],
        }
        mock_compressor.client.messages.create.return_value = _make_mock_response(
            json.dumps(api_response)
        )

        summarizer = Summarizer(async_db, mock_compressor)
        result = await summarizer.summarize_session(session.id)

        assert isinstance(result, SessionSummary)
        assert result.summary == "Implemented JWT auth and reviewed config."
        assert result.key_files == ["src/auth.py", "config.py"]
        assert result.key_decisions == ["Use JWT over session tokens"]

    async def test_summarize_empty_session(self, store, async_db, mock_compressor):
        """Empty session returns default summary without calling the API."""
        session = store.create_session("/project")

        summarizer = Summarizer(async_db, mock_compressor)
        result = await summarizer.summarize_session(session.id)

        assert isinstance(result, SessionSummary)
        assert result.summary == "No observations captured."
        assert result.key_files == []
        assert result.key_decisions == []
        mock_compressor.client.messages.create.assert_not_called()

    async def test_summary_stored_in_db(self, store, async_db, mock_compressor):
        """After summarization, the session row has summary set and status closed."""
        session = store.create_session("/project")
        store.create_observation(
            session_id=session.id,
            tool_name="Write",
            title="Fix bug",
            summary="Fixed off-by-one error.",
        )

        api_response = {
            "summary": "Fixed critical bug in parser.",
            "key_files": ["parser.py"],
            "key_decisions": [],
        }
        mock_compressor.client.messages.create.return_value = _make_mock_response(
            json.dumps(api_response)
        )

        summarizer = Summarizer(async_db, mock_compressor)
        await summarizer.summarize_session(session.id)

        # Read session directly from the async DB to verify persistence
        cursor = await async_db.execute(
            "SELECT summary, status FROM sessions WHERE id = ?",
            (session.id,),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row["summary"] == "Fixed critical bug in parser."
        assert row["status"] == "closed"

    async def test_summarize_uses_structured_outputs(self, store, async_db, mock_compressor):
        """The API call includes extra_body with output_config containing SUMMARY_SCHEMA."""
        session = store.create_session("/project")
        store.create_observation(
            session_id=session.id,
            tool_name="Write",
            title="Add feature",
            summary="Added new feature.",
        )

        api_response = {
            "summary": "Added a new feature.",
            "key_files": ["feature.py"],
            "key_decisions": [],
        }
        mock_compressor.client.messages.create.return_value = _make_mock_response(
            json.dumps(api_response)
        )

        summarizer = Summarizer(async_db, mock_compressor)
        await summarizer.summarize_session(session.id)

        mock_compressor.client.messages.create.assert_called_once()
        call_kwargs = mock_compressor.client.messages.create.call_args
        extra_body = call_kwargs.kwargs["extra_body"]
        assert extra_body == {
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": SUMMARY_SCHEMA,
                }
            }
        }
