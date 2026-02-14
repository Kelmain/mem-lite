"""Tests for the worker FastAPI server."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import httpx
import pytest

from claude_mem_lite.config import Config
from claude_mem_lite.storage.models import HealthResponse, QueueStats, SessionSummary
from claude_mem_lite.storage.sqlite_store import SQLiteStore
from claude_mem_lite.worker import server as server_mod
from claude_mem_lite.worker.processor import IdleTracker
from claude_mem_lite.worker.server import app


@pytest.fixture
def tmp_config(tmp_path):
    config = Config(base_dir=tmp_path / ".claude-mem")
    config.ensure_dirs()
    return config


@pytest.fixture
def store(tmp_config):
    s = SQLiteStore(tmp_config.db_path)
    yield s
    s.close()


@pytest.fixture
async def configured_app(tmp_config, store):
    """Set up app.state for testing without running full lifespan."""
    db = await aiosqlite.connect(str(tmp_config.db_path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await db.execute("PRAGMA busy_timeout=3000")

    mock_compressor = MagicMock()
    mock_compressor.client = MagicMock()
    mock_compressor.model = "test-model"
    mock_compressor.close = AsyncMock()

    mock_summarizer = MagicMock()

    idle_tracker = IdleTracker(timeout_minutes=999)

    app.state.config = tmp_config
    app.state.db = db
    app.state.compressor = mock_compressor
    app.state.summarizer = mock_summarizer
    app.state.idle_tracker = idle_tracker
    app.state.processor = MagicMock()

    yield app

    await db.close()


@pytest.fixture
async def client(configured_app):
    """Async HTTP client for testing."""
    transport = httpx.ASGITransport(app=configured_app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_endpoint(tmp_config, store, client, configured_app):
    """GET /api/health returns status, queue_depth, and uptime_s."""
    # Seed data: create session, enqueue a raw item
    session = store.create_session("/test/project")
    store.enqueue(session_id=session.id, tool_name="Read", raw_output="output1")

    # Set _start_time so uptime_s is computable
    server_mod._start_time = 0.0

    response = await client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    health = HealthResponse(**data)
    assert health.status == "ok"
    assert health.queue_depth == 1
    assert health.uptime_s >= 0


async def test_context_placeholder(client):
    """GET /api/context returns empty context placeholder."""
    response = await client.get("/api/context")

    assert response.status_code == 200
    data = response.json()
    assert data == {"context": "", "tokens": 0}


async def test_queue_stats(tmp_config, store, client):
    """GET /api/queue/stats returns counts by status."""
    # Seed items with different statuses
    session = store.create_session("/test/project")
    store.enqueue(session_id=session.id, tool_name="Read", raw_output="o1")
    item2 = store.enqueue(session_id=session.id, tool_name="Write", raw_output="o2")
    item3 = store.enqueue(session_id=session.id, tool_name="Bash", raw_output="o3")
    store.complete_queue_item(item2.id)  # done
    store.fail_queue_item(item3.id)  # error

    response = await client.get("/api/queue/stats")

    assert response.status_code == 200
    data = response.json()
    stats = QueueStats(**data)
    assert stats.raw == 1
    assert stats.done == 1
    assert stats.error == 1
    assert stats.processing == 0


async def test_summarize_endpoint(client, configured_app):
    """POST /api/summarize returns session summary from summarizer."""
    expected_summary = SessionSummary(
        summary="Implemented auth module",
        key_files=["src/auth.py"],
        key_decisions=["Use JWT"],
    )
    configured_app.state.summarizer.summarize_session = AsyncMock(return_value=expected_summary)

    response = await client.post("/api/summarize", params={"session_id": "test-123"})

    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == "Implemented auth module"
    assert data["key_files"] == ["src/auth.py"]
    assert data["key_decisions"] == ["Use JWT"]
    configured_app.state.summarizer.summarize_session.assert_awaited_once_with("test-123")
