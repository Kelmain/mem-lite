"""Shared fixtures for all test phases."""

import sqlite3

import aiosqlite
import pytest

from claude_mem_lite.config import Config
from claude_mem_lite.logging.logger import MemLogger
from claude_mem_lite.storage.migrations import migrate
from claude_mem_lite.storage.sqlite_store import SQLiteStore


@pytest.fixture
def tmp_config(tmp_path):
    """Config pointing to a temp directory — fresh DB per test."""
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
def logger(tmp_config, store):
    """MemLogger writing to temp dir."""
    return MemLogger(tmp_config.log_dir, store.conn)


# -----------------------------------------------------------------------
# Phase 4: Search fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def mock_embedder(tmp_config):
    """Mock embedder returning random L2-normalized vectors — no real model needed."""
    import numpy as np

    from claude_mem_lite.search.embedder import Embedder

    e = Embedder(tmp_config)
    e._available = True
    e._model = None
    dim = tmp_config.embedding_dim

    def fake_embed_texts(texts, query_type="document"):
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((len(texts), dim))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / norms).tolist()

    def fake_embed_single(text, query_type="document"):
        return fake_embed_texts([text], query_type)[0]

    e.embed_texts = fake_embed_texts  # type: ignore[method-assign]
    e.embed_single = fake_embed_single  # type: ignore[method-assign]
    return e


@pytest.fixture
def lance_store(tmp_config, mock_embedder):
    """LanceDB store with mock embedder for fast tests."""
    from claude_mem_lite.search.lance_store import LanceStore

    store = LanceStore(tmp_config, mock_embedder)
    store.connect()
    return store


@pytest.fixture
async def async_db(tmp_config):
    """Async SQLite connection with migrated schema."""
    sync_conn = sqlite3.connect(str(tmp_config.db_path))
    migrate(sync_conn)
    sync_conn.close()

    db = await aiosqlite.connect(str(tmp_config.db_path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await db.execute("PRAGMA busy_timeout=3000")

    yield db
    await db.close()
