"""Shared fixtures for Phase 0 tests."""

import pytest

from claude_mem_lite.config import Config
from claude_mem_lite.logging.logger import MemLogger
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
