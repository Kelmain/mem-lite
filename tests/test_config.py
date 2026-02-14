"""Tests for Config — 3 tests."""

from pathlib import Path

from claude_mem_lite.config import Config


class TestConfig:
    def test_default_paths(self):
        """Default base_dir is ~/.claude-mem with correct derived paths."""
        config = Config()
        assert config.base_dir == Path.home() / ".claude-mem"
        assert config.db_path == config.base_dir / "claude-mem.db"
        assert config.socket_path == config.base_dir / "worker.sock"
        assert config.pid_path == config.base_dir / "worker.pid"
        assert config.log_dir == config.base_dir / "logs"
        assert config.lance_path == config.base_dir / "lance"

    def test_custom_base_dir(self, tmp_path):
        """Custom base_dir propagates to all derived paths."""
        custom = tmp_path / "custom-mem"
        config = Config(base_dir=custom)
        assert config.base_dir == custom
        assert config.db_path == custom / "claude-mem.db"
        assert config.log_dir == custom / "logs"

    def test_ensure_dirs_creates_structure(self, tmp_path):
        """ensure_dirs creates base_dir and log_dir."""
        config = Config(base_dir=tmp_path / "new-dir")
        assert not config.base_dir.exists()
        config.ensure_dirs()
        assert config.base_dir.exists()
        assert config.log_dir.exists()
