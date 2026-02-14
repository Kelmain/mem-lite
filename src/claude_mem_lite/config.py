"""Configuration management for claude-mem-lite."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Central configuration with path properties and token budgets."""

    base_dir: Path = field(default_factory=lambda: Path.home() / ".claude-mem")

    # Context token budget (2000 total)
    ctx_session_index: int = 300
    ctx_function_map: int = 500
    ctx_learnings: int = 300
    ctx_observations: int = 600
    ctx_call_graph: int = 300

    # Compression
    compression_model: str = "claude-haiku-4-5"
    ab_test_enabled: bool = False

    # Embedding (Phase 4 — config defined here for forward-compat)
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024

    @property
    def db_path(self) -> Path:
        return self.base_dir / "claude-mem.db"

    @property
    def lance_path(self) -> Path:
        return self.base_dir / "lance"

    @property
    def socket_path(self) -> Path:
        return self.base_dir / "worker.sock"

    @property
    def pid_path(self) -> Path:
        return self.base_dir / "worker.pid"

    @property
    def log_dir(self) -> Path:
        return self.base_dir / "logs"

    def ensure_dirs(self) -> None:
        """Create directory tree if it doesn't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
