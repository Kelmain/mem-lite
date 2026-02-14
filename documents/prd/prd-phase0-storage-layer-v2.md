# PRD: Phase 0 â€” Storage Layer (v2 â€” Review-Corrected)

**Project**: claude-mem-lite (fork of claude-mem, Python rewrite)
**Phase**: 0 of 9
**Status**: Ready for implementation
**Dependencies**: None (foundation layer)
**Estimated effort**: 1 session (~4-6 hours)
**Python**: 3.14.3 (latest stable, released Feb 3, 2026)

---

## Changelog from v1

| Item | v1 (incorrect/outdated) | v2 (corrected) |
|------|------------------------|----------------|
| Python version | `>=3.10` | `>=3.14` (3.14.3 latest stable) |
| LanceDB version | "~0.15-0.17" | 0.29.1 (still Alpha, `>=0.25` minimum) |
| LanceDB in Phase 0 | Install + create schemas | **Deferred to Phase 4** â€” install only pydantic |
| Embedding dim | 256 (Matryoshka truncation) | **1024** (full native dimension) |
| Qwen3-Embedding context | "32K context window" | **8K** (`max_length=8192` per HuggingFace) |
| Qwen3-Embedding validation date | "Feb 2025" | **June 2025** (model released June 5, 2025) |
| Primary keys | Mixed (TEXT UUID vs INTEGER AUTO) | **TEXT UUID** everywhere for consistency |
| sentence-transformers | `>=3.0` | `>=2.7.0` (compatible per HF model card) |
| Token counting | "tiktoken or word-based" | `len(text) / 4` approximation (tiktoken is OpenAI-specific) |
| dequeue_batch | SELECT + UPDATE in transaction | **`UPDATE ... RETURNING`** (atomic, SQLite 3.35+, ships w/ Python 3.14) |
| Test count / effort | 39 tests in ~2-4 hours | 39 tests in **~4-6 hours** |
| macOS SDPA bug | Not mentioned | **N/A** â€” target is Ubuntu |
| Matryoshka dims | "1024â†’512â†’256â†’128â†’64â†’32" | Any dimension 32â€“1024 (not predefined steps) |

---

## 1. Purpose & Context

### 1.1 What this phase delivers
The storage layer is the foundation that every other phase plugs into. It provides:
- SQLite database with all tables, indexes, and migration system
- Pydantic models for type-safe data passing between components
- Configuration management (paths, token budgets, model settings)
- Structured logging (JSONL file + SQLite event_log)

### 1.2 What this phase does NOT deliver
- **LanceDB schemas** â€” deferred to Phase 4 when embeddings are actually needed
- **Embedding model loading** â€” deferred to Phase 4
- **Vector search** â€” deferred to Phase 4

**Rationale**: LanceDB is still Alpha (0.29.1), pulls in pyarrow (~100MB+), and we don't use any LanceDB features until Phase 4 (4-6 sessions away). Deferring keeps Phase 0 lean, fast to install, and avoids coupling to an unstable API. The `lance_store.py` module will be a thin wrapper added in Phase 4 â€” refactoring cost is minimal.

### 1.3 Why it exists separately
Hooks (Phase 1) need to write to SQLite directly. The worker (Phase 3) needs to read from and write to the same database. Search (Phase 4) needs LanceDB tables. By building storage first, we avoid coupling data access to any single consumer and can test each table's CRUD in isolation.

### 1.4 Relationship to claude-mem (original)
claude-mem uses SQLite + ChromaDB (for vector search) + Express.js worker on port 37777. We replace:
- ChromaDB â†’ LanceDB (embedded, no server, Apache Arrow, Tantivy FTS built-in) â€” **Phase 4**
- Express.js â†’ FastAPI over Unix Domain Socket (Phase 3)
- TypeScript â†’ Python throughout
- Node.js/Bun runtime â†’ pure Python + stdlib sqlite3

---

## 2. Technical Specification

### 2.1 Project Structure

```
claude-mem-lite/
â”œâ”€â”€ pyproject.toml
â”œâ”€â”€ src/
â”‚   â””â”€â”€ claude_mem_lite/
â”‚       â”œâ”€â”€ __init__.py
â”‚       â”œâ”€â”€ config.py              # This phase
â”‚       â”œâ”€â”€ storage/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â”œâ”€â”€ models.py          # This phase
â”‚       â”‚   â”œâ”€â”€ migrations.py      # This phase
â”‚       â”‚   â””â”€â”€ sqlite_store.py    # This phase
â”‚       â”œâ”€â”€ logging/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â””â”€â”€ logger.py          # This phase
â”‚       â”œâ”€â”€ ast_tracker/           # Phase 2
â”‚       â”œâ”€â”€ worker/                # Phase 3
â”‚       â”œâ”€â”€ hooks/                 # Phase 1
â”‚       â”œâ”€â”€ search/                # Phase 4
â”‚       â”œâ”€â”€ context/               # Phase 5
â”‚       â”œâ”€â”€ learnings/             # Phase 6
â”‚       â””â”€â”€ cli/                   # Phase 8
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ conftest.py
â”‚   â””â”€â”€ test_storage.py           # This phase
â””â”€â”€ plugin/                        # Phase 1
```

**Note**: No `lance_store.py` in this phase. Added in Phase 4.

### 2.2 Configuration (`config.py`)

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    base_dir: Path = field(default_factory=lambda: Path.home() / ".claude-mem")

    # Derived paths
    @property
    def db_path(self) -> Path:
        return self.base_dir / "claude-mem.db"

    @property
    def lance_path(self) -> Path:       # Used in Phase 4
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

    # Context token budget (2000 total)
    ctx_session_index: int = 300
    ctx_function_map: int = 500
    ctx_learnings: int = 300
    ctx_observations: int = 600
    ctx_call_graph: int = 300

    # Compression
    compression_model: str = "claude-haiku-4-5"
    ab_test_enabled: bool = False

    # Embedding (Phase 4 â€” config defined here for forward-compat)
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024  # Full native dimension â€” no truncation at our scale

    def ensure_dirs(self) -> None:
        """Create directory tree if it doesn't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
```

**Design notes:**

1. **Token budget of 2000**: The reference target is ~1,500 tokens (claude-mem v7). A 2000-token budget gives headroom. Claude Code has a ~200k context window, so 2000 tokens is ~1% â€” acceptable. Should be validated empirically in Phase 5.

2. **Embedding dim 1024 (full, no truncation)**: At our scale (<10K observations), 1024d Ã— 4 bytes Ã— 10K = ~40MB. Negligible storage cost, sub-millisecond search, maximum retrieval quality. Matryoshka truncation only makes sense at 1M+ vectors.

3. **Embedding model context window**: Qwen3-Embedding-0.6B has an **8K token context** (not 32K as previously claimed). The base Qwen3 architecture supports 32K, but the embedding model was trained/validated at 8192. This is sufficient for our observations (compressed to ~500 tokens) but may require chunking for very large raw tool outputs.

4. **`base_dir` default `~/.claude-mem`**: Matches original claude-mem. Compatible with their data directory if we want migration.

### 2.3 Data Models (`storage/models.py`)

All models use Pydantic v2 for validation and serialization. All primary keys are TEXT (UUID strings) for consistency.

#### Enums

| Enum | Values | Usage |
|------|--------|-------|
| `SessionStatus` | `active`, `closed`, `error` | Session lifecycle |
| `QueueStatus` | `raw`, `processing`, `done`, `error` | Pending queue items |
| `ChangeType` | `new`, `modified`, `deleted`, `unchanged` | Function map tracking |
| `FunctionKind` | `function`, `method`, `async_function`, `async_method`, `class` | AST extraction |
| `CallResolution` | `direct`, `self_method`, `import`, `unresolved` | Call graph edges |
| `LearningCategory` | `architecture`, `convention`, `gotcha`, `dependency`, `pattern` | Learning classification |

#### Models

**Session**
```
id: str (UUID)
project_dir: str
started_at: datetime
ended_at: Optional[datetime]
status: SessionStatus
summary: Optional[str]
observation_count: int = 0
```

**Observation**
```
id: str (UUID)
session_id: str (FK â†’ sessions)
tool_name: str
title: str
summary: str
detail: Optional[str]
files_touched: str (JSON array)
functions_changed: str (JSON array)
tokens_raw: int
tokens_compressed: int
created_at: datetime
```

**FunctionMapEntry**
```
id: str (UUID)
session_id: str (FK â†’ sessions)
file_path: str
qualified_name: str
kind: FunctionKind
signature: str
docstring: Optional[str]
body_hash: str (MD5 of ast.dump, full 32 hex chars)
decorators: str (JSON array)
change_type: ChangeType
updated_at: datetime
```

**CallGraphEdge**
```
id: str (UUID)
caller_file: str
caller_function: str
callee_file: str
callee_function: str
resolution: CallResolution
confidence: float = 1.0          # Phase 6 adds self-healing
times_confirmed: int = 0         # Phase 6 adds confirmation tracking
source: str = "ast"              # "ast" or "observation"
session_id: str (FK â†’ sessions)
created_at: datetime
```

**Learning**
```
id: str (UUID)
category: LearningCategory
content: str
confidence: float = 0.5
source_session_id: str
is_active: bool = True
created_at: datetime
updated_at: datetime
```

**PendingQueueItem**
```
id: str (UUID)
session_id: str (FK â†’ sessions)
tool_name: str
raw_output: str                  # Original tool output (10KB-500KB)
files_touched: str (JSON array)
status: QueueStatus
attempts: int = 0
created_at: datetime
```

**EventLogEntry**
```
id: str (UUID)
session_id: Optional[str]
event_type: str
data: str (JSON)
duration_ms: Optional[int]
tokens_in: Optional[int]
tokens_out: Optional[int]
created_at: datetime
```

**Design notes:**

1. **`raw_output` column**: Stores 10KB-500KB per tool use. For a session with 50 tool calls, that's 0.5-25MB. SQLite handles this fine. **Retention policy**: Keep for 30 days, then purge via cleanup job (Phase 9). Keeping enables A/B re-evaluation.

2. **JSON string fields** (`files_touched`, `functions_changed`, `decorators`): Stored as JSON text, deserialized in Python. Our query patterns don't need SQL-level JSON access. SQLite's `json_extract()` available if needed later.

3. **`body_hash` using MD5**: Detecting changes, not securing data. No collision resistance needed. Full 32 hex chars â€” storage cost negligible in a TEXT column.

### 2.4 Migration System (`storage/migrations.py`)

**Approach**: `PRAGMA user_version` â€” SQLite's built-in integer version counter.

```python
MIGRATIONS: list[tuple[int, str]] = [
    (1, """
        BEGIN;
        CREATE TABLE sessions (...);
        CREATE TABLE observations (...);
        CREATE TABLE function_map (...);
        CREATE TABLE call_graph (...);
        CREATE TABLE learnings (...);
        CREATE TABLE pending_queue (...);
        CREATE TABLE event_log (...);
        CREATE INDEX idx_obs_session ON observations(session_id);
        CREATE INDEX idx_fn_file ON function_map(file_path);
        CREATE INDEX idx_fn_session ON function_map(session_id);
        CREATE INDEX idx_fn_change ON function_map(change_type);
        CREATE INDEX idx_cg_caller ON call_graph(caller_file, caller_function);
        CREATE INDEX idx_cg_callee ON call_graph(callee_file, callee_function);
        CREATE INDEX idx_queue_status ON pending_queue(status, created_at);
        CREATE INDEX idx_log_session ON event_log(session_id);
        CREATE INDEX idx_log_type ON event_log(event_type);
        CREATE INDEX idx_log_time ON event_log(created_at);
        COMMIT;
    """),
]

def get_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]

def migrate(conn: sqlite3.Connection) -> None:
    current = get_version(conn)
    for version, sql in MIGRATIONS:
        if version > current:
            conn.executescript(sql)
            conn.execute(f"PRAGMA user_version = {version}")

LATEST_VERSION = MIGRATIONS[-1][0]
```

**Why not Alembic/yoyo-migrations**: ~30 lines total for our needs. Single-user local database, one developer, maybe 5-10 lifetime migrations. Alembic adds SQLAlchemy as a dependency.

**Constraint**: Migrations must be appended sequentially to the `MIGRATIONS` list. No branching â€” version N+1 always follows version N. If this project ever has multiple contributors, revisit with a named-migration system.

**Note**: Each migration wraps in explicit `BEGIN`/`COMMIT` for safety. If a migration fails midway, the transaction rolls back cleanly. (v1 PRD correctly identified this risk but didn't include the fix.)

### 2.5 SQLite Store (`storage/sqlite_store.py`)

**Connection settings:**
```python
conn = sqlite3.connect(
    db_path,
    isolation_level=None,       # Autocommit mode â€” explicit transaction control
    check_same_thread=False,    # Safe with WAL mode
)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA foreign_keys=ON")
conn.execute("PRAGMA busy_timeout=3000")
conn.execute("PRAGMA wal_autocheckpoint=1000")  # Explicit default, prevents WAL bloat
conn.row_factory = sqlite3.Row
```

**WAL maintenance**: Add a `checkpoint()` method to `SQLiteStore` called when a session closes:
```python
def checkpoint(self) -> None:
    """Force WAL checkpoint. Call on session close to keep WAL file small."""
    self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
```

With large `raw_output` values (10KB-500KB per tool use), the WAL file can grow during heavy sessions. Checkpointing on session close keeps it bounded.

**Why these settings:**

| Setting | Value | Rationale |
|---------|-------|-----------|
| `isolation_level=None` | Autocommit | Explicit transaction control via `BEGIN`/`COMMIT`. Avoids Python's implicit transaction behavior which can hold locks unexpectedly. |
| `check_same_thread=False` | Allow cross-thread | WAL mode supports concurrent readers + single writer safely. Hooks and worker may access from different threads/processes. |
| `journal_mode=WAL` | Write-Ahead Log | Concurrent reads don't block writes. Critical for hooks writing while worker reads. |
| `foreign_keys=ON` | Enforce | Observations reference sessions, etc. |
| `busy_timeout=3000` | 3 seconds | If writer is busy, retry for 3s before failing. Prevents immediate `SQLITE_BUSY` errors. |
| `wal_autocheckpoint=1000` | 1000 pages | Explicit default. Prevents WAL bloat during heavy write sessions with large `raw_output` values. |

**Thread safety note**: `sqlite3.Connection` is **not** thread-safe for concurrent writes from the same connection object. Each thread/process must use its own connection. The store should document this or provide a connection-per-thread pattern.

**CRUD operations by table:**

| Table | Operations |
|-------|-----------|
| sessions | `create`, `get`, `update`, `list(status_filter)`, `checkpoint` (WAL truncate on close) |
| observations | `create`, `get`, `list_by_session` |
| function_map | `upsert`, `get_latest_functions(file)`, `get_changed_functions(session)` |
| call_graph | `add_edge`, `get_callers(func)`, `get_callees(func)` |
| learnings | `create`, `get_active(category, min_confidence)`, `update`, `soft_delete` |
| pending_queue | `enqueue`, `dequeue_batch(n)`, `complete`, `retry_failed`, `get_stats` |
| event_log | `log_event` (best-effort, never raises), `query_events(type, session)` |

**Critical: `dequeue_batch` must be atomic:**
```python
def dequeue_batch(self, n: int = 5) -> list[PendingQueueItem]:
    """Atomically claim N pending items for processing.
    
    Uses BEGIN IMMEDIATE to acquire write lock before the SELECT runs,
    plus UPDATE ... RETURNING (SQLite 3.35+) for single-statement atomicity.
    Redundant WHERE status='raw' on outer query is a safety net.
    """
    self.conn.execute("BEGIN IMMEDIATE")
    try:
        rows = self.conn.execute(
            """
            UPDATE pending_queue SET status = 'processing'
            WHERE status = 'raw'
              AND id IN (
                SELECT id FROM pending_queue
                WHERE status = 'raw'
                ORDER BY created_at
                LIMIT ?
              )
            RETURNING *
            """,
            (n,),
        ).fetchall()
        self.conn.execute("COMMIT")
    except Exception:
        self.conn.execute("ROLLBACK")
        raise
    return [PendingQueueItem(**dict(r)) for r in rows]
```

**Why BEGIN IMMEDIATE**: SQLite's single-writer model already prevents true races, but `BEGIN IMMEDIATE` makes intent explicit â€” the write lock is acquired *before* the subquery runs, not upgraded mid-statement. The redundant `WHERE status = 'raw'` on the outer UPDATE is a safety net against future refactoring. Belt and suspenders. SQLite 3.35+ guaranteed since Python 3.14 bundles SQLite 3.46+.

**`best-effort` event_log**: The logger writes in strict order: (1) JSONL file first â€” this is the source of truth and must succeed (propagate exception on failure), (2) SQLite event_log second â€” best-effort projection for programmatic queries, swallowed on failure with stderr warning. If the JSONL write fails, the event is lost â€” but that implies disk failure, at which point SQLite is also compromised.

### 2.6 Logger (`logging/logger.py`)

```python
class MemLogger:
    def __init__(self, log_dir: Path, db_conn: sqlite3.Connection | None = None):
        self.log_dir = log_dir
        self.db_conn = db_conn
        # JSONL file handler with TimedRotatingFileHandler
        # Midnight rotation, 30-day retention

    def log(self, event_type: str, data: dict, **kwargs) -> None:
        """Write to JSONL file first (source of truth), then SQLite event_log (best-effort).
        
        JSONL write failure propagates. SQLite write failure is swallowed with stderr warning.
        """

    @contextmanager
    def timed(self, event_type: str, **kwargs):
        """Context manager that auto-captures duration and status."""
        context = {"status": "started"}
        start = time.monotonic()
        try:
            yield context
            context["status"] = "success"
        except Exception as e:
            context["status"] = "error"
            context["error"] = str(e)
            raise
        finally:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.log(event_type, context, duration_ms=duration_ms, **kwargs)
```

**Design decisions:**
- **JSONL is source of truth**: Easy to grep, tail, debug. SQLite event_log is for programmatic queries (Phase 8 CLI reports).
- **`timed()` context manager**: Reduces boilerplate. Every operation that talks to an API or processes data should be timed.
- **Unique logger per instance**: Prevents handler sharing when multiple Config instances point to different directories (tests).

---

## 3. Database Schema (DDL)

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    project_dir TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    summary TEXT,
    observation_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE observations (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    tool_name TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    detail TEXT,
    files_touched TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(files_touched)),
    functions_changed TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(functions_changed)),
    tokens_raw INTEGER NOT NULL DEFAULT 0,
    tokens_compressed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE function_map (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    file_path TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    signature TEXT NOT NULL,
    docstring TEXT,
    body_hash TEXT NOT NULL,
    decorators TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(decorators)),
    change_type TEXT NOT NULL DEFAULT 'new',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE call_graph (
    id TEXT PRIMARY KEY,
    caller_file TEXT NOT NULL,
    caller_function TEXT NOT NULL,
    callee_file TEXT NOT NULL,
    callee_function TEXT NOT NULL,
    resolution TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    times_confirmed INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'ast',
    session_id TEXT NOT NULL REFERENCES sessions(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE learnings (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    source_session_id TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE pending_queue (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    tool_name TEXT NOT NULL,
    raw_output TEXT NOT NULL,
    files_touched TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(files_touched)),
    status TEXT NOT NULL DEFAULT 'raw',
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE event_log (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    event_type TEXT NOT NULL,
    data TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(data)),
    duration_ms INTEGER,
    tokens_in INTEGER,
    tokens_out INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes
CREATE INDEX idx_obs_session ON observations(session_id);
CREATE INDEX idx_fn_file ON function_map(file_path);
CREATE INDEX idx_fn_session ON function_map(session_id);
CREATE INDEX idx_fn_change ON function_map(change_type);
CREATE INDEX idx_cg_caller ON call_graph(caller_file, caller_function);
CREATE INDEX idx_cg_callee ON call_graph(callee_file, callee_function);
CREATE INDEX idx_queue_status ON pending_queue(status, created_at);
CREATE INDEX idx_log_session ON event_log(session_id);
CREATE INDEX idx_log_type ON event_log(event_type);
CREATE INDEX idx_log_time ON event_log(created_at);
```

---

## 4. Dependencies

### 4.1 Phase 0 runtime dependencies

| Package | Version | Size | Purpose | Required? |
|---------|---------|------|---------|-----------|
| `pydantic` | >=2.0 | ~5MB | Data models, validation | **Yes** |

That's it. **One runtime dependency.** SQLite is stdlib. Logging is stdlib.

### 4.2 Phase 0 dev dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >=8.0 | Testing |
| `ruff` | >=0.8 | Linting + formatting |

### 4.3 Future dependencies (NOT installed in Phase 0)

| Package | Phase | Purpose |
|---------|-------|---------|
| `lancedb` | 4 | Embedded vector + FTS storage (v0.29.1+, Alpha) |
| `anthropic` | 3 | AI compression API calls |
| `fastapi` + `uvicorn` | 3 | Worker HTTP server |
| `aiosqlite` | 3 | Async SQLite for worker (v0.22.1, pure Python) |
| `httpx` | 1 | Hooks â†’ worker HTTP calls |
| `sentence-transformers` | 4 | Embedding model (>=2.7.0) |
| `torch` | 4 | ML runtime |
| `rich` | 8 | CLI terminal output |

---

## 5. `pyproject.toml`

```toml
[project]
name = "claude-mem-lite"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.8",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
target-version = "py314"
line-length = 100
```

Install: `pip install -e ".[dev]"`

---

## 6. Test Plan

### 6.1 Test categories

| Category | Tests | What it validates |
|----------|-------|-------------------|
| **Migrations** | 7 | Fresh DB â†’ latest, idempotent re-run, all tables created, indexes exist, WAL mode, FK enforcement, version increments |
| **Sessions CRUD** | 5 | Create/get, update status, list ordered by started_at DESC, filter by status, observation_count |
| **Observations CRUD** | 2 | Create with metadata, list by session |
| **Function map** | 2 | Upsert with change tracking, get_changed_functions excludes unchanged |
| **Call graph** | 1 | Add edges, query callers/callees |
| **Learnings** | 5 | Create/list, filter by category, confidence threshold, update, soft delete |
| **Pending queue** | 5 | Enqueue/dequeue atomic batch (UPDATE RETURNING), retry failed (respects max_attempts), queue stats, status transitions |
| **Event log** | 2 | Log/query by event_type and session_id, best-effort (never raises) |
| **Concurrency** | 2 | WAL concurrent read/write, threaded writes (10 threads, per-thread connections) |
| **Logger** | 5 | File writes valid JSONL, SQLite writes, timed() captures duration, error status capture, file-only mode (no DB) |
| **Config** | 3 | Default paths correct, custom base_dir, ensure_dirs creates structure |
| **Total** | **39** | |

### 6.2 Test infrastructure

```python
# conftest.py
@pytest.fixture
def tmp_config(tmp_path):
    """Config pointing to a temp directory â€” fresh DB per test."""
    config = Config(base_dir=tmp_path / ".claude-mem")
    config.ensure_dirs()
    return config

@pytest.fixture
def store(tmp_config):
    """SQLiteStore with migrated DB."""
    s = SQLiteStore(tmp_config.db_path)
    return s

@pytest.fixture
def logger(tmp_config, store):
    """MemLogger writing to temp dir."""
    return MemLogger(tmp_config.log_dir, store.conn)
```

### 6.3 Performance targets

| Operation | Target | Rationale |
|-----------|--------|-----------|
| Single INSERT | <1ms | Hook latency budget |
| `dequeue_batch(5)` | <2ms | Worker poll frequency |
| `list_by_session` (100 rows) | <5ms | Context builder read |
| Migration (fresh DB) | <50ms | First-run setup |
| Concurrent 10-thread writes | 0 errors | WAL correctness |

---

## 7. Acceptance Criteria

Phase 0 is complete when:

- [ ] All 39 tests pass (pytest, <10s total runtime)
- [ ] `ruff check` and `ruff format --check` pass with zero warnings
- [ ] SQLite database created with WAL mode, FK enforcement, all 7 tables, 10+ indexes
- [ ] PRAGMA user_version migration system works (v0 â†’ v1, idempotent)
- [ ] MemLogger dual-writes to JSONL + SQLite, `timed()` context manager works
- [ ] Config handles default and custom paths, `ensure_dirs()` creates directory tree
- [ ] `pip install -e ".[dev]"` installs cleanly on Python 3.14
- [ ] No runtime dependencies beyond `pydantic` for this phase
- [ ] `dequeue_batch` uses atomic `UPDATE ... RETURNING` pattern

---

## 8. Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SQLite WAL + cross-process access edge cases | Low | High | One connection per thread/process, busy_timeout=3s |
| Pydantic v2 serialization overhead | Low | Low | We serialize to dict, not JSON. Overhead is negligible. |
| `raw_output` column grows DB size | High (by design) | Low | 30-day retention policy (Phase 9). SQLite handles multi-GB DBs fine. |
| LanceDB API breaks when added in Phase 4 | Medium | Medium | Thin wrapper isolates API surface. Deferred install reduces exposure. |

---

## 9. Resolved Questions

| Question | Decision | Rationale |
|----------|----------|-----------|
| **LanceDB in Phase 0?** | **No â€” Phase 4** | Alpha API, 100MB+ deps, not used until embeddings needed |
| **Python version floor** | **>=3.14** | Latest stable (3.14.3), gets `tomllib` stdlib, improved sqlite3, t-strings |
| **Primary key type** | **TEXT (UUID)** | Consistent across all tables, portable, no auto-increment gaps |
| **Embedding dimension** | **1024 (full)** | No truncation â€” storage/speed cost negligible at <10K vectors |
| **Token counting** | **`len(text) / 4`** | tiktoken is OpenAI-specific. Approximate is fine for budgeting. |
| **Keep raw_output?** | **Yes, 30-day TTL** | Enables A/B re-eval, debugging, reprocessing |
| **Target OS** | **Ubuntu** | macOS SDPA/NaN bug for Qwen3-Embedding not relevant |
