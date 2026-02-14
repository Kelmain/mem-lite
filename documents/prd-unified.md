# claude-mem-lite — Unified PRD

> **Date**: 2026-02-12
> **Status**: Consolidated from 20 source documents (10 phase PRDs + 7 amendments + architecture + implementation plan + key numbers)
> **Rule**: All amendments have been merged inline. Content originating from amendments is marked with **(amended)**.

---

## Table of Contents

1. [Vision & Goals](#1-vision--goals)
2. [Architecture Overview](#2-architecture-overview)
3. [Performance Targets & Constraints](#3-performance-targets--constraints)
4. [Dependency Graph & Implementation Order](#4-dependency-graph--implementation-order)
5. [Project Structure](#5-project-structure)
6. [Phase 0: Storage Layer](#6-phase-0-storage-layer)
7. [Phase 1: Hook Scripts + Direct Capture](#7-phase-1-hook-scripts--direct-capture)
8. [Phase 2: AST Tracker](#8-phase-2-ast-tracker)
9. [Phase 3: Worker Service + Compression](#9-phase-3-worker-service--compression)
10. [Phase 4: Embeddings + Search](#10-phase-4-embeddings--search)
11. [Phase 5: Context Injection](#11-phase-5-context-injection)
12. [Phase 6: Learnings Engine + Call Graph Self-Healing](#12-phase-6-learnings-engine--call-graph-self-healing)
13. [Phase 7: Eval Framework](#13-phase-7-eval-framework)
14. [Phase 8: CLI Reports](#14-phase-8-cli-reports)
15. [Phase 9: Hardening](#15-phase-9-hardening)
16. [Cross-Cutting Concerns](#16-cross-cutting-concerns)
17. [Independent Review Notes](#17-independent-review-notes)
18. [MVP Definition](#18-mvp-definition)

---

## 1. Vision & Goals

A lightweight, Python-native memory system for Claude Code that tracks **what happened**, **what changed**, and **what was learned** — without the complexity overhead of the original claude-mem.

**Core Principle**: Give Claude just enough context to be effective, with the ability to drill deeper on demand. Don't store everything — store what matters.

### What is NOT Being Built

| Excluded | Rationale |
|----------|-----------|
| Web UI | `report` command is enough; original claude-mem's React+SSE viewer was ~30% of codebase for a rarely-used feature |
| Multi-provider agents | Single model: Claude for compression |
| MCP tools | Skills are ~250 tokens vs ~2500 for MCP |
| ChromaDB | LanceDB does hybrid search natively, no Python server needed |
| TypeScript/Bun | Python is simpler, fewer moving parts |
| Full AST diffing (GumTree) | Hash comparison catches changes; Git has full diffs |
| Cross-language AST | Python `ast` for .py only; tree-sitter later if needed |
| Real-time SSE streaming | Not needed without web UI |
| Endless mode | Experimental in claude-mem, adds 60-90s latency |

---

## 2. Architecture Overview

### 2.1 Technology Stack

| Technology | Role | Rationale |
|---|---|---|
| **Python 3.12+** | Primary language | First-class LanceDB SDK, Pydantic schemas, `ast` stdlib for zero-dep Python parsing, `anthropic` SDK, primary language of maintainer |
| **SQLite** (stdlib `sqlite3` in hooks, `aiosqlite` in worker) | Structured data storage | Sessions, queue, metadata, function maps, learnings. WAL mode, 0.1-5ms queries on local SSD |
| **LanceDB** (embedded, Rust core) | Search / vector+FTS | Hybrid vector+FTS via Tantivy, RRF reranking. Replaces claude-mem's triple-store (SQLite+FTS5+ChromaDB) |
| **Qwen3-Embedding-0.6B** | Local embeddings | 600M params, ~1.2GB FP16, #2 sub-1B on MTEB-Code (avg 75.0), 8K token context, Matryoshka dims |
| **FastAPI + uvicorn** | Worker HTTP server | UDS binding at `~/.claude-mem/worker.sock`. Single-process daemon |
| **httpx** | Async HTTP client | Native UDS support for hooks connecting to worker |
| **Anthropic SDK** (raw `client.messages.create()`) | AI compression | NOT Agent SDK — Agent SDK caused memory leaks (40GB+), zombie processes, stale sessions |
| **rich** | Terminal output | Reports rendered in terminal |
| **typer** | CLI framework | Wraps Click, includes rich |
| **PRAGMA user_version** | Schema migrations | SQLite-native, ~30 lines of code |
| **Python `ast` stdlib** | Code intelligence | ~80-95% accuracy for definitions and calls |

### 2.2 System Layers

**Hook Layer** (synchronous scripts, direct SQLite writes):
- `context.py` (SessionStart) — creates session record, injects context
- `capture.py` (PostToolUse) — captures raw tool output to `pending_queue` (<5ms)
- `summary.py` (Stop) — triggers AI session summary
- `cleanup.py` (SessionEnd) — marks session closed

**Worker Service** (FastAPI/uvicorn daemon):
- UDS at `~/.claude-mem/worker.sock`, PID at `~/.claude-mem/worker.pid`
- 30-minute idle timeout, auto-start from alias
- Contains: Compressor (Claude API raw), AST Tracker (ast stdlib), Learnings Engine (pattern detect, confidence weighting)

**Storage Layer** (dual-store):
- SQLite/aiosqlite (structured): sessions, observations, function_map, call_graph, learnings, pending_queue, event_log
- LanceDB (search): observations_vec, summaries_vec, learnings_vec

**Logging/Eval Layer**: structured JSON logs to `~/.claude-mem/logs/`, perf metrics, token counts, search quality

### 2.3 Data Flow

1. Hook fires (PostToolUse) → `capture.py` writes raw data directly to SQLite `pending_queue` (<5ms, no HTTP)
2. Worker polls `pending_queue`, compresses via Claude API, stores structured observation in SQLite + embeds in LanceDB
3. On SessionStart, `context.py` calls worker `GET /api/context` for rich context injection via HTTP over UDS (~100-300ms)
4. On Stop, `summary.py` calls worker `POST /api/summarize` to generate session summary

### 2.4 IPC Model

- **Hooks → SQLite**: Direct stdlib `sqlite3` (synchronous, <5ms)
- **Hooks → Worker**: HTTP over Unix Domain Socket (~1-2ms latency, eliminates port conflicts). Only 2-3 HTTP calls per session
- **Worker internal**: `aiosqlite` for async DB operations

### 2.5 Key Design Decisions

1. **Dual-store split**: SQLite for relations/transactions/exact lookups, LanceDB for fuzzy/semantic search
2. **No Agent SDK**: Raw `anthropic.AsyncAnthropic().messages.create()` only
3. **Progressive disclosure context injection**: 5 layers in priority order, each with token budget
4. **Crash-safe queue**: `pending_queue` in SQLite ensures no data loss on worker crash/API downtime
5. **Graceful degradation**: Every component fails safely — reduced context, not crashes
6. **Self-healing call graph**: Static analysis + runtime observation confirmation, accuracy improves 82% → 94%
7. **Skills over MCP**: ~30 tokens cost vs ~2000+ for MCP tool registration
8. **Worker lifecycle**: Daemon with PID file, UDS socket, 30min idle timeout

---

## 3. Performance Targets & Constraints

| Metric | Target |
|---|---|
| Context injection budget | 2,000 tokens (default) |
| Raw tool output size | 10KB-500KB per tool use |
| Compressed observation | ~500 tokens (~2KB) |
| Compression ratio | 10:1 to 100:1 (38:1 avg observed) |
| Hook latency (capture) | <200ms end-to-end **(amended)**, <30ms no-op |
| SQLite query latency | 0.1-5ms on local SSD with WAL mode |
| Search latency (LanceDB hybrid) | <50ms for typical queries |
| /api/search end-to-end | <300ms |
| Embedding latency | ~80-150ms per embedding on CPU (Qwen3-0.6B) |
| Embedding model load (cached) | <5s |
| Context injection (worker path) | <500ms |
| Context injection (SQLite fallback) | <100ms |
| Worker IPC latency (UDS) | ~1-2ms |
| AI API compression latency | ~100-300ms |
| Worker idle timeout | 30 minutes |
| AST parse latency | <10ms per file |
| AST call resolution accuracy | ~82% initial, ~94% with self-healing |
| Learning dedup threshold | cosine similarity > 0.90 **(amended)** |
| Queue retry | 3 attempts, exponential backoff (5, 10, 20s) |

### Token Budget Allocation

| Layer | Cap (tokens) |
|---|---|
| Session index | 400 |
| Function map | 500 |
| Learnings | 300 |
| Observations | 600 |
| Reserve | 200 |
| **Total** | **2,000** |

### Cost Estimates

| Operation | Cost |
|---|---|
| Compression per session (~50 tool calls) | ~$0.50-0.60 |
| Learnings extraction per session | ~$0.0035 |
| QAG eval per observation | ~$0.003 |
| Full A/B benchmark (30 samples) | ~$1.47 |

---

## 4. Dependency Graph & Implementation Order

```
Phase 0: Storage Layer (foundation — no deps)
 │
 ├──► Phase 1: Hooks + Capture (needs: Phase 0)
 │     │
 │     └──► Phase 3: Worker + Compression (needs: Phase 0, 1)
 │           │
 │           ├──► Phase 4: Embeddings + Search (needs: Phase 3)
 │           │     │
 │           │     ├──► Phase 5: Context Injection (needs: Phase 3, 4)
 │           │     │
 │           │     └──► Phase 6: Learnings Engine (needs: Phase 3, 4)
 │           │
 │           └──► Phase 7: Eval Framework (needs: Phase 3, 4, 5, 6)
 │
 ├──► Phase 2: AST Tracker (needs: Phase 0 only — parallel with Phase 1)
 │
 └──► Phase 8: CLI Reports (needs: Phase 0, 2, 4, 6, 7)

Phase 9: Hardening (needs: all of the above)
```

**Critical path**: `0 → 1 → 3 → 4 → 5` (minimum viable memory system)

**Parallel opportunity**: Phase 2 can be developed alongside Phase 1.

### Effort Estimates

| Phase | Description | Effort | Risk |
|---|---|---|---|
| 0 | Storage Layer | ~1 session | Low |
| 1 | Hook Scripts + Direct Capture | ~1-2 sessions | Medium |
| 2 | AST Tracker | ~2 sessions | Low |
| 3 | Worker Service + Compression | ~2-3 sessions | **HIGH** |
| 4 | Embeddings + Search | ~2-3 sessions | Medium |
| 5 | Context Injection | ~1-2 sessions | Medium |
| 6 | Learnings + Self-Healing | ~2 sessions | Medium |
| 7 | Eval Framework | ~1 session | Low |
| 8 | CLI Reports | ~1 session | Low |
| 9 | Hardening | ~2-3 sessions + ongoing | Medium |
| **Total** | | **~15-19 sessions** | |

---

## 5. Project Structure

```
claude-mem-lite/
├── pyproject.toml
├── src/
│   └── claude_mem_lite/
│       ├── __init__.py
│       ├── config.py
│       ├── hooks/
│       │   ├── __init__.py
│       │   ├── capture.py         # PostToolUse handler
│       │   ├── context.py         # SessionStart handler
│       │   ├── summary.py         # Stop handler
│       │   └── cleanup.py         # SessionEnd handler
│       ├── worker/
│       │   ├── __init__.py
│       │   ├── server.py          # FastAPI app + lifespan + uvicorn
│       │   ├── processor.py       # Queue consumer
│       │   ├── compressor.py      # AI compression
│       │   ├── prompts.py         # Compression prompt templates
│       │   ├── lifecycle.py       # Daemon management
│       │   └── summarizer.py      # Session summarization
│       ├── ast_tracker/
│       │   ├── __init__.py        # Public API: scan_file(), scan_files(), diff_file()
│       │   ├── extractor.py       # FunctionExtractor (ast.NodeVisitor)
│       │   ├── call_graph.py      # CallExtractor + noise filtering
│       │   ├── diff.py            # compare_snapshots()
│       │   └── mermaid.py         # generate_mermaid()
│       ├── learnings/
│       │   ├── __init__.py
│       │   ├── engine.py          # LearningsEngine
│       │   ├── prompts.py         # Extraction prompt template
│       │   └── healer.py          # CallGraphHealer
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── models.py          # Pydantic models + enums
│       │   ├── migrations.py      # PRAGMA user_version migrations
│       │   └── sqlite_store.py    # SQLiteStore CRUD
│       ├── search/
│       │   ├── __init__.py
│       │   ├── embedder.py        # Qwen3-Embedding-0.6B wrapper
│       │   ├── hybrid.py          # Hybrid search orchestration
│       │   └── lance_store.py     # LanceDB table management
│       ├── context/
│       │   ├── __init__.py
│       │   ├── builder.py         # Progressive disclosure engine
│       │   └── estimator.py       # Token estimation
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── evaluator.py       # Compression quality scoring
│       │   ├── benchmark.py       # A/B model comparison
│       │   ├── queries.sql        # Reference SQL for monitoring
│       │   ├── models.py          # Eval Pydantic models
│       │   └── prompts.py         # QAG prompt
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py            # Root CLI group
│       │   ├── install.py         # install-hooks / uninstall-hooks
│       │   ├── report.py          # claude-mem report
│       │   ├── search_cmd.py      # claude-mem search
│       │   ├── mermaid_cmd.py     # claude-mem mermaid
│       │   ├── eval_cmd.py        # claude-mem eval
│       │   └── compress_cmd.py    # claude-mem compress --pending
│       └── logging/
│           ├── __init__.py
│           └── logger.py          # JSONL + SQLite structured logging
├── plugin/
│   ├── hooks.json
│   ├── scripts/
│   │   ├── context-hook.py
│   │   ├── capture-hook.py
│   │   ├── summary-hook.py
│   │   └── cleanup-hook.py
│   └── skills/
│       └── mem-search/
│           └── SKILL.md
├── tests/
│   ├── conftest.py
│   ├── test_storage.py
│   ├── test_hooks.py
│   ├── test_install.py
│   ├── test_extractor.py
│   ├── test_call_graph.py
│   ├── test_compressor.py
│   ├── test_search.py
│   ├── test_learnings.py
│   └── test_context_builder.py
└── README.md
```

**Data directory**: `~/.claude-mem/` containing `worker.sock`, `worker.pid`, `worker.info`, `logs/`, `claude-mem.db`, `lance/`

---

## 6. Phase 0: Storage Layer

### Purpose

Every other phase writes to or reads from SQLite. There is no useful work without the schema, config system, and logger. This is pure plumbing — no AI, no network, no external services.

### Dependencies

None (foundation layer). Python >= 3.12.

### Deliverables

- SQLite database with all 7 tables, indexes, and migration system
- Pydantic models for type-safe data passing between components
- Configuration management (paths, token budgets, model settings)
- Structured logging (JSONL file + SQLite event_log)
- **NOT delivered**: LanceDB schemas (Phase 4), embedding model loading, vector search

### Technical Specification

#### Configuration (`config.py`)

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    base_dir: Path = field(default_factory=lambda: Path.home() / ".claude-mem")

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

    # Embedding (Phase 4)
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024  # Full native dimension

    def ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
```

#### Enums (`storage/models.py`)

| Enum | Values | Usage |
|------|--------|-------|
| `SessionStatus` | `active`, `closed`, `error` | Session lifecycle |
| `QueueStatus` | `raw`, `processing`, `done`, `error` | Pending queue items |
| `ChangeType` | `new`, `modified`, `deleted`, `unchanged` | Function map tracking |
| `FunctionKind` | `function`, `method`, `async_function`, `async_method`, `class` | AST extraction |
| `CallResolution` | `direct`, `self_method`, `import`, `unresolved` | Call graph edges |
| `LearningCategory` | `architecture`, `convention`, `gotcha`, `dependency`, `pattern` | Learning classification |

#### Pydantic Models

**Session**: `id` (UUID), `project_dir`, `started_at`, `ended_at`, `status` (SessionStatus), `summary`, `observation_count` (default 0)

**Observation**: `id` (UUID), `session_id` (FK→sessions), `tool_name`, `title`, `summary`, `detail` (optional), `files_touched` (JSON array), `functions_changed` (JSON array), `tokens_raw`, `tokens_compressed`, `created_at`

**FunctionMapEntry**: `id` (UUID), `session_id`, `file_path`, `qualified_name`, `kind` (FunctionKind), `signature`, `docstring` (optional), `body_hash` (MD5 32 hex), `decorators` (JSON array), `change_type` (ChangeType), `updated_at`

**CallGraphEdge**: `id` (UUID), `caller_file`, `caller_function`, `callee_file`, `callee_function`, `resolution` (CallResolution), `confidence` (float, default 1.0), `times_confirmed` (default 0), `source` ("ast"|"observation"), `session_id`, `created_at`

**Learning**: `id` (UUID), `category` (LearningCategory), `content`, `confidence` (float, default 0.5), `source_session_id`, `is_active` (bool, default True), `created_at`, `updated_at`

**PendingQueueItem**: `id` (UUID), `session_id`, `tool_name`, `raw_output` (10KB-500KB), `files_touched` (JSON array), `priority` **(amended)** ("high"|"normal"|"low"), `status` (QueueStatus), `attempts` (default 0), `created_at`

**EventLogEntry**: `id` (UUID), `session_id` (optional), `event_type`, `data` (JSON), `duration_ms` (optional), `tokens_in` (optional), `tokens_out` (optional), `created_at`

#### Complete SQLite DDL (Migration v1)

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

-- (amended) No FK on session_id — parallel hook race: PostToolUse can precede SessionStart
CREATE TABLE function_map (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
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

-- (amended) No FK on session_id — parallel hook race
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
    session_id TEXT NOT NULL,
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

-- (amended) No FK on session_id; added priority column
CREATE TABLE pending_queue (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    raw_output TEXT NOT NULL,
    files_touched TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(files_touched)),
    priority TEXT NOT NULL DEFAULT 'normal',
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
CREATE INDEX idx_queue_status ON pending_queue(status, priority, created_at);  -- (amended) includes priority
CREATE INDEX idx_log_session ON event_log(session_id);
CREATE INDEX idx_log_type ON event_log(event_type);
CREATE INDEX idx_log_time ON event_log(created_at);
```

#### Migration System (`storage/migrations.py`)

- Uses `PRAGMA user_version` — SQLite's built-in integer version counter
- Migrations appended sequentially to `MIGRATIONS` list
- Each migration wrapped in explicit `BEGIN`/`COMMIT`
- `get_version(conn)` → `PRAGMA user_version`
- `migrate(conn)` applies all migrations above current version

#### SQLite Store (`storage/sqlite_store.py`)

**Connection settings:**
```python
conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA foreign_keys=ON")
conn.execute("PRAGMA busy_timeout=3000")
conn.execute("PRAGMA wal_autocheckpoint=1000")
conn.row_factory = sqlite3.Row
```

**CRUD operations per table:**

| Table | Operations |
|-------|-----------|
| sessions | `create`, `get`, `update`, `list(status_filter)`, `checkpoint(mode)` |
| observations | `create`, `get`, `list_by_session` |
| function_map | `upsert`, `get_latest_functions(file)`, `get_changed_functions(session)` |
| call_graph | `add_edge`, `get_callers(func)`, `get_callees(func)` |
| learnings | `create`, `get_active(category, min_confidence)`, `update`, `soft_delete` |
| pending_queue | `enqueue`, `dequeue_batch(n)`, `complete`, `retry_failed`, `get_stats` |
| event_log | `log_event` (best-effort, never raises), `query_events(type, session)` |

**Critical `dequeue_batch` pattern (atomic):**
```python
def dequeue_batch(self, n: int = 5) -> list[PendingQueueItem]:
    self.conn.execute("BEGIN IMMEDIATE")
    try:
        rows = self.conn.execute("""
            UPDATE pending_queue SET status = 'processing'
            WHERE status = 'raw'
              AND id IN (
                SELECT id FROM pending_queue
                WHERE status = 'raw'
                ORDER BY created_at
                LIMIT ?
              )
            RETURNING *
        """, (n,)).fetchall()
        self.conn.execute("COMMIT")
    except Exception:
        self.conn.execute("ROLLBACK")
        raise
    return [PendingQueueItem(**dict(r)) for r in rows]
```

**Checkpoint method** **(amended)**: accepts `mode` parameter (default `"PASSIVE"`)

#### Logger (`logging/logger.py`)

```python
class MemLogger:
    def __init__(self, log_dir: Path, db_conn: sqlite3.Connection | None = None):
        # JSONL file handler with TimedRotatingFileHandler
        # Midnight rotation, 30-day retention

    def log(self, event_type: str, data: dict, **kwargs) -> None:
        # Write JSONL first (source of truth), then SQLite (best-effort)

    @contextmanager
    def timed(self, event_type: str, **kwargs):
        # Context manager that auto-captures duration and status
```

- JSONL write failure propagates (source of truth)
- SQLite write failure swallowed with stderr warning
- `timed()` context manager captures duration_ms, status, error

#### Dependencies

**Runtime**: `pydantic>=2.0` only. **Dev**: `pytest>=8.0`, `ruff>=0.8`

### Acceptance Criteria

39 tests covering: migrations (7), sessions CRUD (5), observations CRUD (2), function map (2), call graph (1), learnings (5), pending queue (5), event log (2), concurrency (2), logger (5), config (3).

Performance: single INSERT <1ms, dequeue_batch(5) <2ms, list_by_session(100) <5ms, migration(fresh) <50ms.

### Estimated Effort

~1 session (4-6 hours)

### Risks & Mitigations

- **Low risk**: Deterministic, no external dependencies, pure plumbing.
- WAL mode concurrent access: tested with 10-thread concurrent writes.

---

## 7. Phase 1: Hook Scripts + Direct Capture

### Purpose

First real integration with Claude Code. Once storage exists, you need data flowing into it. Phase 1 is the **only** way data enters the system — without hooks, there's nothing to compress, embed, search, or inject.

### Dependencies

Phase 0 (Storage Layer). Estimated effort: 1-2 sessions (6-10 hours).

### Deliverables

- 4 hook scripts for Claude Code lifecycle events
- CLI command (`claude-mem install-hooks`) for settings registration
- Direct SQLite writes from hooks — no worker dependency
- Session lifecycle: create on start, capture on tool use, mark on stop, close on end

### Technical Specification

#### v3 Critical Fixes **(amended)**

1. **`python3` path trap**: `install.py` writes `sys.executable` absolute path (not shebang)
2. **Latency KPIs**: Revised to `<200ms` capture, `<30ms` no-op (original <10ms impossible — Pydantic import alone costs more)
3. **`cwd` project identity**: All hooks use `CLAUDE_PROJECT_DIR` env var (stable, doesn't change on `cd`)
4. **`TRUNCATE` checkpoint blocks readers**: Changed to `PASSIVE`
5. **`settings.json` JSONC comments**: `try/except` with diagnostic error

#### Claude Code Hooks API

**Hook Input (stdin JSON):**
```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/transcript.jsonl",
  "cwd": "/current/working/dir",
  "permission_mode": "default",
  "hook_event_name": "PostToolUse"
}
```

**Event-specific additions:**

| Event | Extra Fields |
|-------|-------------|
| `SessionStart` | `source`: `"startup"` \| `"resume"` \| `"clear"` \| `"compact"` |
| `PostToolUse` | `tool_name`, `tool_input`, `tool_response`, `tool_use_id` |
| `Stop` | `stop_hook_active` (bool) |
| `SessionEnd` | `reason`: `"clear"` \| `"logout"` \| `"prompt_input_exit"` \| `"other"` |

**Environment Variables**: `CLAUDE_PROJECT_DIR` (project root, stable), `CLAUDE_SESSION_ID`, `CLAUDE_ENV_FILE` (SessionStart only)

#### Hook Pattern (all 4 follow)

1. Read JSON from stdin
2. Parse with `json.loads` (stdlib) — **before heavy imports**
3. Check early-exit conditions (no-op paths exit in <30ms)
4. Import `claude_mem_lite` modules (pays Pydantic cost only when needed)
5. Resolve project directory from `CLAUDE_PROJECT_DIR` (fallback: `cwd`)
6. SQLite operation (direct, synchronous)
7. JSON to stdout (exit 0) or stderr (exit 2)
8. Exit (<200ms target)

#### Hook 1: `capture.py` — PostToolUse

- **Matcher**: `*` (all tools)
- **Priority tagging**: `high` (Write, Edit, MultiEdit, Bash), `low` (Read, Glob, Grep, TodoRead, TodoWrite), `normal` (everything else)
- Writes to `pending_queue` with raw payload
- Outputs `{"continue": true, "suppressOutput": true}`
- No session FK — session may not exist yet
- `_extract_files()` helper gets file_path from Write/Edit/MultiEdit/Read

#### Hook 2: `context.py` — SessionStart

- **Matcher**: `startup|resume|clear|compact`
- Creates session with `INSERT OR IGNORE` (idempotent for resume/compact)
- Uses `CLAUDE_PROJECT_DIR` for session's `project_dir`
- Returns minimal context (~100-200 tokens): last 5 sessions via `hookSpecificOutput.additionalContext`

#### Hook 3: `summary.py` — Stop

- **No matcher** (fires on every stop)
- Fast exit if `stop_hook_active` is true (<30ms)
- Logs event with pending observation count
- No session status change (Stop fires on every response)

#### Hook 4: `cleanup.py` — SessionEnd

- Updates session status to `closed` with `ended_at`
- Checkpoints WAL with `PASSIVE` mode (non-blocking)
- Logs session_end event with reason

#### Hook Registration (`cli/install.py`)

- Resolves `sys.executable` absolute path at install time
- All hooks have `timeout: 10` seconds
- Merges with existing hooks (doesn't overwrite)
- Idempotent (checks for existing claude_mem_lite hooks)
- JSONC detection with diagnostic error
- Atomic write (temp file + rename)
- CLI entry point: `claude-mem install-hooks`, `claude-mem uninstall-hooks`

#### Phase 0 Schema Changes Required **(amended)**

1. `pending_queue.session_id`: Remove FK constraint
2. Add `priority TEXT NOT NULL DEFAULT 'normal'` column
3. Update index to include priority
4. `checkpoint()` method accepts mode parameter (default `"PASSIVE"`)

#### Dependencies

**Runtime**: None (all stdlib + pydantic from Phase 0). **Dev**: None beyond pytest/ruff.

### Acceptance Criteria

29 tests covering: capture (6), context (5), summary (4), cleanup (4), install (7), integration (3).

Performance: capture/context/cleanup <200ms, summary no-op <30ms, 10 concurrent captures 0 errors.

### Estimated Effort

1-2 sessions (6-10 hours)

### Risks & Mitigations

- **Medium risk**: Claude Code hook mechanics, latency sensitivity
- Mitigation: early-exit pattern, lazy imports, direct SQLite (no HTTP)

---

## 8. Phase 2: AST Tracker

### Purpose

Deterministic, testable code intelligence for Python files. The function map is Layer 2 of context injection (~500 tokens). It gives Claude a lightweight file overview without reading full source. This is novel — the original claude-mem has no AST tracking.

### Dependencies

Phase 0 (Storage Layer) — function_map + call_graph tables. **Parallel with Phase 1** (no dependency). Estimated effort: 2 sessions (8-12 hours).

### Deliverables

- Python AST extraction: functions, methods, classes, signatures, decorators, body hashes
- Call graph construction with confidence-typed edges
- Change detection: new/modified/deleted/unchanged
- Mermaid diagram generation
- Zero hook/worker coupling — pure library module

### Technical Specification

#### Data Types (frozen dataclasses with slots)

```python
@dataclass(frozen=True, slots=True)
class FunctionInfo:
    qualified_name: str          # "AuthService.authenticate"
    kind: str                    # function|method|async_function|async_method|class
    parent_class: str | None
    signature: str               # "authenticate(email: str, password: str) -> Token"
    decorators: list[str]        # ["@router.post('/login')", "@require_auth"]
    docstring: str | None        # First line only
    line_start: int
    line_end: int
    body_hash: str               # MD5 of ast.dump(node), full 32 hex chars
    calls: list[CallInfo] = field(default_factory=list)

@dataclass(frozen=True, slots=True)
class CallInfo:
    raw_name: str                # "self.validate()", "db.query()"
    resolved_name: str | None    # "AuthService.validate" or None
    resolution: str              # direct|self_method|import|unresolved
    line_number: int

@dataclass(frozen=True, slots=True)
class FileSnapshot:
    file_path: str
    functions: list[FunctionInfo]
    import_map: dict[str, str]   # local_name -> qualified_name
    parse_error: str | None = None

@dataclass(frozen=True, slots=True)
class FunctionDiff:
    qualified_name: str
    change_type: str             # new|modified|deleted|unchanged
    current: FunctionInfo | None  # None if deleted
    previous_hash: str | None    # None if new
```

#### FunctionExtractor (`extractor.py`)

`ast.NodeVisitor` subclass. Extracts functions/methods/classes with:
- Signatures including types, defaults, *args/**kwargs (strips self/cls)
- Decorators as string representations
- First-line docstrings
- `body_hash`: `hashlib.md5(ast.dump(node).encode()).hexdigest()` — stable across whitespace/comments, changes on logic

**Import map handling** **(amended)**: Relative imports use `__rel{N}__` prefix placeholder:
```python
if node.level > 0:  # Relative import
    prefix = f"__rel{node.level}__"
    module_part = f"{prefix}.{node.module}" if node.module else prefix
```

#### CallExtractor (`call_graph.py`)

**Resolution types**: direct, self_method, import, unresolved

**Noise filtering** — three frozensets:
- `NOISE_CALLABLES` (~50): builtins (print, len, range, isinstance, super, sorted, min, max, etc.) + stdlib logging
- `NOISE_OBJECTS` (~15): logger, log, os, sys, re, json, math, pathlib, Path, etc.
- `NOISE_ATTRIBUTE_CALLS` (~30): dict/list/set/string methods, dunder methods

Unresolved edges persisted at `confidence=0.5`, EXCEPT when method name is in `NOISE_ATTRIBUTE_CALLS`.

**Resolution accuracy:**

| Pattern | Accuracy |
|---------|----------|
| Function/class definitions | ~100% |
| Direct calls: `foo()` | ~95% |
| `self.method()` | ~90% |
| Imported calls: `module.func()` | ~80% |
| Variable calls: `x = foo; x()` | ~0% |
| Dynamic: `getattr()` | ~0% |

#### Diff (`diff.py`)

`compare_snapshots(current, previous)`: Build dicts by qualified_name, compare body_hash. No rename detection (git handles that). **(amended)** Duplicate function names: last definition wins (matches Python semantics).

#### Mermaid (`mermaid.py`)

`generate_mermaid(functions, file_path, show_all, change_types)`:
- Default scoping: changed functions + direct call targets only
- Style: new=#dfd (green), modified=#ffd (yellow), deleted=#fdd (red), unresolved=dashed

**(amended)** Syntax sanitization for type hints:
```python
def _sanitize_for_mermaid(label: str) -> str:
    return (
        label.replace("[", "⟨").replace("]", "⟩")
        .replace("{", "(").replace("}", ")")
        .replace('"', "'").replace("|", "∣")
        .replace("<", "‹").replace(">", "›")
    )
```

#### Public API (`__init__.py`)

```python
def scan_file(file_path: str, source: str | None = None) -> FileSnapshot
def scan_files(file_paths: list[str]) -> list[FileSnapshot]  # Skips non-.py
def diff_file(current: FileSnapshot, previous: list[FunctionInfo]) -> list[FunctionDiff]
```

#### Integration Plan (post-Phase 1+2)

Hook `capture.py` adds AST scanning for Write/Edit/MultiEdit on .py files:
- DELETE + INSERT per file (not UPSERT)
- Latency budget per file: 5-18ms

#### Edge Cases

SyntaxError → FileSnapshot with parse_error. 500-function limit per file. Encoding: `utf-8` with `errors="replace"`. Nested functions all extracted. `__init__.py` re-exports captured in import map.

#### Dependencies

**Runtime**: None (all stdlib: ast, hashlib, dataclasses). **Dev**: None.

### Acceptance Criteria

56 tests **(amended, +5)** covering: extractor basics (6), signatures (6), classes (4), nesting (3), decorators (2), docstrings (2), body_hash (4), call resolution (7), import map (5), noise filter (3), diff (7), mermaid (5), integration (2).

### Estimated Effort

2 sessions (8-12 hours)

### Risks & Mitigations

- **Low risk**: Deterministic, testable, no external deps.
- Python 3.14 ast changes: `ast.compare()` available, removed legacy constant nodes.

---

## 9. Phase 3: Worker Service + Compression

### Purpose

Background worker that turns raw tool outputs into structured, searchable observations. **Highest-risk phase** — compression quality determines whether the entire system is useful.

### Dependencies

Phase 0 (Storage) + Phase 1 (data in queue). Estimated effort: 2-3 sessions (10-16 hours).

### Deliverables

- FastAPI worker over Unix Domain Socket
- Queue processor (poll → compress → store)
- AI compressor via Anthropic SDK with structured outputs **(amended)**
- Daemon lifecycle with subprocess.Popen **(amended)**
- Session summarization with auto-trigger **(amended)**

### Technical Specification

#### Module Structure

```
src/claude_mem_lite/worker/
├── __init__.py
├── server.py          # FastAPI app + lifespan + uvicorn runner
├── processor.py       # Queue consumer
├── compressor.py      # AI compression
├── prompts.py         # Compression prompt templates
├── lifecycle.py       # Daemon management
└── summarizer.py      # Session summarization
```

#### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | >=0.128 | HTTP endpoints |
| uvicorn[standard] | >=0.30 | ASGI server with UDS support |
| anthropic | >=0.78 | Claude API for compression |
| aiosqlite | >=0.22 | Async SQLite for worker event loop |
| httpx | >=0.27 | Hooks → worker HTTP calls over UDS |

#### FastAPI Endpoints

| Method | Path | Purpose | Response |
|--------|------|---------|----------|
| GET | /api/health | Liveness + stats | `{"status": "ok", "uptime_s": int, "queue_depth": int, "observations_today": int}` |
| GET | /api/context | Context injection (Phase 5) | Placeholder → replaced in Phase 5 |
| POST | /api/summarize | Session summary | `{"summary": str, "tokens": int}` |
| GET | /api/queue/stats | Queue debugging | `{"raw": int, "processing": int, "done": int, "error": int}` |

No `/api/compress` endpoint — compression triggered by background processor polling.

#### Worker Server (`server.py`)

- FastAPI lifespan context manager, single aiosqlite connection
- UDS binding: `uvicorn.run("claude_mem_lite.worker.server:app", uds=str(config.socket_path))`
- IdleTracker: 30-minute timeout, checks every 60s, `time.monotonic()` + `asyncio.Event()`
- Every HTTP request and queue item calls `idle_tracker.touch()`

#### Queue Processor (`processor.py`)

**Constants:**
- `POLL_INTERVAL = 2.0` seconds
- `BATCH_SIZE = 5` items per dequeue
- `MAX_ATTEMPTS = 3` retry limit
- `BACKOFF_BASE = 5.0` seconds (exponential: 5, 10, 20)
- `SUMMARY_IDLE_MINUTES = 2` **(amended)**

**Main loop:**
```python
async def run(self):
    await self.recover_orphaned_items()
    while not self.idle_tracker.should_shutdown:
        items = await self.dequeue_batch()
        if items:
            for item in items:
                self.idle_tracker.touch()
                await self.process_item(item)
        else:
            await self._check_pending_summaries()  # (amended)
            await asyncio.sleep(self.POLL_INTERVAL)
```

**(amended)** Orphan recovery: Reset ALL processing→raw unconditionally (no time threshold). Safe because start() ensures old worker is dead.

**(amended)** Auto-summarization trigger in `_check_pending_summaries()`:
- Session has `hook.stop` event logged
- No remaining `raw` or `processing` items in pending_queue
- Has at least one observation
- `sessions.summary IS NULL` and status != 'closed'
- Last event >2 min ago (debounce)
- LIMIT 3 per poll cycle

#### AI Compressor (`compressor.py`)

- `AsyncAnthropic()` (reads `ANTHROPIC_API_KEY` from env)
- Model: `claude-haiku-4-5-20251001` (pinned snapshot)
- Pricing: $1/1M input, $5/1M output
- max_tokens=1024, no streaming, no system prompt

**(amended)** Uses structured outputs API:
```python
response = await self.client.messages.create(
    model=self.model,
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}],
    output_config={"format": {"type": "json_schema", "schema": COMPRESSION_SCHEMA}},
)
```

**COMPRESSION_SCHEMA:**
```python
{
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Brief action summary, 5-10 words, imperative mood"},
        "summary": {"type": "string", "description": "What happened and why, 1-3 sentences"},
        "detail": {"anyOf": [{"type": "string"}, {"type": "null"}],
                   "description": "Technical details worth remembering. Null if summary is sufficient."},
        "files_touched": {"type": "array", "items": {"type": "string"},
                         "description": "List of file paths actually modified"},
        "functions_changed": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "name": {"type": "string"},
                    "action": {"type": "string", "enum": ["new", "modified", "deleted"]},
                },
                "required": ["file", "name", "action"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "summary", "files_touched", "functions_changed"],
    "additionalProperties": False,
}
```

**Error classification** **(amended)**:

| Error | Classification |
|-------|---------------|
| APIConnectionError | Retryable |
| RateLimitError | Retryable |
| APIStatusError 5xx | Retryable |
| APIStatusError 4xx | Non-retryable |
| json.JSONDecodeError | Retryable **(amended)** |

Fallback parser: `_extract_json_fallback()` strips markdown fences, finds first `{` to last `}`.

Truncation: `MAX_RAW_CHARS = 32,000`. Head+tail (half from start, half from end).

#### Compression Prompt (`prompts.py`)

```
Compress this Claude Code tool output into a structured observation.

Tool: {tool_name}
Files: {files_touched}

<tool_output>
{raw_output}
</tool_output>

Return a JSON object with these fields:
{{
  "title": "Brief action summary, 5-10 words. Example: 'Added JWT auth middleware'",
  "summary": "What happened and why, 1-3 sentences. Include key decisions made.",
  "detail": "Technical details worth remembering across sessions. Null if summary is sufficient.",
  "files_touched": ["list", "of", "file/paths"],
  "functions_changed": [
    {{"file": "path/to/file.py", "name": "function_name", "action": "new|modified|deleted"}}
  ]
}}

Rules:
- title: imperative mood, no period. Focus on WHAT changed, not tool mechanics.
- summary: preserve WHY decisions were made, not just WHAT happened.
- detail: only include if there's non-obvious information. Omit for trivial operations.
- files_touched: only files actually modified, not just read.
- functions_changed: only if identifiable from the output. Empty array if unclear.
- If the tool output is a Read operation with no changes, set title to describe what was
  examined and summary to key findings.
```

#### Worker Lifecycle (`lifecycle.py`)

**(amended — CRITICAL)** Replaces `os.fork()` with `subprocess.Popen`. `os.fork()` is unsafe with threaded libraries (aiosqlite, httpx, uvloop) and deprecated in Python 3.14.

```python
def start(self, daemon: bool = True) -> int:
    existing_pid = self.get_pid()
    if existing_pid is not None:
        if self._is_pid_alive(existing_pid):
            return existing_pid
        self._cleanup_stale_files()

    self.config.ensure_dirs()

    if daemon:
        proc = subprocess.Popen(
            [sys.executable, "-m", "claude_mem_lite.worker.server"],
            cwd=str(self.config.base_dir),
            start_new_session=True,   # setsid equivalent
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._write_pid(proc.pid)
        self._wait_for_socket(timeout=10)
        return proc.pid
    else:
        self._write_pid(os.getpid())
        run_worker(self.config)
        self._cleanup_stale_files()
        return os.getpid()
```

Server.py writes its own PID on startup: `config.pid_path.write_text(str(os.getpid()))`

CLI entry point: `claude-mem-worker = "claude_mem_lite.worker.lifecycle:cli_main"`. Commands: start, stop, status, restart **(amended)**.

Stop: SIGTERM, wait max 5s, clean up PID/socket files.

#### Session Summarizer (`summarizer.py`)

**SUMMARY_SCHEMA** **(amended — structured outputs):**
```python
{
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_files": {"type": "array", "items": {"type": "string"}},
        "key_decisions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "key_files", "key_decisions"],
    "additionalProperties": False,
}
```

Uses max_tokens=512. Stores summary in sessions table, sets status='closed'.

#### Pydantic Models

```python
class CompressedObservation(BaseModel):
    title: str
    summary: str
    detail: str | None = None
    files_touched: list[str] = []
    functions_changed: list[FunctionChange] = []
    tokens_in: int = 0
    tokens_out: int = 0

class FunctionChange(BaseModel):
    file: str
    name: str
    action: str  # "new" | "modified" | "deleted"

class SessionSummary(BaseModel):
    summary: str
    key_files: list[str] = []
    key_decisions: list[str] = []

class RetryableError(Exception): pass
class NonRetryableError(Exception): pass
```

### Acceptance Criteria

34 tests **(amended, +6)**. Key criteria: daemon start/stop, PID/socket files, 30-min idle timeout, atomic queue claiming, structured outputs compression, retry logic, orphan recovery (unconditional), auto-summarization trigger.

### Estimated Effort

2-3 sessions (10-16 hours)

### Risks & Mitigations

- **HIGH risk**: Compression quality determines system value. Bad compression = useless observations.
- Mitigation: Budget extra time for prompt iteration. Phase 7 eval framework measures quality.
- Structured outputs **(amended)** eliminates JSON parsing errors.

---

## 10. Phase 4: Embeddings + Search

### Purpose

Makes compressed observations retrievable. Local embedding model, LanceDB vector+FTS indexes, hybrid search API.

### Dependencies

Phase 3 (Worker + Compression). Estimated effort: 2-3 sessions (10-16 hours).

### Deliverables

- Qwen3-Embedding-0.6B integration via sentence-transformers
- LanceDB tables with vector + FTS indexes
- Hybrid search (vector + BM25 via RRF reranking)
- FTS-only fallback
- Search API endpoints
- Claude Code SKILL.md for on-demand search

### Technical Specification

#### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| lancedb | >=0.29,<0.31 | Embedded vector DB |
| sentence-transformers | >=5.0 | Local embedding model loading |
| torch | >=2.2 | PyTorch backend |
| transformers | >=4.51.0 | Required by Qwen3 |
| tantivy | >=0.22 | FTS engine (auto-installed with lancedb) |

Total install: ~2-3GB.

#### Embedder (`search/embedder.py`)

Model: **Qwen3-Embedding-0.6B** (released June 5, 2025). Context: 8K tokens. Dimension: **1024** (full native). Device: CPU. Inference: ~80-150ms per embedding.

```python
SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu", truncate_dim=config.embedding_dim)
```

Instruction-aware prefixes:
```python
INSTRUCTIONS = {
    "observation": "Instruct: Find development observations about code changes and decisions\nQuery: ",
    "code": "Instruct: Find code snippets, functions, and implementation details\nQuery: ",
    "learning": "Instruct: Find project learnings, patterns, and best practices\nQuery: ",
    "document": "",  # No instruction for indexing
}
```

`encode()` with `normalize_embeddings=True, show_progress_bar=False`. Failure is non-fatal — falls back to FTS-only.

**(amended)** Model loading wrapped in `asyncio.to_thread()` — CPU-bound ~3-5s operation.

#### LanceDB Storage (`search/lance_store.py`)

Uses pyarrow schemas (NOT LanceModel Pydantic classes). Sync API wrapped in `asyncio.to_thread()`.

**Tables (1024d vectors):**
1. `observations_vec`: observation_id, session_id, title, summary, files_touched, functions_changed, created_at, vector(1024)
2. `summaries_vec`: session_id, summary_text, project_path, created_at, vector(1024)
3. `learnings_vec`: learning_id, content, category, confidence, vector(1024)

FTS index: Tantivy with `en_stem` tokenizer on `["title", "summary"]`.

#### Search API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | /api/search | Hybrid search: vector + BM25 + RRF reranking |
| GET | /api/callgraph | Call graph for file (reads Phase 2 data) |
| GET | /api/observation/{id} | Full observation detail |

`/api/search` params: `q` (text), `limit` (1-20, default 5), `type` (observation\|code\|learning). Falls back to FTS-only when embeddings unavailable.

#### Processor Integration

After compression, embed and write to LanceDB (non-fatal on failure):
```python
if self.lance_store and self.embedder.available:
    await asyncio.to_thread(
        self.lance_store.add_observation,
        obs_id=obs_id, session_id=..., title=..., summary=..., ...
    )
```

#### Embedding Backfill **(amended)**

Runs in processor's `run()` method after HTTP server is bound:
```python
async def run(self):
    if self.lance_store and self.embedder and self.embedder.available:
        await self.backfill_embeddings()
    # Normal processing loop...
```

Processes observations with `embedding_status = 'pending'`, LIMIT 100 per startup (8-15s).

Schema migration: `ALTER TABLE observations ADD COLUMN embedding_status TEXT DEFAULT 'pending';`

#### SKILL.md **(amended — curl URL encoding fix)**

```bash
# Search observations
curl -s -G --unix-socket ~/.claude-mem/worker.sock http://localhost/api/search \
    --data-urlencode "q=QUERY" -d "limit=5"

# Get observation detail
curl -s --unix-socket ~/.claude-mem/worker.sock http://localhost/api/observation/OBS_ID

# Get call graph
curl -s -G --unix-socket ~/.claude-mem/worker.sock http://localhost/api/callgraph \
    --data-urlencode "file=PATH"
```

~150 tokens when loaded by Claude Code.

#### Configuration

```python
embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
embedding_dim: int = 1024
embedding_device: str = "cpu"
lance_db_path: str = ""  # ~/.claude-mem/lance/
search_limit_default: int = 5
search_limit_max: int = 20
```

### Acceptance Criteria

24 tests. Key: model loads <5s, observations embedded after compression, hybrid search works, FTS fallback, backfill mechanism, search latency <300ms, worker startup <8s.

### Estimated Effort

2-3 sessions (10-16 hours)

### Risks & Mitigations

- **Medium risk**: Qwen3 model load time (3-5s), search quality tuning.
- Mitigation: asyncio.to_thread for model load, FTS fallback, Phase 9 ONNX INT8 option.

---

## 11. Phase 5: Context Injection

### Purpose

The payoff. When Claude Code starts a new session, it receives a pre-built context block summarizing recent sessions, code changes, project learnings, and recent observations. Eliminates cold-start problem.

### Dependencies

Phase 3 (Worker) + Phase 4 (Search). Estimated effort: 1-2 sessions (5-8 hours).

### Deliverables

- Progressive disclosure context builder with token budgeting
- 4 layers with priority and caps
- Updated context hook with dual-path (worker / SQLite fallback)

### Technical Specification

#### Token Estimation (`context/estimator.py`)

```python
def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5))
```

Accuracy: +/-10-15% vs Claude's actual tokenizer.

#### Context Builder (`context/builder.py`)

**Layer 1: Session Index** (priority 1, cap 400 tokens)
- Last 10 sessions with summaries, ordered by recency
- Format: `## Recent Sessions` + `- [{relative_time}] {summary_truncated_200chars}`

**Layer 2: Function Map** (priority 2, cap 500 tokens)
- Functions with change_type in (new, modified, deleted) from last 5 sessions
- Grouped by file, deduplicated, LIMIT 30

**Layer 3: Learnings** (priority 3, cap 300 tokens)
- Active learnings with confidence >= 0.5, sorted by confidence DESC, LIMIT 10

**Layer 4: Relevant Observations** (priority 4, cap 600 tokens) **(amended — CRITICAL)**
- **Recency-based retrieval ONLY** — no semantic search
- Original used previous session's summary as search query → context pollution when sessions work on unrelated topics
- Most recent observations from last 5 sessions, LIMIT 10
- Header: `## Recent Observations`
- Semantic search deferred to UserPromptSubmit (Phase 9)

**Assembly:**
1. Build all 4 layers concurrently via `asyncio.gather(return_exceptions=True)`
2. Sort by priority, greedy assembly within budget (minus reserve)
3. Append search hint: `Use curl --unix-socket ~/.claude-mem/worker.sock http://localhost/api/search?q=... for deeper context.`

```python
@dataclass
class ContextResult:
    text: str
    total_tokens: int
    budget: int
    layers_included: list[str]
    layers_skipped: list[str]
    build_time_ms: float
```

#### Worker Endpoint: GET /api/context

Replaces Phase 3 placeholder. Params: `project_path`, `session_id`.
Returns: `{"context": str, "tokens": int, "layers": list, "build_ms": float}`

#### Context Hook (`hooks/context.py`)

**Dual-path design:**
1. Worker available (worker.sock exists) → GET /api/context → rich context (all layers)
2. Worker unavailable → basic context from SQLite directly (Layer 1 only, last 5 sessions)

Uses `http.client` + custom `_UDSHTTPConnection` (NOT httpx — ~200ms import). 5-second worker timeout.

**(amended)** AF_UNIX guard: `if not hasattr(socket, 'AF_UNIX'): return ""`
**(amended)** SQLite fallback timeout: 3s (was 1s)
**(amended)** UTC enforcement in `_relative_time`

Output: `{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "..."}}`

#### Configuration

```python
@dataclass
class ContextConfig:
    budget: int = 2000
    session_index_cap: int = 400
    function_map_cap: int = 500
    learnings_cap: int = 300
    observations_cap: int = 600
    reserve: int = 200
    worker_timeout_s: float = 5.0
    fallback_db_timeout_s: float = 3.0  # (amended)
    max_sessions: int = 10
    max_functions: int = 30
    max_learnings: int = 10
    max_observations: int = 10
    min_learning_confidence: float = 0.5
```

#### Graceful Degradation

Every failure → reduced context or no context. Never a crash.
- Worker not running → basic SQLite context (Layer 1 only)
- Worker slow (>5s) → fallback to basic
- SQLite doesn't exist → no context
- Layer exception → skip that layer, others still included
- Budget exceeded → skip lower-priority layers

### Acceptance Criteria

27 tests. Key: /api/context returns all layers within budget, hook outputs valid JSON, fallback works, empty DB handled, layer failures isolated, token estimation +/-20%.

### Estimated Effort

1-2 sessions (5-8 hours)

### Risks & Mitigations

- **Medium risk**: Token budgeting accuracy, context usefulness
- Mitigation: progressive disclosure, graceful degradation, conservative budget

---

## 12. Phase 6: Learnings Engine + Call Graph Self-Healing

### Purpose

The system gets smarter over time. Extracts project-specific knowledge from session summaries: architecture decisions, coding conventions, known gotchas, dependency notes, recurring patterns.

### Dependencies

Phase 3 (summarizer, compressor) + Phase 4 (embeddings for dedup). Estimated effort: 2 sessions (10-14 hours).

### Deliverables

- Learning extraction from session summaries via Claude API
- Dedup via semantic similarity (cosine > 0.90)
- Confidence evolution (diminishing returns)
- Contradiction detection and resolution **(amended v2)**
- Call graph self-healing from observations
- Schema migration for learnings table
- CLI: `claude-mem learnings`

### Technical Specification

#### Schema Migration

```sql
ALTER TABLE learnings ADD COLUMN times_seen INTEGER NOT NULL DEFAULT 1;
ALTER TABLE learnings ADD COLUMN source_sessions TEXT NOT NULL DEFAULT '[]';
ALTER TABLE learnings ADD COLUMN is_manual INTEGER NOT NULL DEFAULT 0;
UPDATE learnings SET source_sessions = json_array(source_session_id)
    WHERE source_session_id IS NOT NULL AND source_sessions = '[]';
ALTER TABLE learnings DROP COLUMN source_session_id;  -- try/except for safety
CREATE INDEX IF NOT EXISTS idx_learnings_active
    ON learnings(is_active, confidence DESC) WHERE is_active = 1;
```

#### LearningsEngine (`learnings/engine.py`)

**Constants:**
- `INITIAL_CONFIDENCE = 0.5`
- `MAX_AUTO_CONFIDENCE = 0.95` (only manual = 1.0)
- `BOOST_FACTOR = 0.2`
- `CONTRADICTION_PENALTY = 0.3`
- `MIN_CONFIDENCE = 0.1`

**Extraction flow:** `extract_from_session(session_id, summary, observations, project_path)` →
1. Fetch existing learnings for context (`_get_active_learnings`)
2. Call Haiku to extract candidates
3. Process each: insert / merge / contradict / skip

**Dedup** (`_find_duplicate`):
- LanceDB: semantic search, cosine >= 0.90. **(amended)** Why 0.90 not 0.85: reduced Matryoshka dimensionality inflates cosine for short text.
- Fallback: LIKE substring match

**(amended)** `_get_active_learnings` returns ALL active learnings including low-confidence. Prevents "ghost learning" re-extraction loop. Low-confidence prefixed with `[low-confidence]`.

**Confidence evolution:**
- Boost: `min(0.95, current + 0.2 * (0.95 - current))` — diminishing returns: 0.50 → 0.59 → 0.67 → 0.73 → 0.79...
- Penalty: `max(0.1, current - 0.3)`

**Candidate outcomes:**
- `inserted`: New learning, confidence=0.5, times_seen=1
- `merged`: Duplicate found, confidence boosted, times_seen incremented
- `contradicted` **(amended v2 — CRITICAL)**: Penalize OLD + INSERT NEW as fresh entry + embed in LanceDB. v1 bug: penalized old but never inserted new — system forgot the topic.
- `skipped`: Extraction confidence < 0.3

#### CallGraphHealer (`learnings/healer.py`)

**Constants:**
- `CONFIRMATION_BOOST = 0.15`
- `NEW_EDGE_CONFIDENCE = 0.6`
- `STALE_DECAY = 0.05`
- `STALE_SESSION_THRESHOLD = 10`

`confirm_edges_from_observation(observation, session_id)`:
- Extract function references from `functions_changed`
- Existing edge → bump confidence + times_confirmed
- Missing edge but both in function_map → insert at confidence=0.6, source='observation'

`decay_stale_edges(project_path)`:
- Only affects observation-source edges (AST edges re-confirmed on scan)
- Stale = not confirmed in 10 sessions → decay by 0.05/session
- Edge at 0.6 takes 10 closes to reach 0.1

#### Worker Integration

```
[Phase 3] dequeue → compress → store observation
[Phase 6] → confirm call graph edges from observation

[On summarization]
[Phase 3] aggregate observations → generate summary
[Phase 6] → extract learnings from summary + observations
[Phase 6] → decay stale call graph edges
```

Both learnings and call graph healing are NON-FATAL.

#### CLI: `claude-mem learnings`

Actions: list, add, edit, remove, reset. Manual additions: `is_manual=True`, `confidence=1.0` (immune to decay/contradiction).

Cost: ~$0.0035/session, ~$0.90/month.

### Acceptance Criteria

33 tests. Key: extraction, dedup, confidence evolution, contradiction (penalize + insert), call graph healing, stale decay.

### Estimated Effort

2 sessions (10-14 hours)

### Risks & Mitigations

- **Medium risk**: Dedup accuracy, confidence tuning
- Mitigation: 0.90 threshold, diminishing returns formula, ghost learning prevention

---

## 13. Phase 7: Eval Framework

### Purpose

Offline evaluation measuring compression quality, search effectiveness, and system health. Answers: "Is the compressed observation good enough to preserve information a future session needs?"

### Dependencies

Phase 3, 4, 5, 6. Estimated effort: ~1 session.

### Deliverables

- Compression quality scoring (deterministic + LLM-judge)
- A/B model comparison via offline replay (NOT online routing) **(amended)**
- SQL analysis queries for monitoring
- CLI: `claude-mem eval`

### Technical Specification

#### Design Corrections **(amended)**

- **Online A/B routing REJECTED** → offline replay. At ~50 obs/day, online needs weeks for significance.
- **Semantic similarity DROPPED** → QAG-based info_preservation. Embedding cosine between raw (500KB) and compressed (500 tokens) is meaningless.
- **Eval is offline analysis**, NOT production routing. No production code changes.
- **Sonnet 4.5 as judge** (stronger-model-as-judge).

#### Scoring (CompressionScore)

| Dimension | Type | Range | Method |
|-----------|------|-------|--------|
| structural_validity | Deterministic | 0.0 or 1.0 | Valid JSON, required fields |
| compression_ratio | Deterministic | float | len(raw) / tokens_compressed |
| title_quality | Deterministic | 0.0-1.0 | Length 3-15 words, no period, no weak start |
| info_preservation | LLM-judge (QAG) | 0.0-1.0 | Questions from raw, answer from compressed |
| decision_rationale | LLM-judge (binary) | 0.0 or 1.0 | Does summary explain WHY? |
| latency_ms | Measured | int | API call duration |
| cost_usd | Calculated | float | Token-based pricing |

**Composite formula:**
```
quality = 0.15 * structural_validity + 0.10 * title_quality
        + 0.50 * info_preservation + 0.25 * decision_rationale
cost_adjusted_quality = quality / cost_usd
```

#### QAG-Based Info Preservation **(amended)**

- Generate **UP TO 5** factual questions from raw (minimum 2) — not "exactly 5"
- Explicit inclusion: architectural decisions, API changes, dependencies, trade-offs, error handling
- Explicit exclusion: line numbers, formatting, renaming, internals, boilerplate
- Dynamic denominator: `answerable / len(questions)`
- Questions < 2 = eval failure (returns 0.0). Questions > 5 = truncated to 5.
- Judge model: `claude-sonnet-4-5-20250929` ($3/$15 per MTok)
- Raw truncated to 4K chars

#### A/B Benchmark (Offline Replay)

`BenchmarkRunner.run(model_a, model_b, sample_size=30, judge_model)`:
1. Sample raw outputs from pending_queue (`ORDER BY RANDOM()`)
2. Compress each through both models
3. Score each (deterministic + QAG)
4. Produce paired comparison report

**(amended)** Performance note: `ORDER BY RANDOM()` is a full table scan. Run `claude-mem prune --keep-raw 100` first.

#### MODEL_RATES **(amended — moved to Config)**

```python
MODEL_RATES = {
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
}
```

#### CLI

- `claude-mem eval compression [--limit] [--with-qag] [--json] [--since]`
- `claude-mem eval benchmark [--model-a] [--model-b] [--samples] [--judge] [--json]`
- `claude-mem eval health [--days] [--json]`

### Acceptance Criteria

22 tests. Key: deterministic scoring, QAG prompt/scoring, benchmark runner, CLI, SQL queries.

### Estimated Effort

~1 session

### Risks & Mitigations

- **Low risk**: Offline analysis, no production impact.

---

## 14. Phase 8: CLI Reports

### Purpose

Human-facing presentation layer. CLI commands that surface data from all previous phases.

### Dependencies

Phase 0, 2, 4, 6, 7. Estimated effort: ~1 session.

### Deliverables

- `claude-mem report` — session summary, function changes, learnings
- `claude-mem search <query>` — hybrid search with FTS5 fallback
- `claude-mem mermaid` — call graph export
- `claude-mem status` **(amended)** — system diagnostics

### Technical Specification

#### Dependencies **(amended — standardized on Typer)**

| Package | Version | Purpose |
|---------|---------|---------|
| typer | >=0.12 | CLI framework (wraps Click, includes rich) |
| rich | >=14.0 | Terminal formatting (transitive dep of typer) |

#### Command Tree

```
claude-mem
    report [--days N] [--session ID] [--md] [--json] [--learnings N]
    search <query> [--limit N] [--type observation|code|learning] [--json]
    mermaid [file] [--session ID] [--all] [--output F]
    status
    eval compression | benchmark | health  (Phase 7)
```

Entry point: `claude-mem = "claude_mem_lite.cli.main:app"` (Typer app)

#### Search Dual-Path **(amended)**

1. Check worker via `_discover_worker()` (socket/PID + health endpoint)
2. Worker available → HTTP GET /api/search (hybrid)
3. Worker unavailable OR error → SQLite FTS5 fallback **(amended — fallback on HTTP errors too)**
4. Badge: `[hybrid]` or `[fts]`

#### FTS5 Virtual Table **(amended — BLOCKER)**

Phase 4 rejected SQLite FTS5 in favor of LanceDB Tantivy. But CLI search needs FTS5 fallback when worker is down. Migration:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts
    USING fts5(title, summary, detail, content=observations, content_rowid=rowid);

INSERT INTO observations_fts(rowid, title, summary, detail)
    SELECT rowid, title, summary, COALESCE(detail, '') FROM observations;

-- INSERT trigger
CREATE TRIGGER observations_ai AFTER INSERT ON observations BEGIN
    INSERT INTO observations_fts(rowid, title, summary, detail)
    VALUES (new.rowid, new.title, new.summary, COALESCE(new.detail, ''));
END;

-- UPDATE trigger (delete + insert)
CREATE TRIGGER observations_au AFTER UPDATE ON observations BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, title, summary, detail)
    VALUES ('delete', old.rowid, old.title, old.summary, COALESCE(old.detail, ''));
    INSERT INTO observations_fts(rowid, title, summary, detail)
    VALUES (new.rowid, new.title, new.summary, COALESCE(new.detail, ''));
END;

-- DELETE trigger
CREATE TRIGGER observations_ad AFTER DELETE ON observations BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, title, summary, detail)
    VALUES ('delete', old.rowid, old.title, old.summary, COALESCE(old.detail, ''));
END;
```

#### `claude-mem status` **(amended)**

Checks: database stats (obs/session/learning counts, DB size, last session age), worker status (running/not), pending queue (unprocessed/error counts), hooks installed, learnings active count.

### Acceptance Criteria

29 tests **(amended)**. Key: report rendering (terminal + markdown), search worker + FTS fallback, mermaid output, CLI integration, status command, edge cases.

### Estimated Effort

~1 session

### Risks & Mitigations

- **Low risk**: Read-only presentation layer.

---

## 15. Phase 9: Hardening

### Purpose

Production-readiness backlog from all previous phases. Ongoing, not a single deliverable.

### Dependencies

All phases. Estimated effort: ~2-3 sessions initially, then continuous.

### Deliverables

#### Tier 1: Must-Have for Daily Use (~1 session)

| ID | Item | Description |
|----|------|-------------|
| X-1 | pyproject.toml finalization | Build backend: hatchling. Base deps + `[worker]`, `[dev]`, `[onnx]` extras |
| X-2 | README.md | Quick Start, How It Works, Commands, Configuration, Cost, Architecture |
| D8-1 | `claude-mem status` | DB stats, worker status, pending queue, hooks, learnings |
| D8-2 | `claude-mem compress --pending` | Inline compression without worker **(amended)** |
| D8-4 | `claude-mem prune` | Clear old raw_output, delete old event_log, optional VACUUM |
| D0-1 | raw_output retention + purge | Storage: 50 obs/session × 50KB = 2.5MB/session |

#### pyproject.toml Final Structure

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "claude-mem-lite"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.40.0",
    "typer>=0.12",
    "rich>=14.0",
    "pydantic>=2.0",
    "httpx>=0.27",
]

[project.optional-dependencies]
worker = [
    "aiosqlite>=0.20",
    "uvicorn>=0.32",
    "starlette>=0.41",
    "sentence-transformers>=4.0",
    "lancedb>=0.17",
]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24", "ruff>=0.8", "coverage>=7.0"]
onnx = ["onnxruntime>=1.20", "optimum>=1.23"]

[project.scripts]
claude-mem = "claude_mem_lite.cli.main:app"
claude-mem-worker = "claude_mem_lite.worker.lifecycle:main"
```

#### `compress --pending` **(amended — dependency isolation)**

- Does NOT import from worker.processor, search.lance_store, or search.embedder
- Uses sync `anthropic.Anthropic()` (not AsyncAnthropic)
- Imports Compressor for prompt template only
- Sets `embedding_status='pending'` — worker's backfill indexes later
- Importable without `[worker]` extras

#### `prune` **(amended)**

- Clear raw_output from old pending_queue (SET NULL, preserve recent N for eval)
- Delete old event_log entries
- Optional VACUUM with **(amended)** try/except for SQLITE_BUSY

#### Tier 2: Quality-of-Life (~1 session)

| ID | Item |
|----|------|
| D3-1 | Tiered compression (per-operation model selection) |
| D4-2 | Re-embed missing observations |
| D7-1 | Automated eval triggers (every 20 compressions) |
| D7-3 | Raw output lifecycle management |
| D2-2 | Cross-file import resolution |

#### Tier 3: Opportunistic (trigger-based)

D4-1 (ONNX INT8, if latency >150ms), D4-3 (IVF_PQ, if >1000 obs), D2-1 (Non-Python, if JS/TS heavy), D6-2 (Cross-project learnings), D5-1 (PreCompact hook), X-3/X-4 (DB recovery), D6-1 (Prune learnings), D6-3 (LLM graph healing), and others.

#### Complete CLI Command Tree (after Phase 9)

```
claude-mem
    status, report, search, mermaid
    eval compression | benchmark | health
    compress --pending
    prune
    learnings list | add | edit | remove | reset
    install-hooks, uninstall-hooks
    doctor, reembed
```

### Acceptance Criteria

Tier 1: 16 tests. Tier 2: 10 tests. pip install succeeds, --version works, all commands functional.

### Estimated Effort

~2-3 sessions initially, then ongoing

### Risks & Mitigations

- **Medium risk**: Edge cases, cross-platform issues
- **(amended)** Windows support deferred entirely — targets macOS/Linux only

---

## 16. Cross-Cutting Concerns

### Error Handling

- **Hooks**: Never crash. Exit 0 with reduced/no context on any error.
- **Worker**: Retryable vs non-retryable errors. 3 attempts with exponential backoff.
- **Compression**: Structured outputs **(amended)** eliminate JSON parsing errors. Fallback parser retained.
- **Embedding**: Non-fatal. Falls back to FTS-only search.
- **All layers**: Graceful degradation — reduced functionality, never total failure.

### Logging Strategy

- **JSONL files**: Source of truth, midnight rotation, 30-day retention
- **SQLite event_log**: Best-effort mirror for SQL queries
- **Event types**: hook.context_inject, hook.capture, hook.summary, compress.start/done/error, ast.scan, search.query/result, learning.extracted/reinforced/invalidated, context.token_budget, eval.*
- `MemLogger.timed()` context manager for automatic duration/status capture

### Security Considerations

- `ANTHROPIC_API_KEY` from environment (never hardcoded)
- No user-facing HTTP server (UDS only, local access)
- SQLite parameterized queries throughout
- No shell=True in subprocess calls
- File paths validated where applicable

### Testing Strategy

| Phase | Tests | Cumulative |
|-------|-------|-----------|
| 0 | 39 | 39 |
| 1 | 29 | 68 |
| 2 | 56 | 124 |
| 3 | 34 | 158 |
| 4 | 24 | 182 |
| 5 | 27 | 209 |
| 6 | 33 | 242 |
| 7 | 22 | 264 |
| 8 | 29 | 293 |
| 9 | 26 | **319** |

All tests run with `pytest`. Async tests via `pytest-asyncio`. Coverage tracked with `coverage`.

---

## 17. Independent Review Notes

*(From the PRD Ordering Guide — independent review observations)*

### 1. The "lite" framing vs. actual complexity

The project is called "claude-mem-**lite**" but specifies 10 phases, 50,000+ words of PRD documentation, and 15-19 implementation sessions. The dependency stack includes FastAPI, uvicorn, LanceDB, sentence-transformers, PyTorch (transitive), the Anthropic SDK, aiosqlite, and a 1.2GB embedding model. For a single-developer personal tool, this is substantial.

Not necessarily wrong — the architectural choices are individually defensible. But calling it "lite" sets expectations that the implementation doesn't match. Scope creep in personal tools often means they never ship.

### 2. Python 3.14 minimum is aggressive but defensible

Requiring Python ≥3.14 is bleeding-edge. Most CI systems and developer machines still default to 3.12 or 3.13. However, since this is a personal tool and 3.14 is now in stable bugfix releases (3.14.3), the choice is defensible.

### 3. Over-specification before prototyping

Comprehensive PRDs with specific line numbers and code snippets — yet no working code exists. The amendments demonstrate the problem: FK constraint race conditions, non-existent FTS5 tables, daemonization incompatible with Python 3.14, prompt assumptions that don't match the SDK API.

**Suggestion**: Implement Phases 0-1 as a spike before finalizing remaining PRDs.

### 4. LanceDB risk is lower than stated

The PRDs describe LanceDB as "still Alpha (0.29.1)." As of Feb 2026, the Lance SDK core (Rust) reached 1.0.0, the file format 2.1 is stable, and multiple organizations deploy LanceDB in production.

### 5. Qwen3-Embedding model size vs. alternatives

The 0.6B model is ~1.2GB FP16 and takes 3-5s to load. ModernBERT (gte-modernbert-base) loads in <1s and runs at 15-25ms per embedding vs 80-150ms for Qwen3. For a personal memory tool with hundreds to low thousands of observations, the quality gap (71.1 vs 75.0 on MTEB-Code) is unlikely to produce noticeably different search results. The latency gap is very noticeable every session.

If Phase 4 latency testing shows Qwen3 load time is painful, switching to ModernBERT is a low-risk fallback.

### 6. Amendment documents are essential — not optional reading

The amendments contain blocker-severity fixes. Implementing Phase 3 without the `os.fork()` → `subprocess.Popen` fix will produce deadlocks. Implementing Phase 8 without the FTS5 migration will produce dead code. **All amendments have been merged inline in this unified PRD.**

---

## 18. MVP Definition

**Minimum Viable Product**: Phases 0, 1, 3, 4, 5

This is the minimum path to a working memory system:

| Phase | What it gives you |
|-------|-------------------|
| 0 | SQLite schema + config + logging |
| 1 | Data flowing in via hooks |
| 3 | Compressed, structured observations |
| 4 | Searchable observations |
| 5 | Claude gets context at SessionStart |

**Critical path**: `0 → 1 → 3 → 4 → 5`

**Total MVP effort**: ~8-13 sessions

**After MVP**:
- Phase 2 (AST tracker) — slots in whenever, independent
- Phase 6 (Learnings) — system gets smarter over time
- Phase 7 (Eval) — measure what you built
- Phase 8 (CLI) — human-facing reports
- Phase 9 (Hardening) — production polish

Everything beyond MVP enhances quality, observability, or robustness. The MVP alone delivers the core value proposition: Claude remembers what happened in previous sessions.
