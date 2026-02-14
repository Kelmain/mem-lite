# PRD: Phase 5 â€” Context Injection (v1)

**Project**: claude-mem-lite (fork of claude-mem, Python rewrite)
**Phase**: 5 of 9
**Status**: Ready for implementation
**Dependencies**: Phase 3 (Worker + Compression â€” `/api/context` placeholder), Phase 4 (Embeddings + Search â€” hybrid search API)
**Estimated effort**: 1â€“2 sessions (~5â€“8 hours)
**Python**: 3.14

---

## 1. Purpose & Context

### 1.1 What this phase delivers

When Claude Code starts a new session, it receives a pre-built context block that summarizes what happened in recent sessions, what code changed, and what the project's key learnings are. This context is assembled from data captured in Phases 1â€“4 and injected via the SessionStart hook.

Specifically:

- **`context/builder.py`** â€” Progressive disclosure engine that assembles context within a token budget
- **`context/estimator.py`** â€” Token estimation for budget management (no external dependencies)
- **`GET /api/context` endpoint** â€” Replaces the Phase 3 placeholder with a real implementation
- **Updated `context-hook.py`** â€” Calls the worker's `/api/context` endpoint over UDS; falls back to SQLite when the worker is unavailable
- **Configurable token budget** â€” Default 2000 tokens, user-adjustable

### 1.2 What this phase does NOT deliver

- **UserPromptSubmit context injection** â€” Per-prompt context augmentation is a future enhancement. This phase only injects at SessionStart.
- **PreCompact context preservation** â€” Ensuring injected context survives compaction (possible future Phase 9 enhancement).
- **Learnings engine** â€” Phase 6 builds the learnings extraction system. Phase 5 injects any learnings that already exist in the database (manually added or from future phases).
- **Call graph on-demand injection** â€” Architecture spec lists Layer 5 (call graph context) as "on demand only." This is handled by the Phase 4 SKILL.md search, not by context injection.
- **Real-time token counting** â€” We do not call Anthropic's Token Count API. We use local estimation.

### 1.3 Why this matters

Without context injection, every Claude Code session starts from zero. The developer must re-explain project structure, recent work, conventions, and gotchas. Context injection eliminates this cold-start problem by giving Claude a structured summary of project state.

The architecture spec frames the core principle: "Give Claude just enough context to be effective, with the ability to drill deeper on demand." Phase 5 delivers the "just enough" part. The SKILL.md from Phase 4 delivers the "drill deeper" part.

### 1.4 Lessons from claude-mem (original)

claude-mem's context injection has documented problems we explicitly avoid:

| claude-mem Problem | claude-mem-lite Solution |
|---|---|
| Race condition: context-hook fires before worker is ready (issue #775) | Dual-path: worker available â†’ rich context; worker unavailable â†’ basic context from SQLite directly |
| Stdout pollution from npm/install scripts breaks JSON parsing (v4.3.1 fix) | Pure Python script, no subprocess shelling, no package manager invocations |
| Table-formatted context consumes tokens on formatting | Minimal markdown â€” optimized for Claude comprehension, not human readability |
| Fixed context format regardless of session state | Progressive disclosure â€” layers included based on data availability and budget |

---

## 2. Technical Specification

### 2.1 Token Estimation (`context/estimator.py`)

**Problem**: Anthropic does not provide a local tokenizer for Claude 3+ models. The official Token Count API exists but requires HTTP calls (latency + rate limits). tiktoken with `p50k_base` shows ~35% error for Claude models.

**Decision**: Character-based heuristic. Research (Peta Muir, Nov 2025) shows bytes/characters correlate linearly with Claude token counts. For our use case â€” a 2000-token budget where Â±10% accuracy is acceptable â€” a simple heuristic outperforms complex approaches.

```python
def estimate_tokens(text: str) -> int:
    """Estimate Claude token count from text.

    Uses character-based heuristic: ~3.5 characters per token.
    Accuracy: Â±10-15% vs Claude's actual tokenizer.
    Sufficient for context budget management where slight
    overestimation is preferable to exceeding the budget.
    """
    return max(1, int(len(text) / 3.5))
```

**Why not other approaches**:

| Approach | Rejected Because |
|---|---|
| Anthropic Token Count API | Adds 50â€“200ms latency per call. Context building would need multiple calls. Budget management is not billing-grade. |
| tiktoken `p50k_base` | 35% error for Claude. Adds a dependency (~5MB). Worse accuracy than char/3.5 for Claude specifically. |
| `transformers` tokenizer | Requires downloading Claude's tokenizer (not publicly available). Qwen3's tokenizer is different. |
| Word-based (`words * 1.3`) | Less accurate for code-heavy content where tokens â‰  words. |

**Validation**: Phase 7 eval framework can compare estimated vs actual token counts from API responses. If heuristic proves consistently off, swap in a calibrated linear model.

### 2.2 Context Builder (`context/builder.py`)

The builder assembles context in priority order, stopping when the token budget is exhausted.

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class ContextLayer:
    """A single layer of context with its content and metadata."""
    name: str
    content: str
    tokens: int
    priority: int  # lower = higher priority


@dataclass
class ContextResult:
    """Final assembled context with metadata for logging."""
    text: str
    total_tokens: int
    budget: int
    layers_included: list[str]
    layers_skipped: list[str]
    build_time_ms: float


class ContextBuilder:
    """Progressive disclosure context builder.

    Assembles context within a token budget from multiple data sources.
    Layers are added in priority order until the budget is exhausted.
    """

    DEFAULT_BUDGET = 2000

    # Layer budget caps â€” prevent any single layer from dominating
    LAYER_CAPS = {
        "session_index": 400,
        "function_map": 500,
        "learnings": 300,
        "observations": 600,
        "reserve": 200,  # headroom for formatting overhead
    }

    def __init__(self, db, lance_store=None, budget: int = DEFAULT_BUDGET):
        """
        Args:
            db: aiosqlite connection (worker) or sqlite3 connection (hook fallback)
            lance_store: LanceStore instance for semantic search (optional, None in fallback mode)
            budget: Maximum token budget for injected context
        """
        self.db = db
        self.lance_store = lance_store
        self.budget = budget

    async def build(self, project_path: str, session_id: str) -> ContextResult:
        """Build context for a new session.

        Layers in priority order:
        1. Session index â€” what happened recently (always included)
        2. Function map â€” what code changed recently
        3. Learnings â€” project knowledge
        4. Relevant observations â€” semantic search for project-relevant context
        """
        ...
```

#### Layer 1: Session Index (priority 1, cap 400 tokens)

Recent session summaries, most recent first.

```python
async def _build_session_index(self, project_path: str) -> Optional[ContextLayer]:
    """Recent session summaries.

    Query: Last 10 sessions with summaries, ordered by recency.
    Format: Each session as a one-liner with relative timestamp.
    """
    rows = await self.db.execute_fetchall(
        """
        SELECT id, started_at, summary_text
        FROM sessions
        WHERE project_path = ? AND summary_text IS NOT NULL
        ORDER BY started_at DESC
        LIMIT 10
        """,
        (project_path,),
    )
    if not rows:
        return None

    lines = ["## Recent Sessions"]
    for row in rows:
        age = _relative_time(row["started_at"])
        summary = _truncate(row["summary_text"], max_chars=200)
        lines.append(f"- [{age}] {summary}")

    content = "\n".join(lines)
    tokens = estimate_tokens(content)

    # Trim sessions from the bottom if over cap
    while tokens > self.LAYER_CAPS["session_index"] and len(lines) > 2:
        lines.pop()
        content = "\n".join(lines)
        tokens = estimate_tokens(content)

    return ContextLayer(name="session_index", content=content, tokens=tokens, priority=1)
```

**Output example** (for Claude's consumption):
```
## Recent Sessions
- [2h ago] Implemented JWT auth service, added refresh token endpoint, wrote tests
- [yesterday] Set up FastAPI project structure, configured SQLAlchemy + Alembic
- [2 days ago] Initial project scaffold, Docker setup
```

#### Layer 2: Function Map (priority 2, cap 500 tokens)

Recently changed functions across files, grouped by file.

```python
async def _build_function_map(self, project_path: str) -> Optional[ContextLayer]:
    """Recently changed code â€” functions modified in recent sessions.

    Query: Functions with change_type in ('new', 'modified', 'deleted')
    from the last 5 sessions, grouped by file.
    """
    rows = await self.db.execute_fetchall(
        """
        SELECT fm.file_path, fm.qualified_name, fm.signature,
               fm.change_type, fm.line_start, fm.line_end
        FROM function_map fm
        JOIN sessions s ON fm.session_id = s.id
        WHERE s.project_path = ?
          AND fm.change_type IN ('new', 'modified', 'deleted')
        ORDER BY s.started_at DESC, fm.file_path
        LIMIT 30
        """,
        (project_path,),
    )
    if not rows:
        return None

    # Group by file, deduplicate (keep most recent per qualified_name)
    seen = set()
    by_file: dict[str, list] = {}
    for row in rows:
        key = (row["file_path"], row["qualified_name"])
        if key in seen:
            continue
        seen.add(key)
        by_file.setdefault(row["file_path"], []).append(row)

    lines = ["## Recently Changed Code"]
    for file_path, funcs in by_file.items():
        rel_path = _make_relative(file_path, project_path)
        lines.append(f"{rel_path}:")
        for f in funcs:
            tag = f["change_type"].upper()
            lines.append(f"  {f['signature']}  [{tag}]")

    content = "\n".join(lines)
    tokens = estimate_tokens(content)

    # Trim files from the bottom if over cap
    while tokens > self.LAYER_CAPS["function_map"] and len(by_file) > 1:
        last_file = list(by_file.keys())[-1]
        # Remove lines for last file
        lines = [l for l in lines if not l.startswith(f"{_make_relative(last_file, project_path)}")
                 and not l.startswith("  ") or lines.index(l) < len(lines) - len(by_file[last_file]) - 1]
        del by_file[last_file]
        content = "\n".join(lines)
        tokens = estimate_tokens(content)

    return ContextLayer(name="function_map", content=content, tokens=tokens, priority=2)
```

**Output example**:
```
## Recently Changed Code
auth/service.py:
  authenticate(email, password) -> Token  [MODIFIED]
  refresh_token(token) -> Token  [NEW]
api/routes.py:
  login_endpoint(request) -> Response  [MODIFIED]
```

#### Layer 3: Learnings (priority 3, cap 300 tokens)

Active project learnings sorted by confidence.

```python
async def _build_learnings(self) -> Optional[ContextLayer]:
    """Active project learnings, highest confidence first.

    Only includes learnings with confidence >= 0.5.
    """
    rows = await self.db.execute_fetchall(
        """
        SELECT category, content, confidence
        FROM learnings
        WHERE is_active = TRUE AND confidence >= 0.5
        ORDER BY confidence DESC, times_seen DESC
        LIMIT 10
        """,
    )
    if not rows:
        return None

    lines = ["## Project Knowledge"]
    for row in rows:
        cat = row["category"].capitalize()
        lines.append(f"- {cat}: {row['content']}")

    content = "\n".join(lines)
    tokens = estimate_tokens(content)

    while tokens > self.LAYER_CAPS["learnings"] and len(lines) > 2:
        lines.pop()
        content = "\n".join(lines)
        tokens = estimate_tokens(content)

    return ContextLayer(name="learnings", content=content, tokens=tokens, priority=3)
```

**Output example**:
```
## Project Knowledge
- Architecture: FastAPI + SQLAlchemy ORM, Alembic migrations, pytest fixtures
- Convention: JWT tokens in HttpOnly cookies, refresh flow in auth/refresh.py
- Gotcha: user_service.get_by_email() returns None on DB errors â€” always check
```

#### Layer 4: Relevant Observations (priority 4, cap 600 tokens)

Semantic search for observations relevant to the project's recent work. Uses hybrid search from Phase 4 when available.

```python
async def _build_observations(self, project_path: str) -> Optional[ContextLayer]:
    """Relevant recent observations via hybrid search.

    Strategy:
    - If lance_store available: semantic search using recent session summary as query
    - Fallback: most recent observations from SQLite
    """
    if self.lance_store:
        # Use the most recent session summary as the search query
        recent = await self.db.execute_fetchone(
            """
            SELECT summary_text FROM sessions
            WHERE project_path = ? AND summary_text IS NOT NULL
            ORDER BY started_at DESC LIMIT 1
            """,
            (project_path,),
        )
        if recent and recent["summary_text"]:
            query = recent["summary_text"][:200]  # Truncate for search
            results = await asyncio.to_thread(
                self.lance_store.search_observations,
                query=query,
                limit=10,
            )
            if results:
                return self._format_observations(results)

    # Fallback: most recent observations from SQLite
    rows = await self.db.execute_fetchall(
        """
        SELECT title, summary, created_at
        FROM observations
        WHERE session_id IN (
            SELECT id FROM sessions
            WHERE project_path = ?
            ORDER BY started_at DESC LIMIT 5
        )
        ORDER BY created_at DESC
        LIMIT 10
        """,
        (project_path,),
    )
    if not rows:
        return None

    return self._format_observations_from_rows(rows)

def _format_observations(self, results) -> ContextLayer:
    """Format LanceDB search results as context layer."""
    lines = ["## Relevant Past Work"]
    for r in results:
        lines.append(f"- {r.title}: {r.summary}")

    content = "\n".join(lines)
    tokens = estimate_tokens(content)
    while tokens > self.LAYER_CAPS["observations"] and len(lines) > 2:
        lines.pop()
        content = "\n".join(lines)
        tokens = estimate_tokens(content)

    return ContextLayer(name="observations", content=content, tokens=tokens, priority=4)
```

#### Assembly Logic

```python
async def build(self, project_path: str, session_id: str) -> ContextResult:
    """Build context within token budget."""
    start = time.monotonic()

    # Build all layers concurrently
    layers = await asyncio.gather(
        self._build_session_index(project_path),
        self._build_function_map(project_path),
        self._build_learnings(),
        self._build_observations(project_path),
        return_exceptions=True,
    )

    # Filter out None and exceptions, sort by priority
    valid_layers: list[ContextLayer] = []
    for layer in layers:
        if isinstance(layer, ContextLayer):
            valid_layers.append(layer)
        elif isinstance(layer, Exception):
            logger.warning(f"Layer build failed: {layer}")

    valid_layers.sort(key=lambda l: l.priority)

    # Greedy assembly within budget
    included: list[ContextLayer] = []
    skipped: list[str] = []
    remaining = self.budget - self.LAYER_CAPS["reserve"]

    for layer in valid_layers:
        if layer.tokens <= remaining:
            included.append(layer)
            remaining -= layer.tokens
        else:
            skipped.append(layer.name)

    # Assemble final text
    parts = [layer.content for layer in included]

    # Add search hint if SKILL.md is available
    if parts:
        parts.append(
            "\n---\n"
            "Use `curl --unix-socket ~/.claude-mem/worker.sock "
            "http://localhost/api/search?q=...` for deeper context."
        )

    text = "\n\n".join(parts)
    total_tokens = estimate_tokens(text)
    elapsed_ms = (time.monotonic() - start) * 1000

    return ContextResult(
        text=text,
        total_tokens=total_tokens,
        budget=self.budget,
        layers_included=[l.name for l in included],
        layers_skipped=skipped,
        build_time_ms=elapsed_ms,
    )
```

### 2.3 Worker Endpoint: `GET /api/context`

Replaces the Phase 3 placeholder. Added to `worker/server.py`.

```python
@app.get("/api/context")
async def get_context(project_path: str, session_id: str = ""):
    """Build and return context for SessionStart injection.

    Args:
        project_path: Project root (from hook's cwd)
        session_id: Current session ID (for logging, not filtering)

    Returns:
        {"context": str, "tokens": int, "layers": list, "build_ms": float}
    """
    builder = ContextBuilder(
        db=app.state.db,
        lance_store=app.state.lance_store,  # May be None if embeddings unavailable
        budget=app.state.config.context_budget,
    )
    result = await builder.build(project_path, session_id)

    # Log context injection event
    await app.state.logger.log_event(
        session_id=session_id,
        event_type="hook.context_inject",
        data={
            "layers_included": result.layers_included,
            "layers_skipped": result.layers_skipped,
            "tokens": result.total_tokens,
            "budget": result.budget,
            "build_ms": result.build_time_ms,
        },
        duration_ms=int(result.build_time_ms),
        tokens_out=result.total_tokens,
    )

    return {
        "context": result.text,
        "tokens": result.total_tokens,
        "layers": result.layers_included,
        "build_ms": round(result.build_time_ms, 1),
    }
```

### 2.4 Context Hook Update (`hooks/context.py`)

Phase 1 created a placeholder context-hook that returns empty context. Phase 5 wires it to the worker.

**Design: Dual-path context**

```
SessionStart fires
    â”‚
    â”œâ”€ Worker available? (check worker.sock exists + health)
    â”‚   â”œâ”€ YES â†’ GET /api/context â†’ rich context (all layers)
    â”‚   â””â”€ NO  â†’ Basic context from SQLite directly (session index only)
    â”‚
    â””â”€ Output JSON to stdout
        â””â”€ Claude Code injects additionalContext
```

```python
"""context-hook.py â€” SessionStart handler.

Injected context modes:
1. Worker available: Full progressive disclosure (all layers)
2. Worker unavailable: Basic session list from SQLite (lightweight)
3. Empty DB: No context (brand new project, nothing to inject)
"""
import json
import os
import sys
import socket
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Config
MEM_DIR = Path.home() / ".claude-mem"
DB_PATH = MEM_DIR / "claude-mem.db"
WORKER_SOCK = MEM_DIR / "worker.sock"
CONTEXT_TIMEOUT_S = 5  # Max time to wait for worker response


def main():
    """Entry point. Read stdin, build context, write JSON to stdout."""
    event = json.loads(sys.stdin.read())
    session_id = event.get("session_id", "")
    project_path = event.get("cwd", os.getcwd())
    source = event.get("source", "startup")

    # Create/ensure session exists (idempotent)
    _ensure_session(session_id, project_path, source)

    # Build context
    context_text = ""
    try:
        if WORKER_SOCK.exists():
            context_text = _get_worker_context(project_path, session_id)
        if not context_text:
            context_text = _get_basic_context(project_path)
    except Exception:
        # Graceful degradation: no context is fine
        pass

    # Output to stdout for Claude Code
    if context_text:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": context_text,
            }
        }
        print(json.dumps(output), flush=True)

    sys.exit(0)


def _get_worker_context(project_path: str, session_id: str) -> str:
    """Call worker's /api/context endpoint over UDS.

    Uses raw socket + HTTP/1.1 to avoid importing httpx in the hook
    (which would add ~200ms cold import time).
    """
    import http.client

    try:
        # httpx is too heavy for hook cold start (~200ms import).
        # Use stdlib http.client with UDS via socket monkey-patch.
        conn = _UDSHTTPConnection(str(WORKER_SOCK))
        conn.request(
            "GET",
            f"/api/context?project_path={_url_encode(project_path)}&session_id={session_id}",
        )
        conn.sock.settimeout(CONTEXT_TIMEOUT_S)
        response = conn.getresponse()
        if response.status == 200:
            data = json.loads(response.read())
            return data.get("context", "")
    except (ConnectionRefusedError, TimeoutError, OSError, json.JSONDecodeError):
        return ""
    return ""


class _UDSHTTPConnection(http.client.HTTPConnection):
    """HTTP connection over Unix Domain Socket using stdlib only."""

    def __init__(self, socket_path: str):
        super().__init__("localhost")
        self._socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._socket_path)
        self.sock.settimeout(CONTEXT_TIMEOUT_S)


def _get_basic_context(project_path: str) -> str:
    """Build basic context directly from SQLite (no worker needed).

    Only includes session index â€” the cheapest, most useful layer.
    """
    if not DB_PATH.exists():
        return ""

    conn = sqlite3.connect(str(DB_PATH), timeout=1)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT started_at, summary_text FROM sessions
            WHERE project_path = ? AND summary_text IS NOT NULL
            ORDER BY started_at DESC LIMIT 5
            """,
            (project_path,),
        ).fetchall()
    except sqlite3.Error:
        return ""
    finally:
        conn.close()

    if not rows:
        return ""

    lines = ["## Recent Sessions"]
    for row in rows:
        age = _relative_time(row["started_at"])
        summary = row["summary_text"][:200]
        lines.append(f"- [{age}] {summary}")

    return "\n".join(lines)
```

**Key design choices**:

1. **No httpx import in hooks**: httpx adds ~200ms cold start. We use `http.client` (stdlib) with a UDS socket class. Total overhead: ~2ms import, ~5ms connection.

2. **5-second timeout**: If the worker takes longer than 5s, fall back to basic context. SessionStart hooks should be fast (docs say "keep these hooks fast").

3. **Socket existence check**: `WORKER_SOCK.exists()` is a fast filesystem check (~0.1ms). If the socket file exists but worker is dead, `ConnectionRefusedError` is caught within the timeout.

4. **Graceful everything**: Any exception â†’ empty context â†’ Claude Code session starts normally. No stdout pollution on error.

### 2.5 Configuration

Added to `config.py`:

```python
@dataclass
class ContextConfig:
    """Context injection settings."""
    budget: int = 2000                     # Total token budget
    session_index_cap: int = 400           # Max tokens for session history
    function_map_cap: int = 500            # Max tokens for code changes
    learnings_cap: int = 300               # Max tokens for project knowledge
    observations_cap: int = 600            # Max tokens for relevant observations
    reserve: int = 200                     # Headroom for formatting
    worker_timeout_s: float = 5.0          # Max wait for worker response
    max_sessions: int = 10                 # Max sessions in index
    max_functions: int = 30                # Max functions in map
    max_learnings: int = 10                # Max learnings to show
    max_observations: int = 10             # Max observations to show
    min_learning_confidence: float = 0.5   # Min confidence to include
```

### 2.6 Context Format Optimization

The injected context is consumed by Claude, not humans. Format choices that optimize for LLM comprehension:

**Do**:
- Use `##` headers â€” Claude recognizes these as section boundaries
- Use `- ` list items â€” compact, parseable
- Include relative timestamps (`[2h ago]`) â€” Claude can reason about recency
- Include change type tags (`[NEW]`, `[MODIFIED]`) â€” actionable metadata
- End with a search hint â€” teaches Claude it can get more detail

**Don't**:
- Use ASCII tables â€” token-expensive for the information density
- Use emoji â€” adds tokens without adding information
- Include IDs or UUIDs â€” Claude doesn't need database keys
- Include raw timestamps â€” `2025-02-08T10:23:45Z` is less useful than `[2h ago]`
- Repeat the project path â€” Claude already knows from `cwd`

**Full output example** (all 4 layers, ~1600 tokens):

```
## Recent Sessions
- [2h ago] Implemented JWT auth service, added refresh token endpoint, wrote tests
- [yesterday] Set up FastAPI project structure, configured SQLAlchemy + Alembic
- [2 days ago] Initial project scaffold, Docker setup

## Recently Changed Code
auth/service.py:
  authenticate(email, password) -> Token  [MODIFIED]
  refresh_token(token) -> Token  [NEW]
  _validate_password(plain, hashed) -> bool  [UNCHANGED]
api/routes.py:
  login_endpoint(request) -> Response  [MODIFIED]

## Project Knowledge
- Architecture: FastAPI + SQLAlchemy ORM, Alembic migrations, pytest fixtures
- Convention: JWT tokens in HttpOnly cookies, refresh flow in auth/refresh.py
- Gotcha: user_service.get_by_email() returns None on DB errors â€” always check

## Relevant Past Work
- Added JWT auth middleware: Implemented JWT verification in FastAPI middleware
- Fixed user registration validation: Email uniqueness check now uses DB constraint
- Set up pytest fixtures for auth: Created mock JWT tokens and test user factory

---
Use `curl --unix-socket ~/.claude-mem/worker.sock http://localhost/api/search?q=...` for deeper context.
```

### 2.7 Changelog from Architecture Spec

| Spec Says | PRD Corrects | Rationale |
|---|---|---|
| Layer 1 budget: "~200 tokens", Layer 4: "~300 tokens" | Layer 1: 400, Layer 4: 600 | Architecture spec estimates were too tight. Session summaries average 30â€“50 tokens each Ã— 10 sessions = 300â€“500 tokens. Observation summaries need more room. Rebalanced. |
| Layer 5: "Call graph context (~200 tokens, on demand)" | Removed from context injection | "On demand" = search via SKILL.md, not automatic injection. Call graph is noise for most session starts. |
| Token counting: "tiktoken or simple word-based estimation" | Character-based: `len(text) / 3.5` | tiktoken is wrong tokenizer for Claude (35% error). Character-based is more accurate for Claude and zero-dependency. |
| Budget allocation totals 2000 | Budget allocations total 1800 (2000 - 200 reserve) | Reserve ensures formatting overhead (headers, separators, search hint) doesn't exceed budget. |
| `context/builder.py` only | Added `context/estimator.py` | Separated token estimation into its own module for testability and potential future swap to calibrated model. |

---

## 3. Integration Points

### 3.1 Phase 1 (Hooks)

**File modified**: `hooks/context.py`

Phase 1 created this hook with a placeholder that returns empty context. Phase 5 replaces the placeholder body with the dual-path logic in Â§2.4.

The hook's external interface is unchanged: reads JSON from stdin, writes JSON to stdout. The only difference is the stdout now contains `additionalContext`.

### 3.2 Phase 3 (Worker)

**File modified**: `worker/server.py`

Phase 3 created a placeholder endpoint:
```python
@app.get("/api/context")
async def get_context():
    return {"context": "", "tokens": 0}
```

Phase 5 replaces this with the real implementation from Â§2.3, which accepts `project_path` and `session_id` query parameters.

**Dependency**: The worker must have `app.state.db` (aiosqlite connection) and `app.state.lance_store` (LanceStore, may be None) already initialized by the lifespan manager.

### 3.3 Phase 4 (Search)

**Read-only dependency**: Context builder calls `lance_store.search_observations()` to find semantically relevant past observations. If `lance_store` is None (embeddings unavailable), falls back to recency-based SQLite query.

### 3.4 Future Phases

- **Phase 6 (Learnings)**: When the learnings engine populates the `learnings` table, Layer 3 automatically picks them up. No Phase 5 changes needed.
- **Phase 7 (Eval)**: Eval framework logs `hook.context_inject` events for analysis. Phase 5 already logs these.
- **Phase 9 (Hardening)**: May add `PreCompact` hook to preserve critical context during compaction.

---

## 4. Error Handling & Graceful Degradation

| Failure | Behavior | User Impact |
|---|---|---|
| Worker not running | Hook checks `worker.sock` existence â†’ doesn't exist â†’ basic context from SQLite | Session starts with recent session list only (Layer 1) |
| Worker running but slow (>5s) | `CONTEXT_TIMEOUT_S` fires â†’ fall back to basic context | Same as above |
| Worker running but `/api/context` errors | HTTP non-200 â†’ empty string â†’ basic context from SQLite | Same as above |
| SQLite database doesn't exist | `DB_PATH.exists()` check â†’ no context | Brand new install, nothing to inject |
| SQLite corrupt or locked | `sqlite3.Error` caught â†’ no context | Session starts normally |
| LanceDB unavailable | `lance_store is None` â†’ recency fallback in Layer 4 | Observations layer uses SQLite recency instead of semantic search |
| Layer build raises exception | `asyncio.gather(return_exceptions=True)` â†’ skip that layer | Other layers still included |
| Context exceeds budget | Greedy algorithm skips lower-priority layers | Some layers omitted; most important (session index) always included |
| SessionStart hook output dropped (Claude Code bug #10373/#13650) | Nothing we can do â€” Claude Code issue | User can `/clear` to trigger re-injection (works around the bug) |

**Principle**: Every failure mode results in either (a) reduced context or (b) no context. Never a crash, never a stderr message that Claude sees as an error, never stdout pollution that corrupts JSON parsing.

---

## 5. Test Plan

### 5.1 Test categories

| Category | Tests | What it validates |
|---|---|---|
| **estimator.py** | 4 | Token estimation accuracy across text types (prose, code, mixed, empty) |
| **builder.py â€” layers** | 5 | Each layer builds correctly from fixture data, handles empty tables |
| **builder.py â€” assembly** | 5 | Budget enforcement, priority ordering, layer skipping, concurrent build, exception handling |
| **builder.py â€” format** | 3 | Output format matches spec, relative timestamps, file path relativization |
| **endpoint /api/context** | 3 | Returns context JSON, handles missing project_path, logs event |
| **context-hook.py** | 4 | Worker path (mock HTTP), SQLite fallback path, empty DB, JSON stdout format |
| **Integration** | 3 | Full pipeline: observations in DB â†’ build context â†’ output JSON, token budget respected end-to-end, concurrent requests |
| **Total** | **27** | |

### 5.2 Test infrastructure

```python
# conftest.py additions

@pytest.fixture
def populated_db(tmp_path):
    """SQLite database with sessions, observations, function_map, learnings."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Create tables (Phase 0 schema)
    conn.executescript(SCHEMA_SQL)

    # Insert test data
    now = datetime.now(timezone.utc)
    sessions = [
        ("s1", "/project", (now - timedelta(hours=2)).isoformat(),
         "Implemented JWT auth service, added refresh token endpoint"),
        ("s2", "/project", (now - timedelta(days=1)).isoformat(),
         "Set up FastAPI project structure, configured SQLAlchemy"),
        ("s3", "/project", (now - timedelta(days=2)).isoformat(),
         "Initial project scaffold, Docker setup"),
    ]
    for sid, path, started, summary in sessions:
        conn.execute(
            "INSERT INTO sessions (id, project_path, started_at, summary_text, status) "
            "VALUES (?, ?, ?, ?, 'closed')",
            (sid, path, started, summary),
        )

    # Insert function_map entries
    conn.execute(
        "INSERT INTO function_map (file_path, snapshot_at, session_id, "
        "qualified_name, kind, signature, line_start, line_end, body_hash, change_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("/project/auth/service.py", now.isoformat(), "s1",
         "AuthService.authenticate", "method",
         "authenticate(email, password) -> Token",
         12, 45, "abc123", "modified"),
    )

    # Insert learnings
    conn.execute(
        "INSERT INTO learnings (created_at, updated_at, category, content, confidence, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (now.isoformat(), now.isoformat(), "architecture",
         "FastAPI + SQLAlchemy ORM, Alembic migrations", 0.85, True),
    )

    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def mock_lance_store():
    """Mock LanceStore that returns canned search results."""
    store = Mock()
    store.search_observations.return_value = [
        Mock(title="Added JWT auth middleware",
             summary="Implemented JWT verification in FastAPI middleware"),
        Mock(title="Fixed user registration",
             summary="Email uniqueness check now uses DB constraint"),
    ]
    return store
```

### 5.3 Key test cases

```python
class TestTokenEstimator:
    def test_empty_string(self):
        assert estimate_tokens("") == 1  # minimum 1

    def test_english_prose(self):
        text = "The quick brown fox jumps over the lazy dog."
        tokens = estimate_tokens(text)
        # ~45 chars / 3.5 = ~13 tokens. Actual Claude: ~11. Â±20% is fine.
        assert 8 <= tokens <= 18

    def test_code(self):
        text = "def authenticate(self, email: str, password: str) -> Token:"
        tokens = estimate_tokens(text)
        assert tokens > 5  # Code is token-dense

    def test_budget_math(self):
        """Verify our budget caps sum correctly."""
        from context.builder import ContextBuilder
        total_caps = sum(ContextBuilder.LAYER_CAPS.values())
        assert total_caps == ContextBuilder.DEFAULT_BUDGET


class TestContextBuilder:
    @pytest.mark.asyncio
    async def test_full_build(self, populated_db, mock_lance_store):
        builder = ContextBuilder(populated_db, mock_lance_store)
        result = await builder.build("/project", "new-session")

        assert result.total_tokens <= 2000
        assert "session_index" in result.layers_included
        assert "## Recent Sessions" in result.text

    @pytest.mark.asyncio
    async def test_budget_enforcement(self, populated_db):
        builder = ContextBuilder(populated_db, budget=100)
        result = await builder.build("/project", "s")

        assert result.total_tokens <= 100
        assert len(result.layers_skipped) > 0

    @pytest.mark.asyncio
    async def test_empty_db(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_SQL)
        builder = ContextBuilder(conn)
        result = await builder.build("/project", "s")

        assert result.text == ""
        assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_layer_exception_isolated(self, populated_db):
        """One layer failing shouldn't crash the build."""
        builder = ContextBuilder(populated_db)
        # Corrupt the DB to make function_map query fail
        populated_db.execute("DROP TABLE function_map")

        result = await builder.build("/project", "s")
        # session_index should still work
        assert "session_index" in result.layers_included


class TestContextHook:
    def test_worker_path(self, tmp_path, monkeypatch):
        """When worker is available, use its response."""
        # Mock worker response...

    def test_sqlite_fallback(self, populated_db, monkeypatch):
        """When worker is unavailable, build basic context from SQLite."""
        # Remove worker.sock...

    def test_stdout_json_format(self, populated_db, capsys, monkeypatch):
        """Output must be valid JSON with hookSpecificOutput."""
        # Run main(), capture stdout...
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "hookSpecificOutput" in data
        assert data["hookSpecificOutput"]["hookEventName"] == "SessionStart"
```

### 5.4 Performance targets

| Operation | Target | Rationale |
|---|---|---|
| Token estimation | <0.1ms | String length calculation |
| Context build (worker path) | <200ms | 4 concurrent SQLite queries + 1 LanceDB search |
| Context build (SQLite fallback) | <50ms | Single SQLite query |
| Hook end-to-end (worker path) | <500ms | Import + socket connect + build + serialize |
| Hook end-to-end (fallback) | <100ms | Import + SQLite query + serialize |
| `/api/context` endpoint | <300ms | Build + logging |

---

## 6. Operational Considerations

### 6.1 Hook Timeout

The hook configuration should set a timeout appropriate for context building:

```json
{
  "hooks": {
    "SessionStart": [{
      "matcher": "startup|resume|clear|compact",
      "hooks": [{
        "type": "command",
        "command": "python3 -m claude_mem_lite.hooks.context",
        "timeout": 10
      }]
    }]
  }
}
```

Default Claude Code hook timeout is 60 seconds. We set 10 seconds â€” generous enough for worker communication but fast enough not to delay session start noticeably. The internal `CONTEXT_TIMEOUT_S = 5` ensures we fail fast and fall back to SQLite well within this window.

### 6.2 Context on Resume vs Startup

The hook fires for all session start sources: `startup`, `resume`, `clear`, `compact`. Context injection is useful for all of these:

- **startup**: Full context â€” new session needs everything
- **resume**: Full context â€” Claude lost context from compaction or long gap
- **clear**: Full context â€” user explicitly reset, wants fresh start with memory
- **compact**: Full context â€” context window was compacted, re-inject what was lost

### 6.3 Logging

Every context injection is logged to `event_log` with:
- `event_type`: `hook.context_inject`
- `data`: `{layers_included, layers_skipped, tokens, budget, build_ms}`
- `tokens_out`: total tokens injected

This enables Phase 7 eval to analyze:
- Average tokens injected vs budget
- Which layers are consistently skipped (budget too tight?)
- Context build latency trends
- Correlation between context injection and Claude's effectiveness

### 6.4 Disk / Memory Impact

Zero new disk usage. Context is built on-the-fly from existing data. The builder holds at most 10 session summaries + 30 function signatures + 10 learnings + 10 observation summaries in memory â€” well under 1MB.

---

## 7. Dependencies

### 7.1 Phase 5 runtime dependencies (additions to Phase 4)

**None.** All context building uses:
- `sqlite3`, `json`, `sys`, `os`, `socket`, `http.client`, `datetime`, `time`, `asyncio`, `dataclasses` â€” stdlib
- `aiosqlite` â€” already installed (Phase 3)
- `ContextBuilder` queries only use data structures already defined in Phase 0 models

This is intentional. The context-hook script must remain zero-dependency beyond Python stdlib for fast cold starts.

### 7.2 Phase 5 dev dependencies

None beyond pytest and ruff (already in Phase 0).

---

## 8. Acceptance Criteria

Phase 5 is complete when:

- [ ] All 27 tests pass (pytest, <10s total runtime)
- [ ] `ruff check` and `ruff format --check` pass with zero warnings
- [ ] `GET /api/context` returns context with session summaries, function changes, learnings, and observations
- [ ] `GET /api/context` stays within configured token budget
- [ ] Context hook outputs valid JSON with `hookSpecificOutput.additionalContext`
- [ ] Context hook falls back to SQLite when worker is unavailable
- [ ] Context hook completes within 500ms (worker path) and 100ms (fallback path)
- [ ] Context build handles empty database gracefully (no error, empty context)
- [ ] Context build isolates layer failures (one layer crashing doesn't break others)
- [ ] Each context injection is logged to `event_log` with tokens and timing
- [ ] Token estimation is within Â±20% of actual Claude token count (verified manually on 5 samples)
- [ ] Context format is optimized for Claude (markdown headers, relative timestamps, no tables)
- [ ] No new runtime dependencies

---

## 9. Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Claude Code drops SessionStart hook output (bugs #10373, #13650) | Low (fixed in recent versions) | High | System works without context injection. User can `/clear` to re-trigger. We add no complexity to work around this. |
| Token estimation inaccuracy > 20% | Medium | Low | Over-estimation means less context (conservative). Under-estimation means slightly exceeding budget (~2200 tokens instead of 2000) â€” negligible in Claude's 200K context window. |
| Context hook cold start > 500ms | Low | Medium | Stdlib-only imports in hook path. If problematic, consider shell wrapper or `--warm` mode. |
| Worker latency spike during context build | Low | Low | 5-second timeout + SQLite fallback ensures hook never blocks indefinitely. |
| Stale context from old sessions | Medium | Low | Session index is sorted by recency. Old sessions naturally drop off. Observation search uses semantic relevance, not just time. |
| Injected context ignored by Claude | Medium | Medium | Format follows Claude's known patterns (markdown sections). Phase 7 can measure if Claude references injected context. |

---

## 10. Open Questions

| Question | Current Assumption | Resolution Plan |
|---|---|---|
| What's the ideal token budget? | 2000 tokens | Phase 7 eval: measure context utilization. If layers are consistently skipped, increase budget. If Claude ignores injected context, decrease. |
| Should we inject on `compact` events? | Yes â€” re-inject lost context | Monitor if this causes duplication. If context was just compacted, re-injecting the same info may be redundant. |
| Should Layer 4 use the current session's files as search query? | Currently uses last session's summary | Could also use files open in the editor (not available in hook input) or the project's most-changed files. Evaluate in Phase 7. |
| Should `UserPromptSubmit` also inject context? | No â€” SessionStart only | Per-prompt injection could augment context based on what the user is asking about. Evaluate after Phase 5 is stable. |
| Is `chars / 3.5` the right coefficient? | Yes (based on research) | Phase 7 can calibrate by comparing estimates vs actual API token counts from compression responses. |
