# PRD: Phase 4 â€” Embeddings + Search (v1)

**Project**: claude-mem-lite (fork of claude-mem, Python rewrite)
**Phase**: 4 of 9
**Status**: Draft â€” pending review
**Dependencies**: Phase 3 (Worker Service + Compression â€” FastAPI worker, queue processor, compressor, aiosqlite)
**Estimated effort**: 2â€“3 sessions (~10-16 hours)
**Python**: 3.14.3 (latest stable)

---

## Changelog from Architecture Spec

| Item | Architecture spec (may be outdated) | This PRD (verified Feb 2026) |
|------|-------------------------------------|------------------------------|
| LanceDB version | `>=0.15` (later corrected to 0.29.1 in Phase 0) | **0.29.1** (released Feb 7, 2026 â€” still Alpha, Python >=3.10) |
| Embedding dimension | `truncate_dim=256` (Matryoshka) | **1024** (full native dimension â€” corrected in Phase 0 PRD; truncation to 256/512 available later via config) |
| Qwen3-Embedding context window | "32K context window" | **8K** (`max_length=8192` per HuggingFace model card) |
| Qwen3-Embedding release date | "validated against MTEB-Code leaderboard, Feb 2025" | Released **June 5, 2025** |
| sentence-transformers version | `>=3.0` | **>=5.0** (5.1.1 latest, Jan 27 2026 â€” v5.0 introduced SparseEncoder, native `truncate_dim` on constructor) |
| LanceDB embedding registry | `get_registry().get("sentence-transformers").create(name=..., truncate_dim=256)` | **Custom embedding function** â€” registry doesn't support instruction-aware prefixes or truncation passthrough cleanly; see Â§2.4 |
| Tantivy FTS availability | "LanceDB hybrid search via Tantivy" | Tantivy FTS is **sync API only**; native FTS (`use_tantivy=False`) works in both sync and async. See Â§2.5 |
| LanceDB async API | Not discussed | `connect_async()` / `AsyncTable` exists but has gaps: no `to_pydantic`, hybrid search API less mature. **Use sync API via `asyncio.to_thread()`** â€” see Â§2.5 |
| LanceDB Pydantic models | `from lancedb.pydantic import LanceModel, Vector` | Confirmed â€” `LanceModel` works with sync `Table`, not fully tested with async API |
| Search endpoints | 5 endpoints listed | **3 endpoints** for this phase (search, callgraph, observation). Function-history and learnings deferred to Phase 6. See Â§2.6 |
| SKILL.md token cost | "~30 tokens description cost, ~200-500 tokens when loaded" | Deferred verification â€” SKILL.md authored in this phase, token counting validated during integration testing |
| `transformers` version | Not specified | **>=4.51.0** (required by Qwen3-Embedding, per HuggingFace model card) |

---

## 1. Purpose & Context

### 1.1 What this phase delivers

The search layer that makes compressed observations retrievable. This phase wires up the local embedding model, builds the LanceDB vector+FTS indexes, and exposes hybrid search through the worker's HTTP API.

Specifically:
- **Embedding pipeline** â€” Qwen3-Embedding-0.6B loaded at worker startup, generates 1024-dim vectors for observation titles via sentence-transformers
- **LanceDB storage** â€” three search tables (observations, summaries, learnings) with vector + FTS indexes
- **Hybrid search** â€” vector similarity + BM25 full-text search combined via RRF reranking
- **FTS fallback** â€” if Qwen3 fails to load, fall back to FTS-only search (Tantivy BM25)
- **Search API** â€” `/api/search`, `/api/callgraph`, `/api/observation/{id}` endpoints on the worker
- **Processor integration** â€” after compression, embed observation and write to LanceDB
- **Claude Code skill** â€” `SKILL.md` that teaches Claude to `curl` the worker's search endpoints

### 1.2 What this phase does NOT deliver

- **Context injection** â€” deferred to Phase 5. Search is available on-demand via skill/API, but not auto-injected at SessionStart.
- **Learnings search** â€” the `learnings_vec` table is created but not populated until Phase 6.
- **Function-history endpoint** â€” `GET /api/function-history?name=...` requires cross-referencing function_map snapshots over time. Deferred to Phase 6.
- **Learnings endpoint** â€” `GET /api/learnings?category=...` deferred to Phase 6 when learnings engine exists.
- **A/B testing of search quality** â€” deferred to Phase 7.
- **ONNX quantization** â€” deferred to Phase 9. CPU FP32 inference is acceptable (~80-150ms).

### 1.3 Why this phase matters

Without search, Claude has no way to retrieve relevant past context on demand. Phase 3 compresses and stores observations, but they're only accessible via sequential SQLite queries. Hybrid search enables Claude to ask "what did I do with auth last week?" and get semantically relevant results â€” not just keyword matches.

### 1.4 Relationship to claude-mem (original)

claude-mem uses:
- ChromaDB (Python server process, separate from worker) â€” we use LanceDB (embedded, no server, Rust core)
- OpenAI `text-embedding-3-small` (API call, ~$0.02/1M tokens) â€” we use Qwen3-Embedding-0.6B (local, zero cost after model download)
- Separate FTS via SQLite FTS5 â€” we use LanceDB's built-in Tantivy FTS (unified search interface)

### 1.5 Data flow

```
Phase 3 processor: compress raw â†’ observation in SQLite
        â”‚
        â–¼
Phase 4 addition: embed observation title â†’ LanceDB ObservationIndex
        â”‚
        â”œâ”€â”€ Vector index (1024-dim, cosine similarity)
        â””â”€â”€ FTS index (Tantivy, BM25 on title + summary)

Claude Code (via SKILL.md):
        â”‚
        â–¼
curl GET /api/search?q=...&limit=5
        â”‚
        â–¼
Worker: hybrid search â†’ RRF rerank â†’ return top results
        â”‚
        â”œâ”€â”€ If embeddings available: vector + FTS â†’ RRF merge
        â””â”€â”€ If embeddings unavailable: FTS only (graceful fallback)
```

---

## 2. Technical Specification

### 2.1 Module Structure

```
src/claude_mem_lite/
â”œâ”€â”€ search/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ embedder.py        # Qwen3-Embedding-0.6B wrapper
â”‚   â”œâ”€â”€ hybrid.py          # Hybrid search orchestration
â”‚   â””â”€â”€ lance_store.py     # LanceDB table management (deferred from Phase 0)
â”œâ”€â”€ worker/
â”‚   â”œâ”€â”€ server.py          # Updated: embed model pre-load in lifespan
â”‚   â””â”€â”€ processor.py       # Updated: embed after compress
â””â”€â”€ ...

plugin/
â””â”€â”€ skills/
    â””â”€â”€ mem-search/
        â””â”€â”€ SKILL.md        # Search skill for Claude Code
```

### 2.2 Dependencies (Phase 4 additions)

| Package | Version | Size | Purpose |
|---------|---------|------|---------|
| `lancedb` | >=0.29 | ~15MB (+ pyarrow ~100MB, lance ~20MB) | Embedded vector DB with hybrid search |
| `sentence-transformers` | >=5.0 | ~5MB (+ torch, transformers) | Local embedding model loading and inference |
| `torch` | >=2.2 | ~700MB-2GB | PyTorch backend for Qwen3 decoder model |
| `transformers` | >=4.51.0 | ~10MB | Required by Qwen3-Embedding-0.6B model card |
| `tantivy` | >=0.22 | ~5MB | FTS engine used by LanceDB (auto-installed) |

**Updated `pyproject.toml` dependencies section:**
```toml
dependencies = [
    "pydantic>=2.0",
    "fastapi>=0.128",
    "uvicorn[standard]>=0.30",
    "anthropic>=0.78",
    "aiosqlite>=0.22",
    "httpx>=0.27",
    # Phase 4 additions
    "lancedb>=0.29",
    "sentence-transformers>=5.0",
    "torch>=2.2",
    "transformers>=4.51.0",
]
```

**Note on `torch`**: Already a transitive dep of sentence-transformers but listed explicitly because Qwen3-Embedding-0.6B requires the full PyTorch backend (decoder architecture, not ONNX). On Linux x86_64 with CPU-only, `pip install torch --index-url https://download.pytorch.org/whl/cpu` saves ~1.5GB vs the CUDA build.

**Note on `lancedb`**: Still Alpha (0.29.1). The API surface we use (table creation, add, search, FTS index) has been stable since 0.25.x. Pin to `>=0.29,<0.31` to avoid surprise breaking changes.

**Installation concern**: The combined dependency tree (PyTorch + pyarrow + transformers + lancedb) is heavy (~2-3GB installed). This is a one-time cost for a local tool. If install size becomes a concern, ONNX runtime is explored in Phase 9.

### 2.3 Qwen3-Embedding-0.6B Integration (`embedder.py`)

#### Model loading

```python
from sentence_transformers import SentenceTransformer
from pathlib import Path

class Embedder:
    """Manages Qwen3-Embedding-0.6B for local embedding generation."""

    # Instruction prefixes per query type (from Qwen3 model card)
    INSTRUCTIONS = {
        "observation": "Instruct: Find development observations about code changes and decisions\nQuery: ",
        "code": "Instruct: Find code snippets, functions, and implementation details\nQuery: ",
        "learning": "Instruct: Find project learnings, patterns, and best practices\nQuery: ",
        "document": "",  # No instruction for indexing (raw content)
    }

    def __init__(self, config):
        self.config = config
        self._model: SentenceTransformer | None = None
        self._available = False

    def load(self) -> bool:
        """Load model. Returns True on success, False on failure.

        Called during worker startup (lifespan). Failure is non-fatal â€”
        search falls back to FTS-only.
        """
        try:
            self._model = SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B",
                device="cpu",
                truncate_dim=self.config.embedding_dim,  # default 1024
            )
            self._available = True
            return True
        except Exception as e:
            logger.warning(f"Failed to load Qwen3-Embedding-0.6B: {e}. "
                           "Search will use FTS-only fallback.")
            self._available = False
            return False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def ndims(self) -> int:
        return self.config.embedding_dim

    def embed_texts(
        self,
        texts: list[str],
        query_type: str = "document",
    ) -> list[list[float]]:
        """Embed a list of texts with instruction prefix.

        Args:
            texts: Raw text strings to embed.
            query_type: One of "observation", "code", "learning", "document".
                "document" uses no prefix (for indexing content).
                Others prepend task-specific instructions (for queries).
        """
        if not self._available:
            raise RuntimeError("Embedding model not loaded")

        prefix = self.INSTRUCTIONS.get(query_type, "")
        prefixed = [f"{prefix}{t}" for t in texts] if prefix else texts

        embeddings = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_single(self, text: str, query_type: str = "document") -> list[float]:
        """Convenience: embed a single text."""
        return self.embed_texts([text], query_type)[0]
```

#### Key design decisions

**Why `truncate_dim` on the SentenceTransformer constructor, not manual slicing**: sentence-transformers >=2.7.0 natively supports `truncate_dim` â€” it truncates and re-normalizes in one step. Setting it on the constructor means every `encode()` call automatically produces vectors of the configured dimension. This is cleaner than post-hoc slicing and avoids forgetting to re-normalize.

**Why 1024 (full dimension) as default**: Phase 0 PRD already corrected this from the architecture spec's 256. At our scale (<10K observations over months of use), 1024-dim vectors consume ~40KB per observation (1024 * 4 bytes). 10K observations = ~400MB in LanceDB â€” negligible. The quality difference between 1024 and 256 dimensions is significant for code-related queries (per MTEB-Code benchmarks). Truncation can be enabled later via `config.embedding_dim` if storage becomes a concern.

**Why instruction-aware prefixes**: The Qwen3 team reports 1-5% improvement when using task instructions. For a local tool with small result sets, even marginal quality improvement matters â€” the difference between "good enough" and "actually useful" when Claude searches for past context. Documents (indexed content) get no prefix; queries get task-specific prefixes.

**Why not the LanceDB embedding registry**: The architecture spec proposed `get_registry().get("sentence-transformers").create(name=..., truncate_dim=256)` for automatic embedding. Three problems:
1. The registry's `SentenceTransformerEmbeddings` class doesn't forward `truncate_dim` to the underlying model â€” it would need a custom subclass anyway.
2. Instruction-aware prefixes require different prefixes for indexing vs. querying. The registry applies the same embedding function to both source and query, with no way to differentiate.
3. Manual embedding gives us control over batching, error handling, and fallback behavior.

The tradeoff: we manage embedding calls ourselves instead of getting automatic embedding on `table.add()` and `table.search()`. This means ~20 extra lines of code but full control over the embedding pipeline.

### 2.4 LanceDB Storage (`lance_store.py`)

#### Table schemas

```python
import pyarrow as pa
import lancedb

# Schema for ObservationIndex â€” searchable observation summaries
OBSERVATION_SCHEMA = pa.schema([
    pa.field("observation_id", pa.string()),   # FK to SQLite observations.id
    pa.field("session_id", pa.string()),
    pa.field("title", pa.string()),            # embedded for search
    pa.field("summary", pa.string()),          # returned in results
    pa.field("files_touched", pa.string()),    # comma-separated paths
    pa.field("functions_changed", pa.string()),# comma-separated names
    pa.field("created_at", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 1024)),  # config.embedding_dim
])

# Schema for SummaryIndex â€” searchable session summaries
SUMMARY_SCHEMA = pa.schema([
    pa.field("session_id", pa.string()),
    pa.field("summary_text", pa.string()),     # embedded for search
    pa.field("project_path", pa.string()),
    pa.field("created_at", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 1024)),
])

# Schema for LearningIndex â€” searchable project learnings (populated in Phase 6)
LEARNING_SCHEMA = pa.schema([
    pa.field("learning_id", pa.int64()),       # FK to SQLite learnings.id
    pa.field("content", pa.string()),          # embedded for search
    pa.field("category", pa.string()),
    pa.field("confidence", pa.float64()),
    pa.field("vector", pa.list_(pa.float32(), 1024)),
])
```

**Why pyarrow schema instead of `LanceModel` Pydantic classes**: The architecture spec used `LanceModel` with `VectorField()` and `SourceField()` which are tied to the embedding registry. Since we manage embeddings manually (Â§2.3), we don't benefit from the registry integration. Raw pyarrow schemas are simpler, have no magic, and work identically with both sync and async APIs. They also avoid an implicit dependency on the LanceDB embedding machinery that may change across Alpha releases.

#### LanceStore class

```python
class LanceStore:
    """Manages LanceDB tables for vector + FTS search."""

    def __init__(self, config, embedder: Embedder):
        self.config = config
        self.embedder = embedder
        self._db = None
        self._tables: dict[str, lancedb.table.Table] = {}

    def connect(self):
        """Connect to LanceDB and ensure tables exist."""
        self._db = lancedb.connect(self.config.lance_db_path)
        self._ensure_tables()
        self._ensure_fts_indexes()

    def _ensure_tables(self):
        """Create tables if they don't exist."""
        existing = set(self._db.table_names())
        for name, schema in [
            ("observations_vec", OBSERVATION_SCHEMA),
            ("summaries_vec", SUMMARY_SCHEMA),
            ("learnings_vec", LEARNING_SCHEMA),
        ]:
            if name not in existing:
                self._db.create_table(name, schema=schema)
            self._tables[name] = self._db.open_table(name)

    def _ensure_fts_indexes(self):
        """Create FTS indexes if they don't exist.

        Uses Tantivy (default) with English stemming for better
        keyword matching on natural language observation text.
        """
        try:
            self._tables["observations_vec"].create_fts_index(
                ["title", "summary"],
                use_tantivy=True,
                tokenizer_name="en_stem",
                replace=False,  # don't rebuild if exists
            )
        except Exception:
            pass  # index already exists or table empty â€” both fine

    def add_observation(self, obs_id: str, session_id: str, title: str,
                        summary: str, files_touched: str,
                        functions_changed: str, created_at: str):
        """Embed and store an observation."""
        vector = self.embedder.embed_single(title, query_type="document")
        self._tables["observations_vec"].add([{
            "observation_id": obs_id,
            "session_id": session_id,
            "title": title,
            "summary": summary,
            "files_touched": files_touched,
            "functions_changed": functions_changed,
            "created_at": created_at,
            "vector": vector,
        }])

    def add_summary(self, session_id: str, summary_text: str,
                    project_path: str, created_at: str):
        """Embed and store a session summary."""
        vector = self.embedder.embed_single(summary_text, query_type="document")
        self._tables["summaries_vec"].add([{
            "session_id": session_id,
            "summary_text": summary_text,
            "project_path": project_path,
            "created_at": created_at,
            "vector": vector,
        }])

    def search_observations(self, query: str, limit: int = 5,
                            query_type: str = "observation") -> list[dict]:
        """Hybrid search over observations.

        Returns list of dicts with observation fields + _distance score.
        Falls back to FTS-only if embeddings are unavailable.
        """
        table = self._tables["observations_vec"]

        if self.embedder.available:
            vector = self.embedder.embed_single(query, query_type=query_type)
            results = (
                table.search(query, query_type="hybrid",
                             vector_column_name="vector",
                             fts_columns=["title", "summary"])
                .vector(vector)
                .limit(limit)
                .to_list()
            )
        else:
            # FTS-only fallback
            results = (
                table.search(query)
                .limit(limit)
                .to_list()
            )

        return results
```

### 2.5 Sync vs Async LanceDB in the Worker

**Decision: Use sync LanceDB API wrapped in `asyncio.to_thread()`.**

Rationale:
1. **Tantivy FTS is sync-only**: The Tantivy-based full-text search (which powers our hybrid search) is only available in the Python sync SDK (`lancedb.connect()`, `Table`). The async SDK (`connect_async()`, `AsyncTable`) supports native FTS (`use_tantivy=False`) but native FTS lacks Tantivy's stemming, phrase queries, and mature BM25 scoring.
2. **Async API gaps**: LanceDB's async API is still catching up â€” missing `to_pydantic`, the hybrid search `.search(query_type="hybrid")` API is less documented for async, and there are open issues around API parity (e.g., GitHub issue #2437).
3. **Negligible overhead**: LanceDB is embedded (Rust core, local disk). Search operations complete in 1-50ms. The `asyncio.to_thread()` overhead is ~0.1ms. At our load (~5 req/sec, single user), this is invisible.
4. **Simplicity**: One LanceDB connection (sync), one well-tested code path. No need to maintain both sync and async wrappers.

Implementation pattern:

```python
# In worker/server.py endpoints
from functools import partial

@app.get("/api/search")
async def search(q: str, limit: int = 5, type: str = "observation"):
    results = await asyncio.to_thread(
        app.state.lance_store.search_observations,
        query=q, limit=limit, query_type=type,
    )
    return {"results": results, "query": q, "count": len(results)}
```

**Why not native FTS (`use_tantivy=False`)**: Native FTS is newer, supports async and object storage, and has incremental indexing. But it lacks:
- Stemming tokenizer (English `en_stem`)
- Boolean query operators (AND, OR)
- Phrase search
- Fuzzy search

For a memory system searching natural language observations, stemming and phrase search meaningfully improve result quality. Tantivy via sync + thread pool is the right tradeoff.

### 2.6 Search API Endpoints

Three endpoints for this phase, added to the existing worker `server.py`:

| Method | Path | Purpose | Response |
|--------|------|---------|----------|
| `GET` | `/api/search` | Hybrid search over observations | `{"results": [...], "query": str, "count": int, "search_type": "hybrid"|"fts"}` |
| `GET` | `/api/callgraph` | Call graph for a file (reads SQLite) | `{"file": str, "functions": [...], "edges": [...]}` |
| `GET` | `/api/observation/{id}` | Full observation detail (progressive disclosure) | `{"id": str, "title": str, "summary": str, "detail": str, ...}` |

**Deferred endpoints** (Phase 6):
- `GET /api/function-history?name=...` â€” requires function_map snapshots over time
- `GET /api/learnings?category=...` â€” requires learnings engine

#### `/api/search` detail

```python
@app.get("/api/search")
async def search(
    q: str,
    limit: int = Query(default=5, ge=1, le=20),
    type: str = Query(default="observation", pattern="^(observation|code|learning)$"),
):
    """Hybrid search over observations.

    Query params:
        q: Search query text
        limit: Max results (1-20, default 5)
        type: Query type â€” affects instruction prefix for embeddings
              "observation" (default), "code", "learning"
    """
    idle_tracker.touch()

    results = await asyncio.to_thread(
        lance_store.search_observations,
        query=q, limit=limit, query_type=type,
    )

    search_type = "hybrid" if embedder.available else "fts"

    # Log search for eval (Phase 7)
    logger.log("search.query", {
        "query": q,
        "type": search_type,
        "query_type": type,
        "result_count": len(results),
        "top_score": results[0].get("_distance") if results else None,
    }, duration_ms=elapsed_ms)

    return {
        "results": [_format_result(r) for r in results],
        "query": q,
        "count": len(results),
        "search_type": search_type,
    }

def _format_result(r: dict) -> dict:
    """Format LanceDB result for API response.

    Strips internal fields (_distance, _rowid) and renames for clarity.
    """
    return {
        "observation_id": r["observation_id"],
        "session_id": r["session_id"],
        "title": r["title"],
        "summary": r["summary"],
        "files_touched": r["files_touched"],
        "score": r.get("_relevance_score", r.get("_distance")),
        "created_at": r["created_at"],
    }
```

#### `/api/callgraph` detail

```python
@app.get("/api/callgraph")
async def callgraph(file: str):
    """Return call graph for a specific file.

    Reads from SQLite call_graph + function_map tables (Phase 2 data).
    Returns the most recent snapshot.
    """
    idle_tracker.touch()

    functions = await db.execute_fetchall(
        "SELECT qualified_name, kind, signature, change_type, line_start, line_end "
        "FROM function_map WHERE file_path = ? "
        "ORDER BY snapshot_at DESC LIMIT 50",
        (file,)
    )
    edges = await db.execute_fetchall(
        "SELECT caller_name, callee_name, callee_resolved, resolution "
        "FROM call_graph WHERE caller_file = ? "
        "ORDER BY snapshot_at DESC LIMIT 100",
        (file,)
    )

    return {
        "file": file,
        "functions": [dict(row) for row in functions],
        "edges": [dict(row) for row in edges],
    }
```

#### `/api/observation/{id}` detail

```python
@app.get("/api/observation/{obs_id}")
async def get_observation(obs_id: str):
    """Get full observation details â€” progressive disclosure drill-down.

    The search endpoint returns title + summary (~60 tokens).
    This endpoint returns the full detail field (~200 tokens).
    """
    idle_tracker.touch()

    row = await db.execute_fetchone(
        "SELECT id, session_id, created_at, hook_type, tool_name, "
        "title, summary, detail, files_touched, functions_changed, "
        "raw_size_bytes, compressed_tokens, compression_time_ms "
        "FROM observations WHERE id = ?",
        (obs_id,)
    )
    if not row:
        raise HTTPException(status_code=404, detail="Observation not found")

    return dict(row)
```

### 2.7 Processor Integration

The Phase 3 processor is updated to embed observations after compression:

```python
# In worker/processor.py â€” process_item method

async def process_item(self, item):
    # Phase 3: compress raw â†’ observation
    compressed = await self.compressor.compress(item.payload)
    obs_id = await self._store_observation(item, compressed)

    # Phase 4 addition: embed â†’ LanceDB
    if self.lance_store and self.embedder.available:
        try:
            await asyncio.to_thread(
                self.lance_store.add_observation,
                obs_id=obs_id,
                session_id=item.session_id,
                title=compressed.title,
                summary=compressed.summary,
                files_touched=",".join(compressed.files_touched),
                functions_changed=",".join(
                    f"{fc['name']}" for fc in compressed.functions_changed
                ),
                created_at=item.created_at,
            )
        except Exception as e:
            # Non-fatal: observation is in SQLite, just not searchable via vector
            logger.log("embed.error", {
                "observation_id": obs_id,
                "error": str(e),
            }, session_id=item.session_id)
```

**Embedding failure is non-fatal**: If LanceDB write fails, the observation is still in SQLite and searchable via FTS if the text was indexed. A future catch-up mechanism (Phase 9 hardening) can re-embed missing observations.

### 2.8 Worker Startup Updates (`server.py`)

The worker lifespan from Phase 3 is extended to load the embedding model and connect LanceDB:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Phase 3 startup
    app.state.db = await aiosqlite.connect(config.db_path)
    app.state.db.row_factory = aiosqlite.Row
    app.state.compressor = Compressor(config)
    app.state.idle_tracker = IdleTracker(timeout_minutes=30)

    # Phase 4 additions: embedding model + LanceDB
    app.state.embedder = Embedder(config)
    embed_loaded = app.state.embedder.load()  # ~3-5s, logs warning on failure

    app.state.lance_store = LanceStore(config, app.state.embedder)
    app.state.lance_store.connect()  # creates tables if needed

    if embed_loaded:
        logger.log("worker.startup", {
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dim": config.embedding_dim,
            "lance_tables": list(app.state.lance_store._tables.keys()),
        })

    # Phase 3: processor with Phase 4 lance_store
    app.state.processor = Processor(
        app.state.db,
        app.state.compressor,
        logger,
        app.state.idle_tracker,
        lance_store=app.state.lance_store if embed_loaded else None,
        embedder=app.state.embedder,
    )
    processor_task = asyncio.create_task(app.state.processor.run())

    yield

    processor_task.cancel()
    await app.state.db.close()
```

**Startup time impact**: Qwen3-Embedding-0.6B takes 3-5s to load on CPU (first run downloads ~1.2GB model to HuggingFace cache). Subsequent startups are ~3s (model cached on disk). This is acceptable â€” the worker starts once per session via the `cc` alias.

### 2.9 Configuration Additions

```python
# In config.py

@dataclass
class Config:
    # ... existing Phase 0-3 config ...

    # Phase 4: Embeddings + Search
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024          # Full native dim; can reduce to 256/512
    embedding_device: str = "cpu"      # "cpu" or "cuda" if available
    lance_db_path: str = ""            # Set to ~/.claude-mem/lance/ in __post_init__
    search_limit_default: int = 5
    search_limit_max: int = 20

    def __post_init__(self):
        if not self.lance_db_path:
            self.lance_db_path = str(self.base_dir / "lance")
```

### 2.10 SKILL.md â€” Search Skill for Claude Code

```markdown
# Memory Search

Search past development observations, code changes, and project context.

## Usage

The memory worker must be running (starts automatically with `cc` alias).

### Search observations
```bash
curl -s --unix-socket ~/.claude-mem/worker.sock http://localhost/api/search?q=QUERY&limit=5
```

### Get observation detail
```bash
curl -s --unix-socket ~/.claude-mem/worker.sock http://localhost/api/observation/OBS_ID
```

### Get call graph for a file
```bash
curl -s --unix-socket ~/.claude-mem/worker.sock http://localhost/api/callgraph?file=PATH
```

### Check worker health
```bash
curl -s --unix-socket ~/.claude-mem/worker.sock http://localhost/api/health
```
```

**Token cost**: The SKILL.md is ~150 tokens when loaded. Claude Code loads skills on demand when the skill's trigger pattern matches, so the cost is zero unless Claude decides to search.

**Skill activation**: Claude Code discovers skills from `~/.claude/skills/` or project-local `.claude/skills/`. The `claude-mem install-hooks` CLI (Phase 1) should also install the skill file. Added to acceptance criteria.

---

## 3. Schema Migration

### SQLite migration (v_next)

Add `embedding_status` column to track which observations have been embedded:

```sql
-- Migration: add embedding_status to observations
ALTER TABLE observations ADD COLUMN embedding_status TEXT DEFAULT 'pending';
-- Values: 'pending' | 'embedded' | 'failed'
```

This allows the catch-up mechanism (Phase 9) to find un-embedded observations:

```sql
SELECT id, title FROM observations WHERE embedding_status = 'pending';
```

### LanceDB tables

Created on first worker startup (Â§2.4). No migration system needed â€” LanceDB tables are schemaless in practice (add columns by adding new data with additional fields). If we need to change vector dimensions, we drop and recreate the table (acceptable for a local tool â€” it's a search index, not the source of truth; SQLite is the source of truth).

---

## 4. Embedding Backfill

When Phase 4 is first deployed, there may be existing observations from Phase 3 that lack embeddings. The processor handles this:

```python
async def backfill_embeddings(self):
    """Embed existing observations that are missing from LanceDB.

    Called once at worker startup after initial connection.
    """
    rows = await self.db.execute_fetchall(
        "SELECT id, session_id, title, summary, files_touched, "
        "functions_changed, created_at "
        "FROM observations WHERE embedding_status = 'pending' "
        "ORDER BY created_at DESC LIMIT 100"
    )

    for row in rows:
        try:
            await asyncio.to_thread(
                self.lance_store.add_observation,
                obs_id=row["id"],
                session_id=row["session_id"],
                title=row["title"],
                summary=row["summary"],
                files_touched=row["files_touched"] or "",
                functions_changed=row["functions_changed"] or "",
                created_at=row["created_at"],
            )
            await self.db.execute(
                "UPDATE observations SET embedding_status = 'embedded' WHERE id = ?",
                (row["id"],)
            )
        except Exception as e:
            await self.db.execute(
                "UPDATE observations SET embedding_status = 'failed' WHERE id = ?",
                (row["id"],)
            )
            logger.log("embed.backfill_error", {
                "observation_id": row["id"],
                "error": str(e),
            })

    await self.db.commit()
```

**Limit 100**: Backfilling is capped to avoid a long startup delay. If more than 100 observations exist, subsequent startups will continue backfilling. At ~80-150ms per embedding, 100 observations = ~8-15s.

---

## 5. Graceful Degradation

```
Qwen3-Embedding-0.6B fails to load:
  â†’ LanceDB FTS-only search (Tantivy BM25 on title + summary)
  â†’ /api/search returns search_type="fts"
  â†’ Log "embed.model_load_failed" warning
  â†’ Processor skips embedding step, observations still stored in SQLite

LanceDB fails to connect:
  â†’ Search endpoints return 503 with message
  â†’ Processor stores observations in SQLite only (embedding_status='pending')
  â†’ Worker still functions for compression, summarization

Tantivy FTS index creation fails:
  â†’ Vector-only search if embeddings available
  â†’ Empty results if neither works
  â†’ Log warning, don't crash

Search during backfill:
  â†’ Returns whatever is already indexed
  â†’ Not all observations may be in LanceDB yet â€” acceptable
```

---

## 6. Test Plan

### 6.1 Test categories

| Category | Tests | What it validates |
|----------|-------|-------------------|
| **Embedder** | 5 | Model loads successfully, generates 1024-dim vectors, instruction prefixes applied correctly, `truncate_dim` works at reduced dimensions, graceful failure on load error |
| **LanceStore** | 6 | Tables created on connect, add_observation writes to LanceDB, search returns relevant results, FTS index created, hybrid search merges vector+FTS, empty table search returns empty list |
| **Hybrid search** | 4 | Semantic query finds related observations, keyword query finds exact matches, hybrid outperforms either alone (manual spot check), FTS fallback when embeddings unavailable |
| **API endpoints** | 4 | `/api/search` returns results with correct schema, `/api/callgraph` returns functions+edges, `/api/observation/{id}` returns full detail, 404 for missing observation |
| **Processor integration** | 3 | Compress+embed pipeline end-to-end, embed failure doesn't break compression, embedding_status updated correctly |
| **Backfill** | 2 | Backfill processes pending observations, respects limit cap |
| **Total** | **24** | |

### 6.2 Test infrastructure

```python
# conftest.py additions for Phase 4

@pytest.fixture
def embedder(tmp_config):
    """Real embedder for integration tests, mock for unit tests."""
    e = Embedder(tmp_config)
    e.load()
    return e

@pytest.fixture
def mock_embedder(tmp_config):
    """Mock embedder that returns fixed-dimension random vectors."""
    e = Embedder(tmp_config)
    e._available = True
    e._model = None
    # Monkey-patch to return random vectors
    import numpy as np
    def fake_embed(texts, **kwargs):
        return np.random.randn(len(texts), tmp_config.embedding_dim).tolist()
    e.embed_texts = fake_embed
    return e

@pytest.fixture
def lance_store(tmp_config, mock_embedder):
    """LanceDB store with mock embedder for fast tests."""
    store = LanceStore(tmp_config, mock_embedder)
    store.connect()
    return store
```

**Why mock embedder for most tests**: Loading Qwen3-Embedding-0.6B takes ~3s and requires ~1.2GB of disk/memory. Unit tests should complete in <5s total. Only integration tests marked with `@pytest.mark.slow` use the real model.

### 6.3 Key test scenarios

**Embedder tests** (mock model for unit, real model for integration):
- `test_embed_produces_correct_dims` â€” output shape is (n, 1024)
- `test_instruction_prefix_applied` â€” "observation" query prepends `Instruct: ...`
- `test_document_no_prefix` â€” "document" query has no prefix
- `test_truncate_dim_respected` â€” config.embedding_dim=256 â†’ output is (n, 256)
- `test_load_failure_graceful` â€” invalid model name â†’ available=False, no exception

**LanceStore tests** (mock embedder):
- `test_tables_created` â€” connect() creates all three tables
- `test_add_observation` â€” add then search returns the observation
- `test_search_empty_table` â€” search on empty table returns []
- `test_fts_index_created` â€” FTS search finds keyword matches
- `test_hybrid_search` â€” search with both vector and text
- `test_add_summary` â€” session summary stored and searchable

**API endpoint tests** (mock everything, FastAPI TestClient):
- `test_search_returns_results` â€” GET /api/search?q=auth â†’ status 200, results array
- `test_search_fts_fallback` â€” embedder unavailable â†’ search_type="fts"
- `test_observation_detail` â€” GET /api/observation/abc â†’ full observation
- `test_observation_not_found` â€” GET /api/observation/xxx â†’ 404

**Integration tests** (`@pytest.mark.slow`, real embedder):
- `test_end_to_end_embed_and_search` â€” add 10 observations, search finds relevant ones
- `test_semantic_search_quality` â€” "authentication middleware" finds "Added JWT auth" (not just keyword match)
- `test_backfill_pending` â€” create observations in SQLite with status='pending', run backfill, verify embedded

### 6.4 Performance targets

| Operation | Target | Rationale |
|-----------|--------|-----------|
| Model load (cached) | <5s | One-time at worker startup |
| Single embedding (CPU) | <200ms | Qwen3-0.6B decoder on CPU, 1024-dim |
| Batch embedding (10 texts) | <500ms | Batched inference amortizes overhead |
| LanceDB add (single row) | <10ms | Local disk, Rust core |
| Hybrid search (100 rows in table) | <50ms | Vector scan + FTS + RRF merge |
| Hybrid search (10K rows in table) | <100ms | With IVF_PQ index (see Â§7.1) |
| `/api/search` end-to-end | <300ms | Embed query + search + serialize |
| FTS-only search | <20ms | No embedding needed |
| Backfill (100 observations) | <20s | 100 Ã— 200ms embedding |

---

## 7. Operational Considerations

### 7.1 Vector index creation

For small datasets (<1000 rows), LanceDB performs brute-force scan, which is fast enough. When the observations table grows beyond ~1000 rows, create an IVF_PQ index:

```python
# Not in Phase 4 code â€” manual step when dataset grows
table.create_index(
    "vector",
    index_type="IVF_PQ",
    num_partitions=16,    # sqrt(n) is a good default
    num_sub_vectors=64,   # 1024 / 16 = 64 sub-vectors
)
```

**When to create**: Phase 9 hardening. At our expected usage (~5-20 observations per session, ~200-500 sessions over months), we'd reach 1000 observations after ~50-100 sessions. Brute-force is fine until then.

### 7.2 Model download

First startup downloads Qwen3-Embedding-0.6B (~1.2GB) to `~/.cache/huggingface/hub/`. This is a one-time cost. Document in README:

```bash
# Pre-download model (optional, avoids delay on first `cc` start)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"
```

### 7.3 Disk usage

| Component | Size | Notes |
|-----------|------|-------|
| Qwen3 model (HuggingFace cache) | ~1.2GB | Shared across all tools using this model |
| LanceDB data (1K observations) | ~5MB | Vectors + metadata, Lance columnar format |
| LanceDB data (10K observations) | ~50MB | Linear growth |
| Tantivy FTS index | ~1-5MB | Depends on text volume |

### 7.4 FTS index rebuild

New data added after FTS index creation is still searchable (LanceDB does flat scan on unindexed portion) but with higher latency. For optimal performance after large batch inserts:

```python
table.create_fts_index(["title", "summary"], use_tantivy=True,
                       tokenizer_name="en_stem", replace=True)
```

This is a manual optimization step, not needed during normal operation.

---

## 8. Acceptance Criteria

Phase 4 is complete when:

- [ ] Qwen3-Embedding-0.6B loads at worker startup in <5s (model cached)
- [ ] Observations are embedded after compression and stored in LanceDB `observations_vec` table
- [ ] `GET /api/search?q=...` returns relevant results via hybrid search (vector + FTS)
- [ ] Search falls back to FTS-only when embedding model fails to load
- [ ] `GET /api/observation/{id}` returns full observation detail from SQLite
- [ ] `GET /api/callgraph?file=...` returns functions and edges from SQLite
- [ ] Backfill mechanism embeds pre-existing observations on startup
- [ ] `embedding_status` column tracks embedding state in SQLite observations table
- [ ] SKILL.md is created and installable to `~/.claude/skills/mem-search/`
- [ ] LanceDB tables auto-created on first startup
- [ ] Search latency <300ms end-to-end for typical queries
- [ ] Worker startup <8s with model load (subsequent starts)
- [ ] All 24 tests pass (pytest + pytest-asyncio, <30s with mock embedder, <60s with real model)
- [ ] `ruff check` and `ruff format --check` pass with zero warnings
- [ ] `pip install -e ".[dev]"` installs all Phase 0-4 dependencies cleanly

---

## 9. Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **LanceDB breaking change in Alpha release** | Medium | Medium | Pin `>=0.29,<0.31`. We use a narrow API surface (connect, create_table, add, search, create_fts_index). If API breaks, the wrapper in `lance_store.py` isolates the change. |
| **Qwen3-Embedding-0.6B slow on CPU** | Low | Low | Benchmarks show 80-150ms per embedding. If problematic, Phase 9 explores ONNX INT8 quantization (~2x speedup). |
| **torch dependency bloats install** | High | Low | ~700MB-2GB install size. Acceptable for local tool. CPU-only build (`--index-url https://download.pytorch.org/whl/cpu`) reduces by ~1GB. Document in README. |
| **Hybrid search returns low-quality results** | Medium | High | Log all queries + results for Phase 7 eval. Instruction-aware prefixes improve retrieval quality. Can tune RRF weights if needed. |
| **Tantivy FTS not rebuilt after new data** | Low | Low | New data is still searchable via flat scan. Periodic rebuild via `create_fts_index(replace=True)` in Phase 9 hardening. |
| **Model not downloaded (first run on air-gapped system)** | Low | High | Document pre-download step. Consider bundling model path in config for manual placement. |
| **LanceDB + pyarrow + aiosqlite contention on disk** | Low | Low | LanceDB writes to `~/.claude-mem/lance/`, SQLite to `~/.claude-mem/mem.db`. Separate directories, separate I/O paths. |

---

## 10. Open Questions

| Question | Current assumption | When to resolve |
|----------|-------------------|-----------------|
| **Should we embed title only, or title + summary?** | Title only (10 tokens, focused semantic signal). Summary is searched via FTS. | Phase 7 A/B testing â€” compare title-only vs concatenated embeddings. |
| **Should session summaries be auto-embedded?** | Yes, via `add_summary()` when summarizer runs. | Implementation time â€” verify summarizer (Phase 3) calls embedding path. |
| **Should we create a vector index (IVF_PQ) early?** | No â€” brute-force scan is fast enough for <1000 rows. | Phase 9, or when search latency exceeds 100ms. |
| **Should SKILL.md include search examples?** | Minimal â€” Claude is good at inferring usage from endpoint descriptions. | Iteration during Phase 5 (context injection) when skill is actively used. |
| **Should we support `embedding_dim` change after initial setup?** | No â€” changing dims requires dropping and recreating LanceDB tables. | If needed, add `claude-mem reindex --dim 256` command in Phase 9. |
| **Should there be a CLI search command (`claude-mem search`)?** | Deferred to Phase 8 (CLI Reports). | Phase 8. |
