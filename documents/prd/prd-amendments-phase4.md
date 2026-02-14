# PRD Amendments: Phase 4 (Post-Review)

**Date**: 2026-02-08
**Triggered by**: External review of Phase 4 PRD (5 comments, 2 actionable)
**Affects**: Phase 4 v1 (Embeddings + Search)

---

## Amendment 1: Wrap `embedder.load()` in `asyncio.to_thread()` and wire backfill as background task

**Severity**: Medium
**Affects**: Phase 4, Section 2.8 (`server.py` lifespan)

### Problem

Two issues in the worker startup sequence:

**Issue A â€” Synchronous model load blocks the event loop.**

`app.state.embedder.load()` (line 603) calls `SentenceTransformer(...)` which is a CPU-bound operation taking 3â€“5s. It runs synchronously inside the `lifespan()` async context manager, blocking the event loop for the entire duration. While no HTTP requests are served during lifespan startup, this is still poor async hygiene â€” if other async setup tasks were added (e.g., async health checks, signal handlers), they'd be starved.

More concretely: `embedder.load()` internally calls `torch.load()` and `model.eval()`, both of which are heavy CPU operations that hold the GIL. In an async context, this should be offloaded to a thread.

**Issue B â€” `backfill_embeddings()` is defined but never called.**

Section 4 defines `backfill_embeddings()` (lines 723â€“762) but it is never wired into the lifespan or processor startup. Without an explicit call site, existing observations from Phase 3 will never get embedded â€” they'll remain `embedding_status='pending'` indefinitely.

Additionally, if `backfill_embeddings` were called synchronously in the lifespan (before `yield`), it would block the HTTP server from binding. At 100 observations Ã— ~150ms each, that's ~15s of startup delay where the `cc` alias (`start --daemon && claude`) would be waiting for the socket.

### Specification Change

Replace the lifespan in Section 2.8:

```python
# BEFORE (sync load, no backfill call)
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

# AFTER (async load, backfill as background task)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Phase 3 startup
    app.state.db = await aiosqlite.connect(config.db_path)
    app.state.db.row_factory = aiosqlite.Row
    app.state.compressor = Compressor(config)
    app.state.idle_tracker = IdleTracker(timeout_minutes=30)

    # Phase 4 additions: embedding model + LanceDB
    # Load model in thread â€” SentenceTransformer() is CPU-bound (~3-5s)
    app.state.embedder = Embedder(config)
    embed_loaded = await asyncio.to_thread(app.state.embedder.load)

    app.state.lance_store = LanceStore(config, app.state.embedder)
    await asyncio.to_thread(app.state.lance_store.connect)

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

    # Phase 4: backfill existing observations AFTER yield â€”
    # HTTP server is already bound and accepting requests
    yield

    processor_task.cancel()
    await app.state.db.close()
```

**Backfill call site**: Move backfill into the processor's `run()` method so it executes as part of the background processing loop, after the HTTP server is ready:

```python
# In worker/processor.py â€” run() method
async def run(self):
    """Main processor loop. Runs as background task."""
    # One-time backfill of pre-existing observations
    if self.lance_store and self.embedder and self.embedder.available:
        try:
            await self.backfill_embeddings()
        except Exception as e:
            logger.log("embed.backfill_failed", {"error": str(e)})

    # Normal processing loop
    while True:
        try:
            item = await self.queue.get()
            await self.process_item(item)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.log("processor.error", {"error": str(e)})
```

This ensures:
1. The HTTP socket is bound before any backfill starts (no `cc` alias timeout).
2. Backfill runs once on first startup, before the normal queue processing loop.
3. If backfill fails, the processor still continues to process new items.

### Why `asyncio.to_thread` for `lance_store.connect()` too

`LanceStore.connect()` calls `lancedb.connect()` (Rust FFI, disk I/O) and `create_fts_index()` (Tantivy initialization). Both are sync operations that can take 100â€“500ms on first run. Wrapping in `to_thread()` is consistent with the pattern already used for search operations (Section 2.5).

### Test Impact

No new tests required. Existing lifespan test should verify that the worker accepts HTTP requests before backfill completes. Add one scenario to the existing backfill tests:

- `test_backfill_runs_on_processor_start` â€” start processor with pending observations, verify they get embedded before queue items are processed.

### Performance Target Update

| Operation | Before | After | Notes |
|-----------|--------|-------|-------|
| Lifespan startup (model cached) | ~3-5s (blocking event loop) | ~3-5s (in thread, event loop free) | Same wall time, better async behavior |
| HTTP ready | After model load + LanceDB connect | After model load + LanceDB connect | Same â€” both must complete before `yield` |
| Backfill start | Never (missing call site) | After HTTP ready, in processor loop | Non-blocking to HTTP |

---

## Amendment 2: Fix `curl` URL encoding in SKILL.md

**Severity**: Low
**Affects**: Phase 4, Section 2.10 (SKILL.md)

### Problem

The SKILL.md search example passes the query directly in the URL:

```bash
curl -s --unix-socket ~/.claude-mem/worker.sock http://localhost/api/search?q=QUERY&limit=5
```

This breaks when `QUERY` contains spaces, special characters, or code-related tokens (extremely common for a dev memory tool). Examples that would fail:

- `JWT auth middleware` (spaces)
- `asyncio.to_thread()` (dots, parens)
- `user's config` (apostrophe)
- `file_path = "/home"` (equals, quotes, slashes)

Additionally, the bare `&limit=5` will be interpreted by the shell as a background operator unless the entire URL is quoted.

### Specification Change

Replace the SKILL.md in Section 2.10:

```markdown
# BEFORE
### Search observations
```bash
curl -s --unix-socket ~/.claude-mem/worker.sock http://localhost/api/search?q=QUERY&limit=5
```

### Get call graph for a file
```bash
curl -s --unix-socket ~/.claude-mem/worker.sock http://localhost/api/callgraph?file=PATH
```

# AFTER
### Search observations
```bash
curl -s -G --unix-socket ~/.claude-mem/worker.sock http://localhost/api/search \
    --data-urlencode "q=QUERY" \
    -d "limit=5"
```

### Get call graph for a file
```bash
curl -s -G --unix-socket ~/.claude-mem/worker.sock http://localhost/api/callgraph \
    --data-urlencode "file=PATH"
```
```

### Why `-G` + `--data-urlencode`

- `-G` tells `curl` to send data as GET query parameters (not POST body).
- `--data-urlencode` automatically percent-encodes the value, handling spaces, special chars, and unicode.
- `-d "limit=5"` doesn't need encoding (it's always a plain integer), but is combined with `-G` to append as a query parameter.

The observation detail endpoint (`/api/observation/{id}`) doesn't need this fix â€” observation IDs are UUIDs with no special characters.

### Test Impact

None â€” SKILL.md is documentation, not code. But integration tests for the search endpoint should include a test case with spaces in the query to verify the server handles URL-decoded parameters correctly (FastAPI does this automatically via Starlette).

---

## Summary of Changes

| # | Amendment | Section | Severity | Lines affected |
|---|-----------|---------|----------|----------------|
| 1 | Async `embedder.load()` + backfill call site | Â§2.8, Â§4 | Medium | ~25 lines changed in lifespan, ~10 lines added to processor.run() |
| 2 | URL-encode `curl` queries in SKILL.md | Â§2.10 | Low | ~6 lines changed in SKILL.md |

### Rejected Review Comments (with rationale)

| # | Comment | Verdict | Rationale |
|---|---------|---------|-----------|
| 1 | "Python 3.14 vs PyTorch deadlock" | **Rejected** | PyTorch 2.10.0 (Jan 21, 2026) ships cp314 wheels. Python 3.14 is fully supported. |
| 2 | "FTS freshness gap â€” new data not searchable" | **Rejected** | LanceDB docs confirm new data is searchable via flat scan on unindexed portion. PRD Â§7.4 and Â§9 risk table already document this. |
| 5 | "'Lite' identity crisis â€” 3GB install" | **Acknowledged, no change** | PRD Â§2.2 already documents the install size. "Lite" refers to architecture (no external servers), not disk weight. README can clarify if desired â€” not a PRD concern. |
