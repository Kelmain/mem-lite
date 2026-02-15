"""FastAPI worker service with lifespan, endpoints, and uvicorn runner."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import aiosqlite
import uvicorn
from fastapi import FastAPI, HTTPException, Query

from claude_mem_lite.storage.models import HealthResponse, QueueStats
from claude_mem_lite.worker.compressor import Compressor
from claude_mem_lite.worker.processor import IdleTracker, Processor
from claude_mem_lite.worker.summarizer import Summarizer

if TYPE_CHECKING:
    from claude_mem_lite.config import Config


_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup/shutdown of worker components."""
    global _start_time  # noqa: PLW0603
    _start_time = time.monotonic()

    config = app.state.config

    # Write PID (defensive — lifecycle already wrote it, but child may differ)
    config.pid_path.write_text(str(os.getpid()))

    db = await aiosqlite.connect(str(config.db_path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await db.execute("PRAGMA busy_timeout=3000")

    compressor = Compressor(config)
    summarizer = Summarizer(db, compressor)
    idle_tracker = IdleTracker(timeout_minutes=30)

    # Phase 4: Load embedding model + connect LanceDB (in threads, non-blocking)
    from claude_mem_lite.search.embedder import Embedder
    from claude_mem_lite.search.lance_store import LanceStore

    embedder = Embedder(config)
    embed_loaded = await asyncio.to_thread(embedder.load)

    lance_store = LanceStore(config, embedder)
    await asyncio.to_thread(lance_store.connect)

    app.state.embedder = embedder
    app.state.lance_store = lance_store

    processor = Processor(
        db,
        compressor,
        idle_tracker,
        lance_store=lance_store if embed_loaded else None,
        embedder=embedder if embed_loaded else None,
    )

    app.state.db = db
    app.state.compressor = compressor
    app.state.summarizer = summarizer
    app.state.idle_tracker = idle_tracker
    app.state.processor = processor

    # Start background tasks
    processor_task = asyncio.create_task(processor.run())
    idle_task = asyncio.create_task(idle_tracker.watch())

    yield

    # Shutdown
    processor_task.cancel()
    idle_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await processor_task
    with contextlib.suppress(asyncio.CancelledError):
        await idle_task
    await compressor.close()
    await db.close()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Phase 3 endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health() -> HealthResponse:
    """Liveness + basic stats."""
    db = app.state.db
    uptime_s = int(time.monotonic() - _start_time)

    cursor = await db.execute("SELECT COUNT(*) as cnt FROM pending_queue WHERE status = 'raw'")
    row = await cursor.fetchone()
    queue_depth = row["cnt"] if row else 0

    cursor = await db.execute(
        "SELECT COUNT(*) as cnt FROM observations WHERE created_at >= date('now')"
    )
    row = await cursor.fetchone()
    obs_today = row["cnt"] if row else 0

    app.state.idle_tracker.touch()
    return HealthResponse(
        status="ok",
        uptime_s=uptime_s,
        queue_depth=queue_depth,
        observations_today=obs_today,
    )


@app.get("/api/context")
async def context(project_path: str = "") -> dict:
    """Build and return context for SessionStart injection."""
    app.state.idle_tracker.touch()

    from claude_mem_lite.context.builder import ContextBuilder

    builder = ContextBuilder(
        db=app.state.db,
        lance_store=getattr(app.state, "lance_store", None),
        budget=app.state.config.context_budget,
        project_dir=project_path,
    )
    result = await builder.build()

    return {
        "context": result.text,
        "tokens": result.total_tokens,
        "layers": result.layers_included,
        "build_ms": round(result.build_time_ms, 1),
    }


@app.post("/api/summarize")
async def summarize(session_id: str) -> dict:
    """Generate session summary."""
    app.state.idle_tracker.touch()
    summary = await app.state.summarizer.summarize_session(session_id)
    result: dict = summary.model_dump()
    return result


@app.get("/api/queue/stats")
async def queue_stats() -> QueueStats:
    """Queue status for debugging."""
    db = app.state.db
    cursor = await db.execute("SELECT status, COUNT(*) as cnt FROM pending_queue GROUP BY status")
    rows = await cursor.fetchall()
    stats = {row["status"]: row["cnt"] for row in rows}

    app.state.idle_tracker.touch()
    return QueueStats(
        raw=stats.get("raw", 0),
        processing=stats.get("processing", 0),
        done=stats.get("done", 0),
        error=stats.get("error", 0),
    )


# ---------------------------------------------------------------------------
# Phase 4 endpoints
# ---------------------------------------------------------------------------


def _format_search_result(r: dict) -> dict:
    """Normalize LanceDB result dict to API response format."""
    return {
        "observation_id": r.get("observation_id", ""),
        "session_id": r.get("session_id", ""),
        "title": r.get("title", ""),
        "summary": r.get("summary", ""),
        "files_touched": r.get("files_touched", ""),
        "score": r.get("_relevance_score", r.get("_distance")),
        "created_at": r.get("created_at", ""),
    }


@app.get("/api/search")
async def search(
    q: str,
    limit: int = Query(default=5, ge=1, le=20),
) -> dict:
    """Hybrid search over observations."""
    app.state.idle_tracker.touch()
    embedder = app.state.embedder
    lance_store = app.state.lance_store

    if embedder.available:
        results = await asyncio.to_thread(lance_store.search_observations, q, limit)
        search_type = "hybrid"
    else:
        results = await asyncio.to_thread(lance_store.search_fts_only, q, limit)
        search_type = "fts"

    return {
        "results": [_format_search_result(r) for r in results],
        "query": q,
        "count": len(results),
        "search_type": search_type,
    }


@app.get("/api/observation/{obs_id}")
async def get_observation(obs_id: str) -> dict:
    """Full observation detail — progressive disclosure."""
    app.state.idle_tracker.touch()
    db = app.state.db
    cursor = await db.execute("SELECT * FROM observations WHERE id = ?", (obs_id,))
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Observation not found")
    return dict(row)


@app.get("/api/callgraph")
async def callgraph(file: str = Query(...)) -> dict:
    """Call graph for a given file (best-effort — populated by Phase 2)."""
    app.state.idle_tracker.touch()
    db = app.state.db

    cursor = await db.execute(
        "SELECT qualified_name, kind, signature FROM function_map WHERE file_path = ?",
        (file,),
    )
    functions = [dict(r) for r in await cursor.fetchall()]

    cursor = await db.execute(
        """SELECT caller_function, callee_file, callee_function, resolution, confidence
           FROM call_graph WHERE caller_file = ?""",
        (file,),
    )
    edges = [dict(r) for r in await cursor.fetchall()]

    return {"functions": functions, "edges": edges}


def run_worker(config: Config) -> None:
    """Run the worker as a uvicorn server on UDS."""
    app.state.config = config
    uvicorn.run(
        app,
        uds=str(config.socket_path),
        log_level="info",
        lifespan="on",
    )


if __name__ == "__main__":
    from claude_mem_lite.config import Config as _Config

    run_worker(_Config())
