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
from fastapi import FastAPI

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
    processor = Processor(db, compressor, idle_tracker)

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
async def context() -> dict:
    """Context injection placeholder (Phase 5)."""
    app.state.idle_tracker.touch()
    return {"context": "", "tokens": 0}


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
