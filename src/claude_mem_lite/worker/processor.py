"""Queue processor — polls pending_queue, orchestrates compression."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

    from claude_mem_lite.learnings.engine import LearningsEngine
    from claude_mem_lite.learnings.healer import CallGraphHealer
    from claude_mem_lite.search.embedder import Embedder
    from claude_mem_lite.search.lance_store import LanceStore
    from claude_mem_lite.worker.compressor import Compressor
    from claude_mem_lite.worker.summarizer import Summarizer

from claude_mem_lite.storage.models import CompressedObservation, PendingQueueItem
from claude_mem_lite.worker.exceptions import NonRetryableError, RetryableError

logger = logging.getLogger(__name__)


class IdleTracker:
    """Track last activity, trigger shutdown after inactivity."""

    def __init__(self, timeout_minutes: int = 30):
        self.timeout = timeout_minutes * 60
        self.last_activity = 0.0  # set on first touch
        self._shutdown_event = asyncio.Event()

    def touch(self):
        self.last_activity = asyncio.get_event_loop().time()

    async def watch(self):
        """Background task — sets shutdown event when idle timeout exceeded."""
        while True:
            await asyncio.sleep(60)
            if self.last_activity == 0.0:
                continue
            elapsed = asyncio.get_event_loop().time() - self.last_activity
            if elapsed >= self.timeout:
                self._shutdown_event.set()
                return

    @property
    def should_shutdown(self) -> bool:
        return self._shutdown_event.is_set()


class Processor:
    """Poll pending_queue, compress raw observations, store results."""

    POLL_INTERVAL = 2.0
    BATCH_SIZE = 5
    MAX_ATTEMPTS = 3
    BACKOFF_BASE = 5.0
    SUMMARY_IDLE_MINUTES = 2
    BACKFILL_LIMIT = 100

    def __init__(
        self,
        db: aiosqlite.Connection,
        compressor: Compressor,
        idle_tracker: IdleTracker,
        summarizer: Summarizer | None = None,
        lance_store: LanceStore | None = None,
        embedder: Embedder | None = None,
        learnings_engine: LearningsEngine | None = None,
        call_graph_healer: CallGraphHealer | None = None,
    ):
        self.db = db
        self.compressor = compressor
        self.idle_tracker = idle_tracker
        self.summarizer = summarizer
        self.lance_store = lance_store
        self.embedder = embedder
        self.learnings_engine = learnings_engine
        self.call_graph_healer = call_graph_healer

    async def run(self):
        """Main processing loop — runs as asyncio task."""
        await self.recover_orphaned_items()

        # Phase 4: one-time backfill of pre-existing observations
        if self.lance_store and self.embedder and self.embedder.available:
            try:
                count = await self.backfill_embeddings()
                if count > 0:
                    logger.info("Backfilled %d observations", count)
            except Exception:
                logger.warning("Backfill failed", exc_info=True)

        while not self.idle_tracker.should_shutdown:
            items = await self.dequeue_batch()
            if items:
                for item in items:
                    self.idle_tracker.touch()
                    await self.process_item(item)
            else:
                await self._check_pending_summaries()
                await asyncio.sleep(self.POLL_INTERVAL)

    async def dequeue_batch(self) -> list[PendingQueueItem]:
        """Atomically claim items from pending_queue."""
        await self.db.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self.db.execute(
                """
                UPDATE pending_queue SET status = 'processing'
                WHERE status = 'raw'
                  AND id IN (
                    SELECT id FROM pending_queue
                    WHERE status = 'raw'
                    ORDER BY CASE priority
                        WHEN 'high' THEN 0
                        WHEN 'normal' THEN 1
                        WHEN 'low' THEN 2
                        ELSE 1
                    END, created_at
                    LIMIT ?
                  )
                RETURNING *
                """,
                (self.BATCH_SIZE,),
            )
            rows = await cursor.fetchall()
            await self.db.execute("COMMIT")
        except Exception:
            await self.db.execute("ROLLBACK")
            raise
        return [PendingQueueItem(**dict(r)) for r in rows]

    async def process_item(self, item: PendingQueueItem):
        """Compress a single queue item and store the observation."""
        try:
            result = await self.compressor.compress(
                raw_output=item.raw_output,
                tool_name=item.tool_name,
                files_touched=item.files_touched,
            )
            obs_id = await self._store_observation(item, result)

            # Phase 4: embed after compress (non-fatal)
            await self._embed_observation(obs_id, item, result)

            # Phase 6: call graph healing from observation (non-fatal)
            await self._post_observation(item, result)

            await self._mark_done(item.id)
        except RetryableError as e:
            await self._handle_retry(item, e)
        except NonRetryableError as e:
            await self._mark_error(item.id, str(e))

    async def _embed_observation(
        self,
        obs_id: str,
        item: PendingQueueItem,
        result: CompressedObservation,
    ) -> None:
        """Embed observation into LanceDB. Non-fatal on failure."""
        if not self.lance_store or not self.embedder or not self.embedder.available:
            return

        try:
            await asyncio.to_thread(
                self.lance_store.add_observation,
                obs_id=obs_id,
                session_id=item.session_id,
                title=result.title,
                summary=result.summary,
                files_touched=",".join(result.files_touched),
                functions_changed=",".join(fc.name for fc in result.functions_changed),
                created_at=datetime.now(UTC).isoformat(),
            )
            await self.db.execute(
                "UPDATE observations SET embedding_status = 'embedded' WHERE id = ?",
                (obs_id,),
            )
            await self.db.commit()
        except Exception:
            logger.warning("Embedding failed for %s", obs_id, exc_info=True)
            await self.db.execute(
                "UPDATE observations SET embedding_status = 'failed' WHERE id = ?",
                (obs_id,),
            )
            await self.db.commit()

    async def _store_observation(
        self, item: PendingQueueItem, result: CompressedObservation
    ) -> str:
        """Store compressed observation in the observations table. Returns obs_id."""
        obs_id = str(uuid.uuid4())
        functions_json = json.dumps([fc.model_dump() for fc in result.functions_changed])
        files_json = json.dumps(result.files_touched)

        await self.db.execute(
            """INSERT INTO observations
               (id, session_id, tool_name, title, summary, detail,
                files_touched, functions_changed, tokens_raw, tokens_compressed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                obs_id,
                item.session_id,
                item.tool_name,
                result.title,
                result.summary,
                result.detail,
                files_json,
                functions_json,
                result.tokens_in,
                result.tokens_out,
            ),
        )
        await self.db.execute(
            "UPDATE sessions SET observation_count = observation_count + 1 WHERE id = ?",
            (item.session_id,),
        )
        await self.db.commit()
        return obs_id

    async def _mark_done(self, item_id: str):
        await self.db.execute(
            "UPDATE pending_queue SET status = 'done' WHERE id = ?",
            (item_id,),
        )
        await self.db.commit()

    async def _mark_error(self, item_id: str, error_msg: str):
        await self.db.execute(
            "UPDATE pending_queue SET status = 'error', attempts = attempts + 1 WHERE id = ?",
            (item_id,),
        )
        await self.db.commit()
        logger.warning("Non-retryable error for item %s: %s", item_id, error_msg)

    async def _handle_retry(self, item: PendingQueueItem, error: Exception):
        """Exponential backoff retry logic."""
        new_attempts = item.attempts + 1
        if new_attempts >= self.MAX_ATTEMPTS:
            await self._mark_error(item.id, f"Max retries exceeded: {error}")
            return

        logger.info(
            "Retrying item %s (attempt %d, backoff %.1fs): %s",
            item.id,
            new_attempts,
            self.BACKOFF_BASE * (2 ** (new_attempts - 1)),
            error,
        )

        await self.db.execute(
            "UPDATE pending_queue SET status = 'raw', attempts = ? WHERE id = ?",
            (new_attempts, item.id),
        )
        await self.db.commit()

    async def recover_orphaned_items(self):
        """Reset all items stuck in 'processing' state.

        Called on worker startup. Since we enforce single-worker via PID file
        and the start() method stops any existing worker before spawning a new
        one, any item in 'processing' state at startup is an orphan from a
        dead worker. No time threshold needed.
        """
        cursor = await self.db.execute(
            "UPDATE pending_queue SET status = 'raw' WHERE status = 'processing'"
        )
        count = cursor.rowcount
        await self.db.commit()

        if count > 0:
            logger.info("Recovered %d orphaned items from processing state", count)

    async def backfill_embeddings(self) -> int:
        """Embed existing observations missing from LanceDB. Returns count processed."""
        if not self.lance_store or not self.embedder or not self.embedder.available:
            return 0

        cursor = await self.db.execute(
            "SELECT id, session_id, title, summary, files_touched, "
            "functions_changed, created_at "
            "FROM observations WHERE embedding_status = 'pending' "
            "ORDER BY created_at DESC LIMIT ?",
            (self.BACKFILL_LIMIT,),
        )
        rows = await cursor.fetchall()
        count = 0

        for row in rows:
            row_dict = dict(row)
            try:
                await asyncio.to_thread(
                    self.lance_store.add_observation,
                    obs_id=row_dict["id"],
                    session_id=row_dict["session_id"],
                    title=row_dict["title"],
                    summary=row_dict["summary"],
                    files_touched=row_dict["files_touched"] or "",
                    functions_changed=row_dict["functions_changed"] or "",
                    created_at=row_dict["created_at"],
                )
                await self.db.execute(
                    "UPDATE observations SET embedding_status = 'embedded' WHERE id = ?",
                    (row_dict["id"],),
                )
                count += 1
            except Exception:
                await self.db.execute(
                    "UPDATE observations SET embedding_status = 'failed' WHERE id = ?",
                    (row_dict["id"],),
                )
                logger.warning("Backfill failed for %s", row_dict["id"], exc_info=True)

        await self.db.commit()
        return count

    async def _check_pending_summaries(self):
        """Detect sessions ready for summarization.

        A session is ready for summary when:
        1. It has a 'hook.stop' event logged (session activity ended)
        2. It has no remaining 'raw' or 'processing' items in pending_queue
        3. It has at least one observation (something to summarize)
        4. It hasn't been summarized yet (sessions.summary IS NULL)
        5. The last activity was >SUMMARY_IDLE_MINUTES ago (debounce)
        """
        if self.summarizer is None:
            return

        idle_threshold = (
            datetime.now(UTC) - timedelta(minutes=self.SUMMARY_IDLE_MINUTES)
        ).isoformat()

        cursor = await self.db.execute(
            """
            SELECT DISTINCT s.id
            FROM sessions s
            INNER JOIN observations o ON o.session_id = s.id
            INNER JOIN event_log e ON e.session_id = s.id
                AND json_extract(e.data, '$.event') = 'hook.stop'
            WHERE s.summary IS NULL
              AND s.status != 'closed'
              AND NOT EXISTS (
                  SELECT 1 FROM pending_queue pq
                  WHERE pq.session_id = s.id
                    AND pq.status IN ('raw', 'processing')
              )
              AND e.created_at < ?
            LIMIT 3
            """,
            (idle_threshold,),
        )
        sessions = await cursor.fetchall()

        for row in sessions:
            session_id = row[0]
            try:
                self.idle_tracker.touch()
                summary_result = await self.summarizer.summarize_session(session_id)
                logger.info("Auto-summarized session %s", session_id)

                # Phase 6: learning extraction + stale edge decay
                await self._post_summarization(session_id, summary_result.summary)
            except Exception:
                logger.exception("Failed to auto-summarize session %s", session_id)

    async def _post_observation(
        self,
        item: PendingQueueItem,
        result: CompressedObservation,
    ) -> None:
        """Phase 6: call graph healing after observation stored."""
        if not self.call_graph_healer:
            return
        try:
            obs_dict = {
                "functions_changed": json.dumps([fc.name for fc in result.functions_changed]),
            }
            await self.call_graph_healer.confirm_edges_from_observation(obs_dict, item.session_id)
        except Exception:
            logger.warning("Call graph healing failed for item %s", item.id, exc_info=True)

    async def _post_summarization(
        self,
        session_id: str,
        summary_text: str,
    ) -> None:
        """Phase 6: learning extraction + stale edge decay after summarization."""
        if not self.learnings_engine:
            return
        try:
            # Get project_dir for the session
            cursor = await self.db.execute(
                "SELECT project_dir FROM sessions WHERE id = ?", (session_id,)
            )
            session_row = await cursor.fetchone()
            if not session_row:
                return
            project_path = session_row["project_dir"]

            # Get observations for extraction prompt
            cursor = await self.db.execute(
                "SELECT title, summary, detail, files_touched, functions_changed "
                "FROM observations WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            )
            observations = [dict(r) for r in await cursor.fetchall()]

            # Extract learnings
            await self.learnings_engine.extract_from_session(
                session_id=session_id,
                summary=summary_text,
                observations=observations,
                project_path=project_path,
            )

            # Decay stale call graph edges
            if self.call_graph_healer:
                await self.call_graph_healer.decay_stale_edges(project_path)

        except Exception:
            logger.warning(
                "Post-summarization failed for session %s",
                session_id,
                exc_info=True,
            )
