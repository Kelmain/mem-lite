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

    def __init__(
        self,
        db: aiosqlite.Connection,
        compressor: Compressor,
        idle_tracker: IdleTracker,
        summarizer: Summarizer | None = None,
    ):
        self.db = db
        self.compressor = compressor
        self.idle_tracker = idle_tracker
        self.summarizer = summarizer

    async def run(self):
        """Main processing loop — runs as asyncio task."""
        await self.recover_orphaned_items()
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
            await self._store_observation(item, result)
            await self._mark_done(item.id)
        except RetryableError as e:
            await self._handle_retry(item, e)
        except NonRetryableError as e:
            await self._mark_error(item.id, str(e))

    async def _store_observation(self, item: PendingQueueItem, result: CompressedObservation):
        """Store compressed observation in the observations table."""
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
                await self.summarizer.summarize_session(session_id)
                logger.info("Auto-summarized session %s", session_id)
            except Exception:
                logger.exception("Failed to auto-summarize session %s", session_id)
