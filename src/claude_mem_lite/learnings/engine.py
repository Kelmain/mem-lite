"""Extract, deduplicate, and evolve project learnings from session data."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING

from claude_mem_lite.learnings.prompts import (
    LEARNING_EXTRACTION_SCHEMA,
    LEARNING_EXTRACTION_SYSTEM,
    build_extraction_prompt,
)

if TYPE_CHECKING:
    import aiosqlite

    from claude_mem_lite.config import Config
    from claude_mem_lite.search.lance_store import LanceStore

logger = logging.getLogger(__name__)


class LearningsEngine:
    """Extract, deduplicate, and evolve project learnings."""

    INITIAL_CONFIDENCE = 0.5
    MAX_AUTO_CONFIDENCE = 0.95
    BOOST_FACTOR = 0.2
    CONTRADICTION_PENALTY = 0.3
    MIN_CONFIDENCE = 0.1

    def __init__(
        self,
        db: aiosqlite.Connection,
        client: object,
        lance_store: LanceStore | None,
        config: Config,
    ) -> None:
        self.db = db
        self.client = client
        self.lance_store = lance_store
        self.config = config

    async def extract_from_session(
        self,
        session_id: str,
        summary: str,
        observations: list[dict],
        project_path: str,
    ) -> list[dict]:
        """Extract learnings from a session summary and process them.

        Returns a list of result dicts with 'action', 'content', 'category' keys.
        Non-fatal: returns empty list on any API or parsing error.
        """
        existing = await self._get_active_learnings(project_path)

        candidates = await self._call_extraction_api(summary, observations, existing)
        if not candidates:
            return []

        results = []
        for candidate in candidates:
            # Skip low-confidence candidates
            if candidate.get("confidence", 0) < 0.3:
                logger.debug(
                    "Skipping low-confidence candidate: %s",
                    candidate.get("content", "")[:50],
                )
                continue

            result = await self._process_candidate(candidate, session_id, project_path)
            results.append(result)

        return results

    async def _call_extraction_api(
        self,
        summary: str,
        observations: list[dict],
        existing: list[dict],
    ) -> list[dict]:
        """Call Claude API with structured outputs to extract learnings."""
        user_msg = build_extraction_prompt(summary, observations, existing)

        try:
            response = await self.client.messages.create(
                model=self.config.compression_model,
                max_tokens=1024,
                system=LEARNING_EXTRACTION_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                extra_body={
                    "output_config": {
                        "format": {
                            "type": "json_schema",
                            "schema": LEARNING_EXTRACTION_SCHEMA,
                        }
                    }
                },
            )
        except Exception:
            logger.warning("Learning extraction API call failed", exc_info=True)
            return []

        try:
            text_block = response.content[0]
            data = json.loads(text_block.text)
            return data.get("learnings", [])
        except (json.JSONDecodeError, IndexError, AttributeError):
            logger.warning("Failed to parse learning extraction response", exc_info=True)
            return []

    async def _get_active_learnings(self, _project_path: str) -> list[dict]:
        """Get all active learnings from the database.

        Args:
            _project_path: Reserved for future per-project filtering.
        """
        cursor = await self.db.execute(
            "SELECT id, category, content, confidence FROM learnings "
            "WHERE is_active = 1 "
            "ORDER BY confidence DESC"
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "category": row["category"],
                "content": row["content"],
                "confidence": row["confidence"],
            }
            for row in rows
        ]

    async def _find_duplicate(
        self,
        content: str,
        category: str,
    ) -> dict | None:
        """Find an existing learning that duplicates the given content.

        Uses LanceDB vector similarity if available, falls back to SQL LIKE.
        """
        if self.lance_store is not None:
            results = await asyncio.to_thread(
                self.lance_store.search_learnings,
                query=content,
                limit=3,
            )
            for r in results:
                if r.get("score", 0) >= self.config.learning_dedup_threshold:
                    # Fetch full DB row for this learning
                    cursor = await self.db.execute(
                        "SELECT id, category, content, confidence, times_seen, "
                        "source_sessions FROM learnings "
                        "WHERE id = ? AND is_active = 1",
                        (r["learning_id"],),
                    )
                    row = await cursor.fetchone()
                    if row:
                        return dict(row)
            return None

        # Fallback: SQL substring match
        cursor = await self.db.execute(
            "SELECT id, category, content, confidence, times_seen, source_sessions "
            "FROM learnings "
            "WHERE category = ? AND is_active = 1 "
            "AND (content LIKE '%' || ? || '%' OR ? LIKE '%' || content || '%') "
            "LIMIT 1",
            (category, content, content),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def _process_candidate(
        self,
        candidate: dict,
        session_id: str,
        _project_path: str = "",
    ) -> dict:
        """Process a single learning candidate: insert, merge, or handle contradiction.

        Args:
            _project_path: Reserved for future per-project filtering.

        Returns a dict with 'action', 'content', 'category' keys.
        """
        content = candidate["content"]
        category = candidate["category"]
        contradicts = candidate.get("contradicts")

        # Handle contradiction
        if contradicts:
            return await self._handle_contradiction(candidate, contradicts, session_id)

        # Check for duplicate
        existing = await self._find_duplicate(content, category)
        if existing:
            return await self._merge_learning(existing, candidate, session_id)

        # Insert new learning
        return await self._insert_learning(candidate, session_id)

    async def _handle_contradiction(
        self,
        candidate: dict,
        contradicts_content: str,
        session_id: str,
    ) -> dict:
        """Penalize the contradicted learning and insert the new one."""
        # Find the contradicted learning
        cursor = await self.db.execute(
            "SELECT id, confidence FROM learnings WHERE content = ? AND is_active = 1",
            (contradicts_content,),
        )
        old_row = await cursor.fetchone()

        if old_row:
            new_conf = self._penalize_confidence(old_row["confidence"])
            await self.db.execute(
                "UPDATE learnings SET confidence = ?, updated_at = datetime('now') WHERE id = ?",
                (new_conf, old_row["id"]),
            )
            logger.info(
                "Penalized learning %s: %.2f -> %.2f",
                old_row["id"],
                old_row["confidence"],
                new_conf,
            )

        # Insert new as fresh entry
        await self._insert_learning(candidate, session_id)
        await self.db.commit()

        return {
            "action": "contradiction",
            "content": candidate["content"],
            "category": candidate["category"],
        }

    async def _merge_learning(
        self,
        existing: dict,
        candidate: dict,
        session_id: str,
    ) -> dict:
        """Merge a duplicate candidate into an existing learning."""
        old_confidence = existing["confidence"]
        new_confidence = self._boost_confidence(old_confidence)
        new_times_seen = existing["times_seen"] + 1

        # Update source_sessions
        sessions = json.loads(existing.get("source_sessions", "[]"))
        if session_id not in sessions:
            sessions.append(session_id)

        # Revival detection
        if old_confidence < 0.5 and candidate.get("confidence", 0) >= 0.5:
            logger.info("Learning revived: %s", existing["id"])

        await self.db.execute(
            "UPDATE learnings SET confidence = ?, times_seen = ?, "
            "source_sessions = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (new_confidence, new_times_seen, json.dumps(sessions), existing["id"]),
        )
        await self.db.commit()

        return {
            "action": "merged",
            "content": candidate["content"],
            "category": candidate["category"],
            "existing_id": existing["id"],
        }

    async def _insert_learning(
        self,
        candidate: dict,
        session_id: str,
    ) -> dict:
        """Insert a new learning into the database."""
        learning_id = str(uuid.uuid4())
        content = candidate["content"]
        category = candidate["category"]

        await self.db.execute(
            "INSERT INTO learnings "
            "(id, category, content, confidence, source_session_id, "
            "source_sessions, times_seen, is_manual, is_active, "
            "created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))",
            (
                learning_id,
                category,
                content,
                self.INITIAL_CONFIDENCE,
                session_id,
                json.dumps([session_id]),
                1,
                0,
                1,
            ),
        )
        await self.db.commit()

        # Embed in LanceDB if available
        if self.lance_store is not None:
            try:
                await asyncio.to_thread(
                    self.lance_store.add_learning,
                    learning_id,
                    category,
                    content,
                )
            except Exception:
                logger.warning(
                    "Failed to embed learning %s in LanceDB",
                    learning_id,
                    exc_info=True,
                )

        return {
            "action": "inserted",
            "content": content,
            "category": category,
            "learning_id": learning_id,
        }

    def _boost_confidence(self, current: float) -> float:
        """Boost confidence with diminishing returns toward MAX_AUTO_CONFIDENCE."""
        return min(
            self.MAX_AUTO_CONFIDENCE,
            current + self.BOOST_FACTOR * (self.MAX_AUTO_CONFIDENCE - current),
        )

    def _penalize_confidence(self, current: float) -> float:
        """Penalize confidence, clamped at MIN_CONFIDENCE."""
        return max(self.MIN_CONFIDENCE, current - self.CONTRADICTION_PENALTY)
