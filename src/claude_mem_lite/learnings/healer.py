"""Call graph self-healing from compressed observations."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


class CallGraphHealer:
    """Confirm or discover call graph edges from compressed observations.

    When an observation mentions multiple functions, we treat co-occurrence
    as a signal that those functions are related. Existing edges get their
    confidence boosted; new edges are created at a lower initial confidence.
    """

    CONFIRMATION_BOOST = 0.15
    NEW_EDGE_CONFIDENCE = 0.6
    STALE_DECAY = 0.05
    STALE_SESSION_THRESHOLD = 10

    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def confirm_edges_from_observation(
        self,
        observation: dict,
        session_id: str,
    ) -> dict:
        """Parse observation for function references, confirm call graph edges.

        Uses functions_changed JSON array from the observation.
        For each pair of functions mentioned, check/create an edge.

        Args:
            observation: Dict with optional "functions_changed" JSON string.
            session_id: Current session ID for new edge attribution.

        Returns:
            Dict with "confirmed" and "discovered" counts.
        """
        functions_changed = json.loads(observation.get("functions_changed", "[]"))
        if len(functions_changed) < 2:
            return {"confirmed": 0, "discovered": 0}

        confirmed = 0
        discovered = 0

        for i, caller in enumerate(functions_changed):
            for callee in functions_changed[i + 1 :]:
                result = await self._check_or_create_edge(caller, callee, session_id)
                if result == "confirmed":
                    confirmed += 1
                elif result == "discovered":
                    discovered += 1

        return {"confirmed": confirmed, "discovered": discovered}

    async def _check_or_create_edge(
        self,
        caller: str,
        callee: str,
        session_id: str,
    ) -> str:
        """Check if edge exists, confirm or create it.

        Args:
            caller: Caller function name.
            callee: Callee function name.
            session_id: Session to attribute new edges to.

        Returns:
            "confirmed", "discovered", or "skipped".
        """
        # Check existing edge (in either direction)
        cursor = await self.db.execute(
            """SELECT id, confidence, times_confirmed FROM call_graph
               WHERE (caller_function = ? AND callee_function = ?)
                  OR (caller_function = ? AND callee_function = ?)
               LIMIT 1""",
            (caller, callee, callee, caller),
        )
        row = await cursor.fetchone()

        if row:
            row_dict = dict(row)
            new_conf = min(1.0, row_dict["confidence"] + self.CONFIRMATION_BOOST)
            await self.db.execute(
                "UPDATE call_graph SET confidence = ?, times_confirmed = times_confirmed + 1 "
                "WHERE id = ?",
                (new_conf, row_dict["id"]),
            )
            await self.db.commit()
            return "confirmed"

        # No existing edge -- discover new one
        edge_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        await self.db.execute(
            """INSERT INTO call_graph
               (id, caller_file, caller_function, callee_file, callee_function,
                resolution, confidence, times_confirmed, source, session_id, created_at)
               VALUES (?, '', ?, '', ?, 'unresolved', ?, 0, 'observation', ?, ?)""",
            (edge_id, caller, callee, self.NEW_EDGE_CONFIDENCE, session_id, now),
        )
        await self.db.commit()
        return "discovered"

    async def decay_stale_edges(self, project_path: str) -> int:
        """Decay confidence on observation-sourced edges not confirmed recently.

        Only affects source='observation' edges. AST edges are untouched.
        Requires at least STALE_SESSION_THRESHOLD sessions to judge staleness.

        Args:
            project_path: Project directory to scope the session history to.

        Returns:
            Number of edges whose confidence was decayed.
        """
        cursor = await self.db.execute(
            """SELECT id FROM sessions
               WHERE project_dir = ? AND status IN ('closed', 'summarized')
               ORDER BY started_at DESC
               LIMIT ?""",
            (project_path, self.STALE_SESSION_THRESHOLD),
        )
        recent_sessions = await cursor.fetchall()
        if len(recent_sessions) < self.STALE_SESSION_THRESHOLD:
            return 0

        oldest_recent_id = recent_sessions[-1]["id"]
        oldest_cursor = await self.db.execute(
            "SELECT started_at FROM sessions WHERE id = ?",
            (oldest_recent_id,),
        )
        oldest_row = await oldest_cursor.fetchone()
        if not oldest_row:
            return 0

        result = await self.db.execute(
            """UPDATE call_graph
               SET confidence = MAX(0.05, confidence - ?)
               WHERE source = 'observation'
               AND created_at < ?
               AND times_confirmed = 0""",
            (self.STALE_DECAY, oldest_row["started_at"]),
        )
        await self.db.commit()
        return result.rowcount
