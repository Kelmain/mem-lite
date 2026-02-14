"""Session summarization -- aggregate observations into summary."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from claude_mem_lite.storage.models import SessionSummary
from claude_mem_lite.worker.prompts import SUMMARY_SCHEMA

if TYPE_CHECKING:
    import aiosqlite

    from claude_mem_lite.worker.compressor import Compressor

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """\
Summarize this development session from its observations.

Session observations (chronological):
{observations}

Return a JSON object with these fields:
{{
  "summary": "2-4 sentence summary of what was accomplished. Focus on outcomes, not process.",
  "key_files": ["list of most important files changed"],
  "key_decisions": ["any significant technical decisions made"]
}}
"""


class Summarizer:
    """Generate session summaries from accumulated observations."""

    def __init__(self, db: aiosqlite.Connection, compressor: Compressor) -> None:
        self.db = db
        self.compressor = compressor

    async def summarize_session(self, session_id: str) -> SessionSummary:
        """Generate summary from all observations in a session."""
        observations = await self._get_session_observations(session_id)
        if not observations:
            empty = SessionSummary(
                summary="No observations captured.", key_files=[], key_decisions=[]
            )
            await self._store_summary(session_id, empty)
            return empty

        # Build observation digest
        obs_text = "\n".join(
            f"- [{obs['tool_name']}] {obs['title']}: {obs['summary']}" for obs in observations
        )

        response = await self.compressor.client.messages.create(
            model=self.compressor.model,
            max_tokens=512,
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(observations=obs_text)}],
            extra_body={
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": SUMMARY_SCHEMA,
                    }
                }
            },
        )

        text_block = response.content[0]
        assert hasattr(text_block, "text"), f"Unexpected content block type: {type(text_block)}"
        data = json.loads(text_block.text)
        summary = SessionSummary(**data)

        await self._store_summary(session_id, summary)
        return summary

    async def _get_session_observations(self, session_id: str) -> list[dict]:
        """Get observations for a session as dicts."""
        cursor = await self.db.execute(
            "SELECT tool_name, title, summary FROM observations "
            "WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def _store_summary(self, session_id: str, summary: SessionSummary) -> None:
        """Store summary in sessions table and close the session."""
        await self.db.execute(
            "UPDATE sessions SET summary = ?, status = 'closed' WHERE id = ?",
            (summary.summary, session_id),
        )
        await self.db.commit()
