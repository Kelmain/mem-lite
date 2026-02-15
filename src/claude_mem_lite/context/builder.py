"""Progressive disclosure context builder for SessionStart injection."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from claude_mem_lite.context.estimator import estimate_tokens

if TYPE_CHECKING:
    import aiosqlite


LAYER_CAPS: dict[str, int] = {
    "session_index": 400,
    "function_map": 500,
    "learnings": 300,
    "observations": 600,
    "reserve": 200,
}


@dataclass
class ContextLayer:
    """A single layer of context with its content and metadata."""

    name: str
    text: str
    tokens: int
    priority: int


@dataclass
class ContextResult:
    """Final assembled context with metadata for logging."""

    text: str
    total_tokens: int
    budget: int
    layers_included: list[str]
    layers_skipped: list[str]
    build_time_ms: float


def _relative_time(db_timestamp: str) -> str:
    """Convert DB timestamp to human-readable relative time.

    Handles naive timestamps from SQLite by assuming UTC (Amendment 4).
    """
    now = datetime.now(UTC)
    dt = datetime.fromisoformat(db_timestamp)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = now - dt

    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    if seconds < 172800:
        return "yesterday"
    return f"{seconds // 86400}d ago"


def _truncate(text: str, max_chars: int = 200) -> str:
    """Truncate text to max_chars with ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _make_relative(file_path: str, project_dir: str) -> str:
    """Make a file path relative to project_dir."""
    if project_dir and file_path.startswith(project_dir):
        rel = file_path[len(project_dir) :]
        return rel.lstrip("/")
    return file_path


class ContextBuilder:
    """Progressive disclosure context builder.

    Assembles context within a token budget from multiple data sources.
    Layers are added in priority order until the budget is exhausted.
    """

    DEFAULT_BUDGET = 2000

    def __init__(
        self,
        db: aiosqlite.Connection,
        lance_store: object | None = None,
        budget: int = DEFAULT_BUDGET,
        project_dir: str | None = None,
    ) -> None:
        self.db = db
        self.lance_store = lance_store
        self.budget = budget
        self.project_dir = project_dir

    async def build(self) -> ContextResult:
        """Build context within token budget."""
        start = time.monotonic()

        layers = await asyncio.gather(
            self._build_session_index(),
            self._build_function_map(),
            self._build_learnings(),
            self._build_observations(),
            return_exceptions=True,
        )

        valid_layers: list[ContextLayer] = [
            layer for layer in layers if isinstance(layer, ContextLayer)
        ]
        valid_layers.sort(key=lambda layer: layer.priority)

        included: list[ContextLayer] = []
        skipped: list[str] = []
        remaining = self.budget - LAYER_CAPS["reserve"]

        for layer in valid_layers:
            if layer.tokens <= remaining:
                included.append(layer)
                remaining -= layer.tokens
            else:
                skipped.append(layer.name)

        parts = [layer.text for layer in included]
        if parts:
            parts.append(
                "\n---\n"
                "Use `curl --unix-socket ~/.claude-mem/worker.sock "
                "http://localhost/api/search?q=...` for deeper context."
            )

        text = "\n\n".join(parts) if parts else ""
        total_tokens = estimate_tokens(text) if text else 0
        elapsed_ms = (time.monotonic() - start) * 1000

        return ContextResult(
            text=text,
            total_tokens=total_tokens,
            budget=self.budget,
            layers_included=[layer.name for layer in included],
            layers_skipped=skipped,
            build_time_ms=elapsed_ms,
        )

    async def _build_session_index(self) -> ContextLayer | None:
        """Recent session summaries, most recent first."""
        query = "SELECT id, started_at, summary FROM sessions WHERE summary IS NOT NULL"
        params: list[str] = []
        if self.project_dir:
            query += " AND project_dir = ?"
            params.append(self.project_dir)
        query += " ORDER BY started_at DESC LIMIT 10"

        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        if not rows:
            return None

        lines = ["## Recent Sessions"]
        for row in rows:
            age = _relative_time(row["started_at"])
            summary = _truncate(row["summary"], max_chars=200)
            lines.append(f"- [{age}] {summary}")

        text = "\n".join(lines)
        tokens = estimate_tokens(text)

        while tokens > LAYER_CAPS["session_index"] and len(lines) > 2:
            lines.pop()
            text = "\n".join(lines)
            tokens = estimate_tokens(text)

        return ContextLayer(name="session_index", text=text, tokens=tokens, priority=1)

    async def _build_function_map(self) -> ContextLayer | None:
        """Recently changed functions grouped by file."""
        query = (
            "SELECT fm.file_path, fm.qualified_name, fm.signature, fm.change_type"
            " FROM function_map fm"
            " JOIN sessions s ON fm.session_id = s.id"
            " WHERE fm.change_type IN ('new', 'modified', 'deleted')"
        )
        params: list[str] = []
        if self.project_dir:
            query += " AND s.project_dir = ?"
            params.append(self.project_dir)
        query += " ORDER BY s.started_at DESC, fm.file_path LIMIT 30"

        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        if not rows:
            return None

        seen: set[tuple[str, str]] = set()
        by_file: dict[str, list] = {}
        for row in rows:
            key = (row["file_path"], row["qualified_name"])
            if key in seen:
                continue
            seen.add(key)
            by_file.setdefault(row["file_path"], []).append(row)

        project_dir = self.project_dir or ""
        lines = ["## Recently Changed Code"]
        for file_path, funcs in by_file.items():
            rel_path = _make_relative(file_path, project_dir) if project_dir else file_path
            lines.append(f"{rel_path}:")
            for f in funcs:
                tag = f["change_type"].upper()
                lines.append(f"  {f['signature']}  [{tag}]")

        text = "\n".join(lines)
        tokens = estimate_tokens(text)

        return ContextLayer(name="function_map", text=text, tokens=tokens, priority=2)

    async def _build_learnings(self) -> ContextLayer | None:
        """Active project learnings, highest confidence first."""
        cursor = await self.db.execute(
            "SELECT category, content, confidence FROM learnings"
            " WHERE is_active = 1 AND confidence >= 0.5"
            " ORDER BY confidence DESC LIMIT 10",
        )
        rows = await cursor.fetchall()
        if not rows:
            return None

        lines = ["## Project Knowledge"]
        for row in rows:
            cat = row["category"].capitalize()
            lines.append(f"- {cat}: {row['content']}")

        text = "\n".join(lines)
        tokens = estimate_tokens(text)

        while tokens > LAYER_CAPS["learnings"] and len(lines) > 2:
            lines.pop()
            text = "\n".join(lines)
            tokens = estimate_tokens(text)

        return ContextLayer(name="learnings", text=text, tokens=tokens, priority=3)

    async def _build_observations(self) -> ContextLayer | None:
        """Recent observations by recency.

        At SessionStart, there is no user input to infer intent.
        Semantic search is deferred to UserPromptSubmit (future phase)
        where actual user intent is available as a search query.
        """
        subquery = "SELECT id FROM sessions"
        params: list[str] = []
        if self.project_dir:
            subquery += " WHERE project_dir = ?"
            params.append(self.project_dir)
        subquery += " ORDER BY started_at DESC LIMIT 5"

        query = (
            "SELECT title, summary, created_at FROM observations"
            f" WHERE session_id IN ({subquery})"
            " ORDER BY created_at DESC LIMIT 10"
        )

        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        if not rows:
            return None

        lines = ["## Recent Observations"]
        for row in rows:
            lines.append(f"- {row['title']}: {row['summary']}")

        text = "\n".join(lines)
        tokens = estimate_tokens(text)

        while tokens > LAYER_CAPS["observations"] and len(lines) > 2:
            lines.pop()
            text = "\n".join(lines)
            tokens = estimate_tokens(text)

        return ContextLayer(name="observations", text=text, tokens=tokens, priority=4)
