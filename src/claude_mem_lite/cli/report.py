"""CLI report command: build and render activity reports."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


# -----------------------------------------------------------------------
# Data classes for report structure
# -----------------------------------------------------------------------


@dataclass
class SessionRow:
    """A session summary row."""

    id: str
    project_dir: str
    started_at: str
    ended_at: str | None
    status: str
    observation_count: int


@dataclass
class FunctionChangeRow:
    """A single function change record."""

    file_path: str
    qualified_name: str
    change_type: str


@dataclass
class FunctionChangeSummary:
    """Aggregated function change counts."""

    new: int = 0
    modified: int = 0
    deleted: int = 0


@dataclass
class SessionDetail:
    """Detailed info for a single session."""

    session_id: str
    summary: str | None
    observations: list[dict] = field(default_factory=list)


@dataclass
class LearningRow:
    """A learning record for reports."""

    category: str
    content: str
    confidence: float
    times_seen: int


@dataclass
class CompressionStats:
    """Aggregate compression statistics."""

    total_observations: int = 0
    total_tokens_raw: int = 0
    total_tokens_compressed: int = 0
    avg_ratio: float = 0.0


@dataclass
class ReportData:
    """Complete report data container."""

    sessions: list[SessionRow] = field(default_factory=list)
    observation_count: int = 0
    function_changes: FunctionChangeSummary = field(default_factory=FunctionChangeSummary)
    function_change_rows: list[FunctionChangeRow] = field(default_factory=list)
    session_detail: SessionDetail | None = None
    learnings: list[LearningRow] = field(default_factory=list)
    compression: CompressionStats = field(default_factory=CompressionStats)
    period_days: int = 1


# -----------------------------------------------------------------------
# ReportBuilder: sync sqlite3 queries
# -----------------------------------------------------------------------


class ReportBuilder:
    """Build report data from sync SQLite connection."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def build(
        self,
        *,
        days: int = 1,
        session_id: str | None = None,
        learnings_limit: int = 5,
    ) -> ReportData:
        """Query database and assemble report data.

        Args:
            days: Number of days to include in the report.
            session_id: Optional specific session to scope to.
            learnings_limit: Max number of top learnings to include.

        Returns:
            ReportData with all queried information.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            data = ReportData(period_days=days)
            data.sessions = self._query_sessions(conn, days=days, session_id=session_id)
            data.observation_count = self._query_observation_count(
                conn, days=days, session_id=session_id
            )
            data.function_changes = self._query_function_changes(
                conn, days=days, session_id=session_id
            )
            data.function_change_rows = self._query_function_change_rows(
                conn, days=days, session_id=session_id
            )
            if session_id:
                data.session_detail = self._query_session_detail(conn, session_id)
            data.learnings = self._query_learnings(conn, limit=learnings_limit)
            data.compression = self._query_compression_stats(conn, days=days, session_id=session_id)
            return data
        finally:
            conn.close()

    def _query_sessions(
        self,
        conn: sqlite3.Connection,
        *,
        days: int,
        session_id: str | None,
    ) -> list[SessionRow]:
        """Query sessions within the time period."""
        if session_id:
            rows = conn.execute(
                "SELECT id, project_dir, started_at, ended_at, status, observation_count "
                "FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, project_dir, started_at, ended_at, status, observation_count "
                "FROM sessions WHERE started_at >= datetime('now', ?)"
                " ORDER BY started_at DESC",
                (f"-{days} days",),
            ).fetchall()
        return [
            SessionRow(
                id=r["id"],
                project_dir=r["project_dir"],
                started_at=r["started_at"],
                ended_at=r["ended_at"],
                status=r["status"],
                observation_count=r["observation_count"],
            )
            for r in rows
        ]

    def _query_observation_count(
        self,
        conn: sqlite3.Connection,
        *,
        days: int,
        session_id: str | None,
    ) -> int:
        """Count observations in the time period."""
        if session_id:
            row = conn.execute(
                "SELECT COUNT(*) FROM observations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM observations WHERE created_at >= datetime('now', ?)",
                (f"-{days} days",),
            ).fetchone()
        return int(row[0]) if row else 0

    def _query_function_changes(
        self,
        conn: sqlite3.Connection,
        *,
        days: int,
        session_id: str | None,
    ) -> FunctionChangeSummary:
        """Aggregate function change counts."""
        if session_id:
            rows = conn.execute(
                "SELECT change_type, COUNT(*) as cnt FROM function_map "
                "WHERE session_id = ? GROUP BY change_type",
                (session_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT change_type, COUNT(*) as cnt FROM function_map "
                "WHERE updated_at >= datetime('now', ?) GROUP BY change_type",
                (f"-{days} days",),
            ).fetchall()

        summary = FunctionChangeSummary()
        for r in rows:
            ct = r["change_type"]
            count = r["cnt"]
            if ct == "new":
                summary.new = count
            elif ct == "modified":
                summary.modified = count
            elif ct == "deleted":
                summary.deleted = count
        return summary

    def _query_function_change_rows(
        self,
        conn: sqlite3.Connection,
        *,
        days: int,
        session_id: str | None,
    ) -> list[FunctionChangeRow]:
        """Query individual function change rows."""
        if session_id:
            rows = conn.execute(
                "SELECT file_path, qualified_name, change_type FROM function_map "
                "WHERE session_id = ? ORDER BY file_path, qualified_name",
                (session_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT file_path, qualified_name, change_type FROM function_map "
                "WHERE updated_at >= datetime('now', ?) ORDER BY file_path, qualified_name",
                (f"-{days} days",),
            ).fetchall()
        return [
            FunctionChangeRow(
                file_path=r["file_path"],
                qualified_name=r["qualified_name"],
                change_type=r["change_type"],
            )
            for r in rows
        ]

    def _query_session_detail(
        self,
        conn: sqlite3.Connection,
        session_id: str,
    ) -> SessionDetail:
        """Query detailed session info with observations."""
        session = conn.execute(
            "SELECT summary FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()

        obs_rows = conn.execute(
            "SELECT id, title, summary, tool_name, created_at FROM observations "
            "WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()

        return SessionDetail(
            session_id=session_id,
            summary=session["summary"] if session else None,
            observations=[
                {
                    "id": r["id"],
                    "title": r["title"],
                    "summary": r["summary"],
                    "tool_name": r["tool_name"],
                    "created_at": r["created_at"],
                }
                for r in obs_rows
            ],
        )

    def _query_learnings(
        self,
        conn: sqlite3.Connection,
        *,
        limit: int,
    ) -> list[LearningRow]:
        """Query top learnings by confidence."""
        rows = conn.execute(
            "SELECT category, content, confidence, times_seen "
            "FROM learnings WHERE is_active = 1 "
            "ORDER BY confidence DESC, times_seen DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            LearningRow(
                category=r["category"],
                content=r["content"],
                confidence=r["confidence"],
                times_seen=r["times_seen"],
            )
            for r in rows
        ]

    def _query_compression_stats(
        self,
        conn: sqlite3.Connection,
        *,
        days: int,
        session_id: str | None,
    ) -> CompressionStats:
        """Aggregate compression statistics."""
        if session_id:
            row = conn.execute(
                "SELECT COUNT(*) as cnt, "
                "COALESCE(SUM(tokens_raw), 0) as total_raw, "
                "COALESCE(SUM(tokens_compressed), 0) as total_compressed "
                "FROM observations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) as cnt, "
                "COALESCE(SUM(tokens_raw), 0) as total_raw, "
                "COALESCE(SUM(tokens_compressed), 0) as total_compressed "
                "FROM observations WHERE created_at >= datetime('now', ?)",
                (f"-{days} days",),
            ).fetchone()

        if not row or row["cnt"] == 0:
            return CompressionStats()

        total_raw = row["total_raw"]
        total_compressed = row["total_compressed"]
        avg_ratio = (total_raw / total_compressed) if total_compressed > 0 else 0.0

        return CompressionStats(
            total_observations=row["cnt"],
            total_tokens_raw=total_raw,
            total_tokens_compressed=total_compressed,
            avg_ratio=round(avg_ratio, 1),
        )


# -----------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------


def render_report(data: ReportData) -> None:
    """Render report data using Rich panels and tables."""
    # Header
    console.print(
        Panel(
            Text(f"Activity Report ({data.period_days}d)", style="bold cyan"),
            border_style="cyan",
        )
    )

    # Sessions overview
    if data.sessions:
        table = Table(title="Sessions", show_lines=True)
        table.add_column("ID", style="dim", max_width=12)
        table.add_column("Project", max_width=30)
        table.add_column("Status")
        table.add_column("Observations", justify="right")
        table.add_column("Started")

        for s in data.sessions:
            status_style = "green" if s.status == "active" else "dim"
            table.add_row(
                s.id[:12],
                s.project_dir,
                Text(s.status, style=status_style),
                str(s.observation_count),
                s.started_at,
            )
        console.print(table)
    else:
        console.print("[dim]No sessions found in period.[/dim]")

    # Observation count
    console.print(f"\nTotal observations: [bold]{data.observation_count}[/bold]")

    # Function changes
    fc = data.function_changes
    if fc.new or fc.modified or fc.deleted:
        console.print(
            f"\nFunction changes: "
            f"[green]+{fc.new} new[/green] | "
            f"[yellow]~{fc.modified} modified[/yellow] | "
            f"[red]-{fc.deleted} deleted[/red]"
        )

    # Session detail
    if data.session_detail:
        detail = data.session_detail
        if detail.summary:
            console.print(Panel(detail.summary, title="Session Summary", border_style="blue"))
        if detail.observations:
            obs_table = Table(title="Observations")
            obs_table.add_column("Title", max_width=40)
            obs_table.add_column("Tool")
            obs_table.add_column("Time")
            for obs in detail.observations:
                obs_table.add_row(obs["title"], obs["tool_name"], obs["created_at"])
            console.print(obs_table)

    # Learnings
    if data.learnings:
        learn_table = Table(title="Top Learnings")
        learn_table.add_column("Category", style="cyan")
        learn_table.add_column("Content", max_width=60)
        learn_table.add_column("Confidence", justify="right")
        for lr in data.learnings:
            learn_table.add_row(lr.category, lr.content, f"{lr.confidence:.2f}")
        console.print(learn_table)

    # Compression stats
    cs = data.compression
    if cs.total_observations > 0:
        console.print(
            f"\nCompression: {cs.total_observations} obs, "
            f"{cs.total_tokens_raw:,} raw -> {cs.total_tokens_compressed:,} compressed "
            f"({cs.avg_ratio:.1f}x ratio)"
        )


def render_markdown(data: ReportData) -> str:
    """Render report data as markdown string."""
    lines: list[str] = []
    lines.append(f"# Activity Report ({data.period_days}d)")
    lines.append("")

    # Sessions
    lines.append("## Sessions")
    if data.sessions:
        lines.append("| ID | Project | Status | Observations |")
        lines.append("|---|---|---|---|")
        for s in data.sessions:
            lines.append(f"| {s.id[:12]} | {s.project_dir} | {s.status} | {s.observation_count} |")
    else:
        lines.append("No sessions found.")
    lines.append("")

    lines.append(f"**Total observations**: {data.observation_count}")
    lines.append("")

    # Function changes
    fc = data.function_changes
    if fc.new or fc.modified or fc.deleted:
        lines.append("## Function Changes")
        lines.append(f"- New: {fc.new}")
        lines.append(f"- Modified: {fc.modified}")
        lines.append(f"- Deleted: {fc.deleted}")
        lines.append("")

    # Learnings
    if data.learnings:
        lines.append("## Top Learnings")
        for lr in data.learnings:
            lines.append(f"- **{lr.category}** ({lr.confidence:.2f}): {lr.content}")
        lines.append("")

    # Compression
    cs = data.compression
    if cs.total_observations > 0:
        lines.append("## Compression")
        lines.append(f"- Observations: {cs.total_observations}")
        lines.append(f"- Tokens: {cs.total_tokens_raw:,} raw -> {cs.total_tokens_compressed:,}")
        lines.append(f"- Ratio: {cs.avg_ratio:.1f}x")
        lines.append("")

    return "\n".join(lines)


# -----------------------------------------------------------------------
# CLI command
# -----------------------------------------------------------------------


def report_cmd(
    days: Annotated[int, typer.Option(help="Number of days to include.")] = 1,
    session: Annotated[str | None, typer.Option(help="Specific session ID.")] = None,
    md: Annotated[bool, typer.Option("--md", help="Output as markdown.")] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
    learnings: Annotated[int, typer.Option(help="Number of top learnings.")] = 5,
) -> None:
    """Generate an activity report from the local memory database."""
    from claude_mem_lite.config import Config

    config = Config()
    db_path = str(config.db_path)

    if not Path(db_path).exists():
        console.print("[red]Database not found.[/red] Run claude-mem first to create it.")
        raise typer.Exit(code=1)

    builder = ReportBuilder(db_path)
    data = builder.build(days=days, session_id=session, learnings_limit=learnings)

    if output_json:
        import dataclasses

        console.print(json.dumps(dataclasses.asdict(data), indent=2, default=str))
    elif md:
        console.print(render_markdown(data))
    else:
        render_report(data)
