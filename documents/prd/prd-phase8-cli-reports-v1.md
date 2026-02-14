# Phase 8 PRD: CLI Reports + Search

**Version**: 1.0
**Date**: 2026-02-08
**Depends on**: Phase 0 (storage), Phase 2 (AST tracker), Phase 4 (search), Phase 6 (learnings), Phase 7 (eval)
**Estimated effort**: ~1 session

---

## 1. Scope

### 1.1 What this phase builds

The **presentation layer** â€” CLI commands that surface data from all previous phases into a terminal-friendly format. This is the human-facing UI for claude-mem-lite.

Components:
- `cli/main.py` â€” Root CLI group, entry point registration
- `cli/report.py` â€” `claude-mem report` (session report, function changes, call graph, learnings)
- `cli/search_cmd.py` â€” `claude-mem search <query>` (hybrid search wrapper)
- `cli/mermaid_cmd.py` â€” `claude-mem mermaid` (call graph export)
- Integration with Phase 7's `cli/eval_cmd.py`

### 1.2 What this phase does NOT build

- **Web UI** â€” the architecture doc explicitly rejects this ("No Web UI... ~30% of the codebase for a feature most devs never open")
- **`report --eval`** â€” this was in the implementation plan but is fully covered by Phase 7's `claude-mem eval health`. No duplication.
- **Worker management CLI** â€” `claude-mem-worker start/stop` is part of Phase 3 (worker lifecycle). Phase 8 builds read-only reporting commands.
- **Interactive TUI** â€” no `textual` app. Static terminal output via `rich`.

### 1.3 Corrections to implementation plan

| Item | Implementation Plan | Corrected (Phase 8 PRD) | Rationale |
|------|---------------------|------------------------|-----------|
| `report --eval` | Separate eval dashboard inside report command | **Removed.** `claude-mem eval health` (Phase 7) already covers this identically. | Architecture doc's `report --eval` mockup matches Phase 7's `eval health` output 1:1. Don't build the same thing twice. |
| `report --mermaid` | Mermaid subcommand of report | **Separate command: `claude-mem mermaid`** | Mermaid has its own options (file filter, format, show-all). Nesting under report makes the interface awkward: `claude-mem report --mermaid --file auth/service.py --all`. Cleaner as `claude-mem mermaid auth/service.py --all`. |
| Search CLI | "CLI wrapper around hybrid search" â€” no detail on worker-down behavior | **Direct SQLite FTS fallback when worker unavailable** | Search should always work. If worker is down, fall back to `FTS5` search directly against SQLite. Degrade gracefully, don't error. |
| Entry point | Not specified | `pyproject.toml` `[project.scripts]` registration | Without this, there's no `claude-mem` command. |
| Markdown export | "optional markdown file" | `--md` flag on report command, writes to `~/.claude-mem/reports/` | Needs an explicit spec for path, naming, and format. |

---

## 2. Dependencies

### 2.1 Python packages

| Package | Version | Purpose | Already in project? |
|---------|---------|---------|---------------------|
| `click` | â‰¥8.1 | CLI argument parsing, command groups | No â€” **new dependency** |
| `rich` | â‰¥14.0 | Terminal formatting, tables, panels | No â€” **new dependency** |

**Why `click` over `typer`**: For a 4-command CLI tool, click is lighter (zero transitive deps beyond the stdlib), more battle-tested (17K stars, 2.3M packages depend on it), and doesn't bring typer's autocompletion machinery we don't need. Both are actively maintained (click: Dec 2025, typer: Jan 2026). If the CLI grows significantly in Phase 9+, migration to typer is trivial since typer wraps click.

**Why not `argparse`**: Click's command groups (`@cli.group()`) make subcommand routing clean. Argparse subparsers work but require more boilerplate, especially for shared options and help formatting.

### 2.2 Internal dependencies

| Module | Phase | What we read |
|--------|-------|-------------|
| `storage/sqlite_store.py` | Phase 0 | Sessions, observations, function_map, call_graph, learnings, event_log |
| `ast_tracker/mermaid.py` | Phase 2 | `generate_mermaid()` |
| `search/hybrid.py` | Phase 4 | Search via worker API or direct FTS fallback |
| `eval/evaluator.py` | Phase 7 | Referenced but not imported â€” eval has its own CLI |
| `logging/logger.py` | All | MemLogger for event logging |
| `config.py` | Phase 0 | Config paths, database location |

---

## 3. CLI Entry Point

### 3.1 Registration (`pyproject.toml`)

```toml
[project.scripts]
claude-mem = "claude_mem_lite.cli.main:cli"
```

### 3.2 Root CLI group (`cli/main.py`)

```python
import click
from claude_mem_lite.cli.report import report
from claude_mem_lite.cli.search_cmd import search
from claude_mem_lite.cli.mermaid_cmd import mermaid
from claude_mem_lite.cli.eval_cmd import eval_group

@click.group()
@click.version_option(package_name="claude-mem-lite")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """claude-mem-lite: Local memory for Claude Code sessions."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = _load_config()
    ctx.obj["db_path"] = ctx.obj["config"].db_path

cli.add_command(report)
cli.add_command(search)
cli.add_command(mermaid)
cli.add_command(eval_group, name="eval")
```

**Resulting command tree:**

```
claude-mem
â”œâ”€â”€ report          Session report (default: today)
â”‚   â””â”€â”€ --md        Export as markdown
â”‚   â””â”€â”€ --session   Specific session ID
â”‚   â””â”€â”€ --days N    Report for last N days
â”‚   â””â”€â”€ --json      JSON output
â”œâ”€â”€ search <query>  Hybrid search over observations
â”‚   â””â”€â”€ --limit N   Max results
â”‚   â””â”€â”€ --type T    observation | code | learning
â”‚   â””â”€â”€ --json      JSON output
â”œâ”€â”€ mermaid         Call graph as Mermaid diagram
â”‚   â””â”€â”€ <file>      Specific file (optional)
â”‚   â””â”€â”€ --all       Show all functions, not just changed
â”‚   â””â”€â”€ --output F  Write to file instead of stdout
â”‚   â””â”€â”€ --session   Session scope
â”œâ”€â”€ eval            (Phase 7 â€” already specified)
â”‚   â”œâ”€â”€ compression
â”‚   â”œâ”€â”€ benchmark
â”‚   â””â”€â”€ health
```

---

## 4. `claude-mem report`

### 4.1 Purpose

The primary "what happened?" command. Shows a summary of recent sessions, observations, function changes, and active learnings. Matches the architecture doc's mockup.

### 4.2 Data queries

All queries run directly against SQLite (no worker needed). This is a read-only operation.

```python
@dataclass
class ReportData:
    """All data needed to render a report."""
    period_start: str           # ISO date
    period_end: str             # ISO date
    sessions: list[SessionRow]
    observations_count: int
    function_changes: FunctionChangeSummary
    latest_session: SessionDetail | None
    top_learnings: list[LearningRow]
    compression_stats: CompressionStats | None

@dataclass
class SessionRow:
    id: str
    started_at: str
    ended_at: str | None
    status: str
    summary_text: str | None
    observation_count: int

@dataclass
class FunctionChangeSummary:
    new: int
    modified: int
    deleted: int
    changes: list[FunctionChangeRow]  # individual changes for detail view

@dataclass
class FunctionChangeRow:
    file_path: str
    qualified_name: str
    signature: str
    change_type: str  # new | modified | deleted
    session_id: str

@dataclass
class SessionDetail:
    """Latest session expanded view."""
    id: str
    summary_text: str
    files_touched: list[str]       # unique files across all observations
    observation_titles: list[str]  # compressed observation titles

@dataclass
class LearningRow:
    category: str
    content: str
    confidence: float
    times_seen: int

@dataclass
class CompressionStats:
    total_observations: int
    avg_ratio: float
    avg_latency_ms: float
    total_cost_usd: float
```

### 4.3 SQL queries

```python
QUERIES = {
    "sessions_in_period": """
        SELECT s.id, s.started_at, s.ended_at, s.status, s.summary_text,
               COUNT(o.id) as observation_count
        FROM sessions s
        LEFT JOIN observations o ON o.session_id = s.id
        WHERE s.started_at >= ?
        GROUP BY s.id
        ORDER BY s.started_at DESC
    """,

    "function_changes_in_period": """
        SELECT fm.file_path, fm.qualified_name, fm.signature, fm.change_type,
               fm.session_id
        FROM function_map fm
        JOIN sessions s ON fm.session_id = s.id
        WHERE s.started_at >= ? AND fm.change_type != 'unchanged'
        ORDER BY fm.file_path, fm.change_type
    """,

    "latest_session_files": """
        SELECT DISTINCT
            json_each.value as file_path
        FROM observations o,
             json_each(o.files_touched)
        WHERE o.session_id = ?
    """,

    "latest_session_titles": """
        SELECT title FROM observations
        WHERE session_id = ?
        ORDER BY created_at
    """,

    "top_learnings": """
        SELECT category, content, confidence, times_seen
        FROM learnings
        WHERE is_active = 1
        ORDER BY confidence DESC
        LIMIT ?
    """,

    "compression_stats": """
        SELECT COUNT(*) as total,
               AVG(json_extract(data, '$.ratio')) as avg_ratio,
               AVG(duration_ms) as avg_latency,
               SUM(
                   COALESCE(tokens_in, 0) * 1.0 / 1000000
                   + COALESCE(tokens_out, 0) * 5.0 / 1000000
               ) as total_cost
        FROM event_log
        WHERE event_type = 'compress.done'
          AND timestamp >= ?
    """,
}
```

### 4.4 Report builder

```python
import sqlite3
from datetime import datetime, timedelta

class ReportBuilder:
    """Build report data from SQLite queries."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def build(
        self,
        days: int = 1,
        session_id: str | None = None,
    ) -> ReportData:
        """Build report data for the given period.

        If session_id is provided, report scopes to that single session.
        Otherwise, reports on last `days` days.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            if session_id:
                return self._build_session_report(conn, session_id)
            else:
                return self._build_period_report(conn, days)
        finally:
            conn.close()

    def _build_period_report(self, conn: sqlite3.Connection, days: int) -> ReportData:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        sessions = self._query_sessions(conn, cutoff)
        function_changes = self._query_function_changes(conn, cutoff)
        latest = self._query_latest_session(conn, sessions)
        learnings = self._query_learnings(conn, limit=5)
        compression = self._query_compression_stats(conn, cutoff)

        obs_count = sum(s.observation_count for s in sessions)

        return ReportData(
            period_start=cutoff,
            period_end=datetime.utcnow().isoformat(),
            sessions=sessions,
            observations_count=obs_count,
            function_changes=function_changes,
            latest_session=latest,
            top_learnings=learnings,
            compression_stats=compression,
        )
    # ... individual query methods map to QUERIES dict above
```

### 4.5 Terminal rendering (`rich`)

```python
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

def render_report(data: ReportData) -> None:
    """Render report to terminal using rich."""

    # Header panel
    header_lines = [
        f"Sessions: {len(data.sessions)}",
        f"Observations captured: {data.observations_count}",
    ]
    fc = data.function_changes
    header_lines.append(
        f"Functions changed: {fc.new} new, {fc.modified} modified, {fc.deleted} deleted"
    )
    if data.compression_stats and data.compression_stats.total_observations > 0:
        cs = data.compression_stats
        header_lines.append(
            f"Compression ratio: {cs.avg_ratio:.0f}:1 avg ({cs.avg_latency_ms:.0f}ms avg)"
        )

    period_label = _format_period(data.period_start, data.period_end)
    console.print(Panel(
        "\n".join(header_lines),
        title=f"Session Report: {period_label}",
        border_style="blue",
    ))

    # Latest session
    if data.latest_session:
        latest = data.latest_session
        lines = []
        if latest.summary_text:
            lines.append(latest.summary_text)
        if latest.files_touched:
            files_str = ", ".join(latest.files_touched[:8])
            if len(latest.files_touched) > 8:
                files_str += f" (+{len(latest.files_touched) - 8} more)"
            lines.append(f"Files: {files_str}")
        if lines:
            console.print(Panel(
                "\n".join(lines),
                title="Latest Session",
                border_style="cyan",
            ))

    # Function changes table
    if fc.changes:
        table = Table(title="Function Changes", show_lines=False)
        table.add_column("Action", style="bold", width=6)
        table.add_column("Function", no_wrap=True)
        table.add_column("File", style="dim")

        style_map = {"new": "green", "modified": "yellow", "deleted": "red"}

        for change in fc.changes[:20]:  # Cap at 20 rows
            action = change.change_type.upper()
            style = style_map.get(change.change_type, "white")
            sig = change.signature or change.qualified_name
            table.add_row(
                Text(action, style=style),
                sig,
                change.file_path,
            )

        if len(fc.changes) > 20:
            table.add_row("...", f"+{len(fc.changes) - 20} more", "")

        console.print(table)

    # Learnings
    if data.top_learnings:
        lines = []
        for l in data.top_learnings:
            conf_style = "green" if l.confidence >= 0.8 else "yellow" if l.confidence >= 0.5 else "dim"
            lines.append(f"[{conf_style}][{l.confidence:.2f}][/{conf_style}] "
                         f"[bold]{l.category}[/bold]: {l.content}")
        console.print(Panel(
            "\n".join(lines),
            title=f"Active Learnings (top {len(data.top_learnings)})",
            border_style="green",
        ))

    # Cost footer (if available)
    if data.compression_stats and data.compression_stats.total_cost_usd > 0:
        console.print(
            f"\n[dim]API cost ({period_label}): "
            f"${data.compression_stats.total_cost_usd:.4f}[/dim]"
        )
```

### 4.6 Markdown export

```python
def render_markdown(data: ReportData) -> str:
    """Render report as markdown string."""
    lines = [f"# Session Report: {_format_period(data.period_start, data.period_end)}", ""]

    lines.append(f"- Sessions: {len(data.sessions)}")
    lines.append(f"- Observations: {data.observations_count}")
    fc = data.function_changes
    lines.append(f"- Functions changed: {fc.new} new, {fc.modified} modified, {fc.deleted} deleted")
    if data.compression_stats and data.compression_stats.total_observations > 0:
        cs = data.compression_stats
        lines.append(f"- Compression: {cs.avg_ratio:.0f}:1 avg, {cs.avg_latency_ms:.0f}ms avg")
    lines.append("")

    if data.latest_session:
        lines.append("## Latest Session")
        if data.latest_session.summary_text:
            lines.append(data.latest_session.summary_text)
        if data.latest_session.files_touched:
            lines.append(f"\nFiles: {', '.join(data.latest_session.files_touched)}")
        lines.append("")

    if fc.changes:
        lines.append("## Function Changes")
        lines.append("| Action | Function | File |")
        lines.append("|--------|----------|------|")
        for c in fc.changes:
            sig = c.signature or c.qualified_name
            lines.append(f"| {c.change_type.upper()} | `{sig}` | {c.file_path} |")
        lines.append("")

    if data.top_learnings:
        lines.append("## Active Learnings")
        for l in data.top_learnings:
            lines.append(f"- [{l.confidence:.2f}] **{l.category}**: {l.content}")
        lines.append("")

    return "\n".join(lines)
```

### 4.7 Click command

```python
import click
import json
from pathlib import Path
from datetime import datetime

@click.command()
@click.option("--days", default=1, help="Report period in days (default: today)")
@click.option("--session", "session_id", default=None, help="Report for specific session ID")
@click.option("--md", "markdown_path", default=None, is_flag=False, flag_value="auto",
              help="Export as markdown. 'auto' saves to ~/.claude-mem/reports/")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--learnings", "learnings_count", default=5, help="Number of learnings to show")
@click.pass_context
def report(ctx, days, session_id, markdown_path, as_json, learnings_count):
    """Show session report with observations, function changes, and learnings."""
    db_path = ctx.obj["db_path"]

    if not Path(db_path).exists():
        click.echo("No database found. Run a Claude Code session first.", err=True)
        raise SystemExit(1)

    builder = ReportBuilder(db_path)
    data = builder.build(days=days, session_id=session_id)

    if as_json:
        click.echo(json.dumps(_to_dict(data), indent=2, default=str))
        return

    if markdown_path:
        md = render_markdown(data)
        if markdown_path == "auto":
            reports_dir = Path(ctx.obj["config"].data_dir) / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            filename = f"report-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.md"
            markdown_path = reports_dir / filename
        Path(markdown_path).write_text(md)
        click.echo(f"Report saved to {markdown_path}")
        return

    render_report(data)
```

---

## 5. `claude-mem search`

### 5.1 Purpose

Interactive search over observations, learnings, and function history. Wraps the hybrid search from Phase 4 with a graceful fallback when the worker isn't running.

### 5.2 Search strategy

```
User runs: claude-mem search "JWT auth"
    â”‚
    â”œâ”€â”€ Worker running? (check worker.sock / worker.pid)
    â”‚   â”œâ”€â”€ YES â†’ HTTP GET /api/search?q=JWT+auth&limit=5
    â”‚   â”‚         (hybrid: vector + FTS, best quality)
    â”‚   â”‚
    â”‚   â””â”€â”€ NO â†’ Direct SQLite FTS5 fallback
    â”‚            SELECT * FROM observations_fts WHERE observations_fts MATCH ?
    â”‚            (text-only search, no semantic matching)
    â”‚
    â””â”€â”€ Render results with rich
```

### 5.3 Worker discovery

```python
from pathlib import Path
import httpx

def _discover_worker(config) -> str | None:
    """Check if worker is running, return socket path or None."""
    info_path = Path(config.data_dir) / "worker.info"
    if not info_path.exists():
        return None

    info = json.loads(info_path.read_text())
    socket_path = info.get("socket")
    if not socket_path or not Path(socket_path).exists():
        return None

    # Quick health check
    try:
        transport = httpx.HTTPTransport(uds=socket_path)
        with httpx.Client(transport=transport, timeout=2.0) as client:
            resp = client.get("http://localhost/health")
            if resp.status_code == 200:
                return socket_path
    except (httpx.ConnectError, httpx.TimeoutException):
        return None

    return None
```

### 5.4 FTS5 fallback

```python
def _search_fts_fallback(
    db_path: str,
    query: str,
    limit: int = 5,
) -> list[dict]:
    """Direct FTS5 search against SQLite when worker is unavailable.

    This is degraded search: text matching only, no semantic similarity.
    Still useful for exact keyword matches.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # observations table has FTS5 index (Phase 0 schema)
        rows = conn.execute(
            """
            SELECT o.id, o.title, o.summary, o.created_at, o.tool_name,
                   o.files_touched
            FROM observations o
            JOIN observations_fts fts ON fts.rowid = o.rowid
            WHERE observations_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        # FTS table might not exist yet
        return []
    finally:
        conn.close()
```

**Important**: This fallback depends on the `observations_fts` virtual table existing. Phase 0's migration creates it:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts
USING fts5(title, summary, detail, content=observations, content_rowid=rowid);
```

If the FTS table doesn't exist (fresh database, no observations yet), the fallback returns an empty list gracefully.

### 5.5 Result rendering

```python
def _render_results(results: list[dict], query: str, search_type: str) -> None:
    """Render search results with rich."""
    if not results:
        console.print(f"[dim]No results for '{query}'[/dim]")
        return

    badge = "[green]hybrid[/green]" if search_type == "hybrid" else "[yellow]fts[/yellow]"
    console.print(f"Search: '{query}' ({badge}, {len(results)} results)\n")

    for i, r in enumerate(results, 1):
        title = r.get("title", "untitled")
        summary = r.get("summary", "")
        created = r.get("created_at", "")
        tool = r.get("tool_name", "")
        files = r.get("files_touched", "[]")

        # Truncate summary for display
        if len(summary) > 200:
            summary = summary[:200] + "..."

        age = _relative_time(created) if created else ""

        console.print(f"[bold]{i}. {title}[/bold]")
        if age or tool:
            meta_parts = [p for p in [age, tool] if p]
            console.print(f"   [dim]{' Â· '.join(meta_parts)}[/dim]")
        if summary:
            console.print(f"   {summary}")
        if files and files != "[]":
            file_list = json.loads(files) if isinstance(files, str) else files
            if file_list:
                console.print(f"   [dim]Files: {', '.join(file_list[:5])}[/dim]")
        console.print()
```

### 5.6 Click command

```python
@click.command()
@click.argument("query")
@click.option("--limit", default=5, help="Max results (default: 5)")
@click.option("--type", "query_type", default="observation",
              type=click.Choice(["observation", "code", "learning"]),
              help="Search type")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def search(ctx, query, limit, query_type, as_json):
    """Search observations, code, or learnings.

    Examples:
        claude-mem search "JWT auth"
        claude-mem search "database migration" --type code --limit 10
    """
    config = ctx.obj["config"]
    db_path = ctx.obj["db_path"]

    socket = _discover_worker(config)

    if socket:
        results, search_type = _search_via_worker(socket, query, limit, query_type)
    else:
        results = _search_fts_fallback(db_path, query, limit)
        search_type = "fts"

    if as_json:
        click.echo(json.dumps({
            "query": query,
            "results": results,
            "count": len(results),
            "search_type": search_type,
        }, indent=2, default=str))
        return

    _render_results(results, query, search_type)
```

---

## 6. `claude-mem mermaid`

### 6.1 Purpose

Export call graph as Mermaid diagram. Uses Phase 2's `generate_mermaid()` with data from `function_map` and `call_graph` tables.

### 6.2 Data flow

```
User: claude-mem mermaid auth/service.py
    â”‚
    â”œâ”€â”€ Query function_map for file functions
    â”œâ”€â”€ Query call_graph for edges
    â”œâ”€â”€ Query function_map.change_type for styling
    â”‚
    â””â”€â”€ Call Phase 2's generate_mermaid()
        â””â”€â”€ Output to stdout or file
```

### 6.3 Multi-file subgraph support

The architecture doc shows Mermaid with `subgraph` blocks per file. Phase 2's `generate_mermaid()` handles single-file graphs. For the CLI, we need to orchestrate multi-file output.

```python
def build_mermaid_from_db(
    db_path: str,
    file_filter: str | None = None,
    session_id: str | None = None,
    show_all: bool = False,
) -> str:
    """Build Mermaid diagram from database.

    If file_filter is provided, scope to that file.
    If session_id is provided, scope to that session's changes.
    Otherwise, show changes from the most recent session.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Determine session scope
        if not session_id:
            row = conn.execute(
                "SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if not row:
                return "graph TD\n    empty[No sessions found]"
            session_id = row["id"]

        # Get files with changes in this session
        if file_filter:
            files = [file_filter]
        else:
            files = [
                r["file_path"] for r in conn.execute(
                    """SELECT DISTINCT file_path FROM function_map
                       WHERE session_id = ? AND change_type != 'unchanged'""",
                    (session_id,),
                ).fetchall()
            ]

        if not files:
            return "graph TD\n    empty[No function changes in session]"

        # Build per-file data and generate mermaid
        mermaid_parts = ["graph TD"]

        for file_path in files:
            functions = _query_functions(conn, file_path, session_id)
            edges = _query_call_edges(conn, file_path, session_id)
            change_types = {
                f["qualified_name"]: f["change_type"]
                for f in functions
                if f["change_type"] != "unchanged"
            }

            # Use Phase 2's generate_mermaid for per-file subgraph
            from claude_mem_lite.ast_tracker.mermaid import generate_mermaid
            from claude_mem_lite.ast_tracker.extractor import FunctionInfo

            func_infos = [_row_to_function_info(f) for f in functions]
            file_mermaid = generate_mermaid(
                func_infos,
                file_path=file_path,
                show_all=show_all,
                change_types=change_types,
            )

            # Extract the content lines (skip "graph TD" header from sub-call)
            content_lines = [
                line for line in file_mermaid.split("\n")
                if line.strip() and not line.strip().startswith("graph")
            ]
            if content_lines:
                safe_path = file_path.replace("/", "_").replace(".", "_")
                mermaid_parts.append(f"    subgraph {safe_path}[{file_path}]")
                mermaid_parts.extend(f"    {line}" for line in content_lines)
                mermaid_parts.append("    end")

        return "\n".join(mermaid_parts)
    finally:
        conn.close()
```

### 6.4 Click command

```python
@click.command()
@click.argument("file", required=False)
@click.option("--session", "session_id", default=None, help="Session ID to scope to")
@click.option("--all", "show_all", is_flag=True, help="Show all functions, not just changed")
@click.option("--output", "-o", default=None, help="Write to file instead of stdout")
@click.pass_context
def mermaid(ctx, file, session_id, show_all, output):
    """Export call graph as Mermaid diagram.

    Examples:
        claude-mem mermaid                     # Latest session, changed files
        claude-mem mermaid auth/service.py      # Specific file
        claude-mem mermaid --all -o graph.md    # All functions, save to file
    """
    db_path = ctx.obj["db_path"]

    diagram = build_mermaid_from_db(
        db_path=db_path,
        file_filter=file,
        session_id=session_id,
        show_all=show_all,
    )

    if output:
        # Wrap in markdown code block for .md files
        content = diagram
        if output.endswith(".md"):
            content = f"```mermaid\n{diagram}\n```"
        Path(output).write_text(content)
        click.echo(f"Mermaid diagram saved to {output}")
    else:
        click.echo(diagram)
```

---

## 7. Event Logging

All CLI commands log usage to `event_log` via MemLogger for Phase 7 tracking.

| Event Type | When | Data |
|---|---|---|
| `cli.report` | After report generated | `{days, sessions_count, observations_count, format}` |
| `cli.search` | After search executed | `{query, results_count, search_type, query_type}` |
| `cli.mermaid` | After diagram generated | `{file_filter, session_id, show_all, output}` |

**Implementation**: Each command wraps execution in `logger.timed()`:

```python
with logger.timed("cli.report", days=days, format="terminal"):
    data = builder.build(days=days)
    render_report(data)
```

**Note**: CLI commands use synchronous `sqlite3` directly (not `aiosqlite`). The CLI is a short-lived process â€” no event loop needed. MemLogger's `_write_to_sqlite` is the synchronous fallback path.

---

## 8. Edge Cases

### 8.1 Empty database

Every command must handle an empty database gracefully (no sessions, no observations).

```python
# In report:
if not data.sessions:
    console.print("[dim]No sessions found. Run a Claude Code session first.[/dim]")
    return

# In search:
if not results:
    console.print(f"[dim]No results for '{query}'[/dim]")
    return

# In mermaid:
if not files:
    click.echo("graph TD\n    empty[No function changes found]")
    return
```

### 8.2 Missing database file

```python
if not Path(db_path).exists():
    click.echo("No database found at {db_path}. Run a Claude Code session first.", err=True)
    raise SystemExit(1)
```

### 8.3 Corrupt database

SQLite corruption is rare but possible. Wrap all DB access:

```python
try:
    conn = sqlite3.connect(db_path)
    # ... queries
except sqlite3.DatabaseError as e:
    click.echo(f"Database error: {e}. Try: sqlite3 {db_path} 'PRAGMA integrity_check;'", err=True)
    raise SystemExit(1)
```

### 8.4 Worker not running for search

Handled by the FTS fallback (Â§5.4). The user sees a `[yellow]fts[/yellow]` badge indicating degraded search. No error, just lower quality results.

### 8.5 Very large function_map tables

Cap display rows (20 for function changes, 10 for learnings, 5 for search). Use `--json` for full data dump.

### 8.6 Unicode in function signatures

Phase 2's `_sanitize_for_mermaid()` handles bracket substitution. Rich handles Unicode natively. No special handling needed in the CLI layer.

---

## 9. Module Structure

### 9.1 New files

```
src/claude_mem_lite/
    cli/
        __init__.py          # Empty
        main.py              # Root CLI group + config loading
        report.py            # ReportBuilder + render_report + render_markdown
        search_cmd.py        # Search with worker/FTS fallback
        mermaid_cmd.py       # Mermaid diagram generation from DB
```

### 9.2 Modified files

| File | Change | Type |
|------|--------|------|
| `pyproject.toml` | Add `[project.scripts]` entry, add `click` and `rich` to dependencies | Config |
| `cli/eval_cmd.py` (Phase 7) | Register as `eval_group` with click group decorator | Minor wire-up |

**Phase 7's `eval_cmd.py` integration**: Phase 7 specifies eval CLI commands but doesn't specify the click integration (since click is introduced in Phase 8). Phase 8 adds the `@click.group()` wrapper:

```python
# In eval_cmd.py, add:
@click.group(name="eval")
@click.pass_context
def eval_group(ctx):
    """Evaluate compression quality and system health."""
    pass

# Existing Phase 7 commands become subcommands:
@eval_group.command()
def compression(...): ...

@eval_group.command()
def benchmark(...): ...

@eval_group.command()
def health(...): ...
```

---

## 10. Test Plan

### 10.1 Test categories

| Category | Tests | What it validates |
|----------|-------|-------------------|
| **ReportBuilder** | 5 | Empty DB â†’ empty report, single session, multi-day period, session-specific report, function changes aggregation |
| **Report rendering (terminal)** | 3 | Renders without error, handles empty data, function change colors correct |
| **Report rendering (markdown)** | 3 | Valid markdown output, table formatting, handles empty data |
| **Search (worker mode)** | 2 | Worker discovery works, results formatted correctly |
| **Search (FTS fallback)** | 3 | Falls back when worker unavailable, FTS returns results, handles missing FTS table |
| **Mermaid builder** | 4 | Single-file graph, multi-file subgraphs, empty session, `--all` flag |
| **CLI integration** | 4 | `report` runs, `search` runs, `mermaid` runs, `--json` produces valid JSON |
| **Edge cases** | 3 | Missing DB file â†’ error, corrupt DB â†’ error, empty queries â†’ graceful |
| **Total** | **27** | |

### 10.2 Test fixtures

```python
@pytest.fixture
def seeded_db(tmp_path):
    """Database with realistic test data across all tables."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    # Create schema (Phase 0 migrations)
    _run_migrations(conn)
    # Seed sessions
    conn.execute(
        "INSERT INTO sessions (id, project_path, started_at, status, summary_text) "
        "VALUES (?, ?, ?, ?, ?)",
        ("s1", "/project", "2026-02-08T10:00:00", "summarized",
         "Implemented JWT auth with refresh tokens"),
    )
    # Seed observations
    conn.execute(
        "INSERT INTO observations (id, session_id, created_at, hook_type, tool_name, "
        "title, summary, files_touched, functions_changed, tokens_compressed) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("o1", "s1", "2026-02-08T10:05:00", "PostToolUse", "Write",
         "Added JWT auth service", "Created AuthService with authenticate and refresh methods",
         '["auth/service.py"]', '[]', 150),
    )
    # Seed function_map
    conn.execute(
        "INSERT INTO function_map (id, session_id, file_path, qualified_name, "
        "signature, change_type, line_start, line_end, body_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("f1", "s1", "auth/service.py", "AuthService.authenticate",
         "authenticate(email: str, password: str) -> Token",
         "modified", 12, 45, "abc123"),
    )
    # Seed learnings
    conn.execute(
        "INSERT INTO learnings (id, category, content, confidence, times_seen, "
        "is_active, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("l1", "architecture", "FastAPI + SQLAlchemy, Alembic migrations",
         0.95, 3, 1, "2026-02-08T10:00:00", "2026-02-08T10:00:00"),
    )
    # Seed event_log for compression stats
    conn.execute(
        "INSERT INTO event_log (id, timestamp, session_id, event_type, data, duration_ms, "
        "tokens_in, tokens_out) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("e1", "2026-02-08T10:05:00", "s1", "compress.done",
         '{"ratio": 42.0}', 87, 8000, 500),
    )
    conn.commit()
    conn.close()
    return str(db_path)

@pytest.fixture
def cli_runner():
    """Click test runner."""
    from click.testing import CliRunner
    return CliRunner()
```

### 10.3 Key test scenarios

**ReportBuilder**:
- Empty DB â†’ `ReportData` with sessions=[], observations_count=0, empty function_changes
- Seeded DB â†’ correct counts, latest session populated, function changes match
- `days=7` â†’ filters sessions correctly
- `session_id="s1"` â†’ scopes to single session
- Function changes: NEW/MOD/DEL counts sum correctly

**Report rendering**:
- `render_report(empty_data)` â†’ prints "No sessions found", no crash
- `render_report(full_data)` â†’ output contains expected strings (session count, function names)
- `render_markdown(full_data)` â†’ valid markdown, table has correct columns

**Search**:
- Worker available: returns hybrid results with correct format
- Worker unavailable: falls back to FTS, user sees `[fts]` badge
- FTS with missing virtual table: returns empty list, no crash
- `--json` produces valid JSON with `query`, `results`, `count`, `search_type` fields

**Mermaid**:
- Single file: output contains `subgraph` with file path
- Multi-file: output contains multiple `subgraph` blocks
- Empty session: output contains `empty[No function changes]`
- `--all`: output contains unchanged functions too

**CLI integration** (Click CliRunner):
- `cli_runner.invoke(cli, ["report"])` â†’ exit code 0
- `cli_runner.invoke(cli, ["search", "JWT"])` â†’ exit code 0
- `cli_runner.invoke(cli, ["mermaid"])` â†’ exit code 0
- `cli_runner.invoke(cli, ["report", "--json"])` â†’ valid JSON output

---

## 11. Performance Targets

| Operation | Target | Rationale |
|-----------|--------|-----------|
| `report` (today) | <200ms | ~5 SQLite queries, rich render |
| `report` (7 days) | <500ms | More rows to aggregate |
| `report --md` | <300ms | Same queries + file write |
| `search` (worker) | <300ms | HTTP round-trip + render |
| `search` (FTS fallback) | <50ms | Direct SQLite FTS5 |
| `mermaid` (single file) | <100ms | SQLite query + string build |
| `mermaid` (multi-file) | <200ms | Multiple queries + string build |
| CLI startup | <100ms | Import time for click + rich |

All operations are I/O-bound on SQLite reads. With WAL mode and typical data sizes (<10K observations), these targets are conservative.

---

## 12. Acceptance Criteria

Phase 8 is complete when:

- [ ] `claude-mem report` renders session report in terminal with panels and tables
- [ ] `claude-mem report --days 7` scopes to 7-day period
- [ ] `claude-mem report --session <id>` scopes to specific session
- [ ] `claude-mem report --md` exports markdown to `~/.claude-mem/reports/`
- [ ] `claude-mem report --json` outputs valid JSON
- [ ] `claude-mem search <query>` returns results via worker when available
- [ ] `claude-mem search <query>` falls back to FTS when worker is down
- [ ] `claude-mem search --json` outputs valid JSON with search_type field
- [ ] `claude-mem mermaid` outputs valid Mermaid for latest session changes
- [ ] `claude-mem mermaid <file>` scopes to specific file
- [ ] `claude-mem mermaid --all` includes unchanged functions
- [ ] `claude-mem mermaid -o graph.md` writes to file
- [ ] `claude-mem eval` (Phase 7 commands) accessible as subcommand group
- [ ] `pyproject.toml` has `[project.scripts]` entry for `claude-mem`
- [ ] All commands handle empty database gracefully (no crash, helpful message)
- [ ] All commands handle missing database file (error message + exit 1)
- [ ] All 27 tests pass (pytest, <5s total)
- [ ] `ruff check` and `ruff format --check` pass with zero warnings
- [ ] `click` and `rich` added to `pyproject.toml` dependencies

---

## 13. Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **FTS5 table not created yet** | Medium (early usage) | Low | Fallback returns empty list. Phase 0 migration creates FTS triggers, but table is empty until first observation. |
| **`rich` terminal width issues** | Low | Low | Rich auto-detects terminal width. Tables truncate gracefully. |
| **Mermaid output too large** | Medium (big projects) | Low | Default scoping: changed + direct targets only. `--all` is opt-in. Cap at 50 nodes with a warning. |
| **Click import time** | Low | Low | Click is lightweight (<50ms import). Rich is heavier (~80ms) but acceptable for CLI startup. |
| **Phase 7 eval_cmd click integration** | Low | Medium | Phase 7 specifies eval commands but not click decorators. Phase 8 adds the wrapper â€” clean separation. |
| **`observations_fts` schema mismatch** | Low | Medium | FTS5 column list must match Phase 0 schema exactly. Test validates this at fixture creation time. |

---

## 14. Future Extensions (Phase 9+)

- **`claude-mem status`** â€” Quick one-liner: "Worker: running, DB: 312 observations, last session 2h ago"
- **`claude-mem compress --pending`** â€” Manual catch-up compression (architecture doc mentions this)
- **`claude-mem export`** â€” Full data export (SQLite dump or structured JSON)
- **`claude-mem prune --older-than 30d`** â€” Cleanup old observations
- **Rich Live display** â€” Auto-refreshing dashboard for active sessions (if there's demand)
- **Shell completion** â€” Click has built-in completion support via `click.shell_completion`
