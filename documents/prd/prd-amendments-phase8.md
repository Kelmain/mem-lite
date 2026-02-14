# PRD Amendments: Phase 8 (Post-Review)

**Date**: 2026-02-08
**Triggered by**: External review of Phase 8 PRD (5 comments, all actionable)
**Affects**: Phase 8 v1 (CLI Reports + Search)

---

## Amendment 1: Create `observations_fts` Virtual Table (Migration)

**Severity**: Blocker
**Affects**: Phase 8, Section 5.4 (`_search_fts_fallback`), Phase 0 migration system

### Problem

The FTS5 fallback (lines 565â€“586) queries `observations_fts` â€” a table that does not exist anywhere in the project schema:

- **Phase 0** defines `sessions`, `observations`, `function_map`, `call_graph`, `learnings`, `pending_queue`, `event_log`. No FTS5 virtual table.
- **Phase 4** (line 64) explicitly rejected SQLite FTS5: *"Separate FTS via SQLite FTS5 â€” we use LanceDB's built-in Tantivy FTS (unified search interface)"*.
- **Phase 0 amendments** â€” no FTS5 additions.

Phase 8 line 589 claims *"Phase 0's migration creates it"* â€” this is fabricated. The `try/except sqlite3.OperationalError` on line 582 prevents a crash, but the "fallback" silently returns zero results every time. That's a dead code path, not a fallback.

### Specification Change

Add a new migration that creates the FTS5 virtual table, insert/update/delete triggers, and backfills existing data.

**Migration SQL** (append as next sequential version in `MIGRATIONS` list):

```sql
-- FTS5 virtual table for CLI search fallback (Phase 8)
-- content-sync'd with observations table via triggers
CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts
USING fts5(
    title,
    summary,
    detail,
    content=observations,
    content_rowid=rowid
);

-- Backfill existing observations into FTS index
INSERT INTO observations_fts(rowid, title, summary, detail)
SELECT rowid, title, summary, COALESCE(detail, '')
FROM observations;

-- Keep FTS in sync on INSERT
CREATE TRIGGER observations_fts_insert AFTER INSERT ON observations BEGIN
    INSERT INTO observations_fts(rowid, title, summary, detail)
    VALUES (new.rowid, new.title, new.summary, COALESCE(new.detail, ''));
END;

-- Keep FTS in sync on UPDATE
CREATE TRIGGER observations_fts_update AFTER UPDATE ON observations BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, title, summary, detail)
    VALUES ('delete', old.rowid, old.title, old.summary, COALESCE(old.detail, ''));
    INSERT INTO observations_fts(rowid, title, summary, detail)
    VALUES (new.rowid, new.title, new.summary, COALESCE(new.detail, ''));
END;

-- Keep FTS in sync on DELETE
CREATE TRIGGER observations_fts_delete AFTER DELETE ON observations BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, title, summary, detail)
    VALUES ('delete', old.rowid, old.title, old.summary, COALESCE(old.detail, ''));
END;
```

**Why `detail` is included**: The `observations` table has three text columns suitable for search: `title`, `summary`, `detail`. Phase 8's original FTS5 definition (line 593) correctly included all three. The `detail` column may be `NULL` (it's optional), hence the `COALESCE` in triggers.

**Why backfill matters**: Without the `INSERT INTO observations_fts ... SELECT` statement, any observations that existed before this migration runs would be invisible to FTS search. The triggers only handle future writes.

**Migration ordering**: This migration must be appended after the Phase 6 learnings migration (which adds `times_seen`, `source_sessions`, `is_manual` to `learnings`). The exact version number depends on how many migrations prior phases have added.

### Updates to Phase 8 Sections

**Section 5.4**: Remove the comment `# observations table has FTS5 index (Phase 0 schema)` and replace with `# observations_fts created by Phase 8 migration`.

**Section 13 (Known Risks)**: Update the `observations_fts` risk row:

```markdown
| **FTS5 table not created yet** | Low (migration handles it) | Low | Migration creates table + backfills existing data. Fresh DB gets FTS on first migration run. |
```

**Section 10.2 (Test fixtures)**: The `seeded_db` fixture must run all migrations (including the new FTS5 migration) before seeding data, so the triggers populate `observations_fts` automatically. Verify with:

```python
def test_fts_table_exists(seeded_db):
    conn = sqlite3.connect(seeded_db)
    rows = conn.execute(
        "SELECT * FROM observations_fts WHERE observations_fts MATCH 'JWT'"
    ).fetchall()
    assert len(rows) >= 1  # seeded observation contains "JWT auth"
    conn.close()
```

### Minor schema note

Phase 8's test fixture (Section 10.2, line 980) inserts a `hook_type` column into `observations`. This column does not exist in Phase 0's schema. The test fixture must match the actual schema â€” remove `hook_type` from the INSERT or add it via a prior migration if another phase introduced it.

---

## Amendment 2: Standardize on Typer (Drop Click)

**Severity**: High
**Affects**: Phase 8, Section 2.1 (Dependencies), all CLI command definitions

### Problem

Phase 6 (Section 2.9, lines 840â€“848) committed to Typer for CLI commands:

```python
def learnings(
    action: str = typer.Argument("list", help="list|add|edit|remove|reset"),
    category: Optional[str] = typer.Option(None, "--category", "-c"),
    ...
)
```

Phase 8 switches to Click, claiming it is "lighter" (Section 2.1, line 51). This creates two problems:

1. **Two CLI parsers in one application.** Phase 6's `learnings_cmd.py` uses Typer semantics. Phase 8's commands use Click decorators. An implementer must either maintain both or retroactively rewrite Phase 6.

2. **The "lighter" argument is wrong.** Typer's default install (`pip install typer`) bundles `click`, `rich`, and `shellingham`. Phase 8 lists `click` and `rich` as "new dependencies" â€” but they're already transitive dependencies of `typer`. Choosing Click over Typer saves zero dependencies while adding inconsistency.

### Specification Change

Replace Click with Typer throughout Phase 8. Remove `click` from direct dependencies.

**Dependencies table (Section 2.1)** â€” replace with:

| Package | Version | Purpose | Already in project? |
|---------|---------|---------|---------------------|
| `typer` | â‰¥0.12 | CLI argument parsing, command groups (wraps Click) | Yes â€” Phase 6 |
| `rich` | â‰¥14.0 | Terminal formatting, tables, panels | Yes â€” transitive dep of `typer` |

**Why not Click justification** â€” replace with:

```markdown
**Why `typer`**: Typer is already a Phase 6 dependency. It wraps Click (so all Click
functionality is available) and uses Python type hints for parameter declaration,
which aligns naturally with Python 3.14. Its default install includes `rich` and
`shellingham`, eliminating two separate dependency entries. For the rare case where
raw Click interop is needed (e.g., registering a Click group), Typer provides
`typer.main.get_command()`.
```

**Root CLI group (Section 3.2)** â€” rewrite from Click to Typer:

```python
import typer
from claude_mem_lite.cli.report import report_cmd
from claude_mem_lite.cli.search_cmd import search_cmd
from claude_mem_lite.cli.mermaid_cmd import mermaid_cmd
from claude_mem_lite.cli.eval_cmd import eval_app

app = typer.Typer(
    name="claude-mem",
    help="claude-mem-lite: Local memory for Claude Code sessions.",
    no_args_is_help=True,
)

app.command(name="report")(report_cmd)
app.command(name="search")(search_cmd)
app.command(name="mermaid")(mermaid_cmd)
app.add_typer(eval_app, name="eval")


def _version_callback(value: bool):
    if value:
        from importlib.metadata import version
        typer.echo(f"claude-mem-lite {version('claude-mem-lite')}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=_version_callback,
                                  is_eager=True, help="Show version"),
):
    """claude-mem-lite: Local memory for Claude Code sessions."""
    pass
```

**Report command (Section 4.7)** â€” example conversion:

```python
def report_cmd(
    days: int = typer.Option(1, help="Report period in days (default: today)"),
    session_id: Optional[str] = typer.Option(None, "--session", help="Specific session ID"),
    markdown_path: Optional[str] = typer.Option(None, "--md",
        help="Export as markdown. Omit value for auto path."),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
    learnings_count: int = typer.Option(5, "--learnings", help="Number of learnings to show"),
):
    """Show session report with observations, function changes, and learnings."""
    config = _load_config()
    db_path = config.db_path
    # ... rest unchanged
```

**Search command (Section 5.6)** â€” example conversion:

```python
def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, help="Max results (default: 5)"),
    query_type: str = typer.Option("observation", "--type",
        help="Search type: observation | code | learning"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Search observations, code, or learnings."""
    # ... rest unchanged
```

**Mermaid command (Section 6.4)** â€” same pattern.

**Entry point (Section 3.1)** â€” update `pyproject.toml`:

```toml
[project.scripts]
claude-mem = "claude_mem_lite.cli.main:app"
```

Note: Typer apps are callable. `app` works directly as the entry point â€” no wrapper needed.

**Test fixtures (Section 10.2)** â€” replace `CliRunner` import:

```python
# BEFORE
from click.testing import CliRunner

# AFTER
from typer.testing import CliRunner
```

Typer's `CliRunner` wraps Click's runner and behaves identically. Test invocation syntax stays the same.

**Phase 7 eval integration (Section 9.2)** â€” use Typer sub-app instead of Click group:

```python
# eval_cmd.py
eval_app = typer.Typer(help="Evaluate compression quality and system health.")

@eval_app.command()
def compression(...): ...

@eval_app.command()
def benchmark(...): ...

@eval_app.command()
def health(...): ...
```

---

## Amendment 3: Define `_row_to_function_info` Hydration Logic

**Severity**: Low-Medium
**Affects**: Phase 8, Section 6.3 (`build_mermaid_from_db`)

### Problem

`build_mermaid_from_db` (line 758) calls `_row_to_function_info(f)` â€” a function that is never defined in the PRD. The data flow has an impedance mismatch:

- **Phase 2's `generate_mermaid()`** expects `list[FunctionInfo]` where each `FunctionInfo` has a `calls: list[CallInfo]` field containing resolved call targets.
- **Phase 8's queries** return flat SQL rows from `function_map` and separate edge rows from `call_graph`.
- The PRD references `_query_call_edges` (line 747) but never shows how those edges get attached to the `FunctionInfo.calls` field.

An implementer reading this PRD would need to reverse-engineer the hydration logic by cross-referencing Phase 2's `FunctionInfo` dataclass, Phase 0's `call_graph` schema, and Phase 8's query results.

### Specification Change

Add the following helper functions after `build_mermaid_from_db` in Section 6.3:

```python
from claude_mem_lite.ast_tracker.extractor import FunctionInfo, CallInfo


def _query_functions(
    conn: sqlite3.Connection,
    file_path: str,
    session_id: str,
) -> list[sqlite3.Row]:
    """Get all functions for a file in a session."""
    return conn.execute(
        """SELECT qualified_name, kind, parent_class, signature, decorators,
                  docstring, line_start, line_end, body_hash, change_type
           FROM function_map
           WHERE file_path = ? AND session_id = ?""",
        (file_path, session_id),
    ).fetchall()


def _query_call_edges(
    conn: sqlite3.Connection,
    file_path: str,
    session_id: str,
) -> list[sqlite3.Row]:
    """Get call graph edges originating from functions in this file."""
    return conn.execute(
        """SELECT caller_function, callee_function, callee_file,
                  call_type, line_number
           FROM call_graph
           WHERE caller_file = ? AND session_id = ?""",
        (file_path, session_id),
    ).fetchall()


def _row_to_function_info(
    row: sqlite3.Row,
    call_edges: list[sqlite3.Row],
) -> FunctionInfo:
    """Hydrate a FunctionInfo from a flat SQL row + associated call edges.

    Filters call_edges to only those where caller_function matches
    this row's qualified_name, then maps each to a CallInfo.
    """
    qualified_name = row["qualified_name"]

    calls = [
        CallInfo(
            raw_name=edge["callee_function"],
            resolved_name=edge["callee_function"],
            resolution=edge.get("call_type", "direct"),
            line_number=edge.get("line_number", 0),
        )
        for edge in call_edges
        if edge["caller_function"] == qualified_name
    ]

    return FunctionInfo(
        qualified_name=qualified_name,
        kind=row.get("kind", "function"),
        parent_class=row.get("parent_class"),
        signature=row["signature"],
        decorators=json.loads(row.get("decorators", "[]")),
        docstring=row.get("docstring"),
        line_start=row["line_start"],
        line_end=row["line_end"],
        body_hash=row["body_hash"],
        calls=calls,
    )
```

**Update the call site** in `build_mermaid_from_db` (line 746â€“758):

```python
# BEFORE (edges fetched but unused in hydration)
functions = _query_functions(conn, file_path, session_id)
edges = _query_call_edges(conn, file_path, session_id)
...
func_infos = [_row_to_function_info(f) for f in functions]

# AFTER (edges passed to hydration)
functions = _query_functions(conn, file_path, session_id)
edges = _query_call_edges(conn, file_path, session_id)
...
func_infos = [_row_to_function_info(f, edges) for f in functions]
```

### Schema note

The `call_graph` table (Phase 0) stores `call_type` not `resolution`. The mapping is:

| `call_graph.call_type` | `CallInfo.resolution` |
|---|---|
| `direct` | `direct` |
| `self_method` | `self_method` |
| `import` | `import` |
| (other/missing) | `unresolved` |

If `call_type` is not stored in `call_graph`, default to `"direct"` since the call graph only records resolved calls.

---

## Amendment 4: Add `claude-mem status` Command (Moved from Phase 9)

**Severity**: Medium
**Affects**: Phase 8, Section 14 (Future Extensions â†’ moved to this phase), Section 3.2 (command tree)

### Problem

`claude-mem status` is listed under Future Extensions (Section 14, line 1114). But when a user's setup breaks â€” worker is down, DB is locked, config is missing â€” the first thing they need is a diagnostic command. Without `status`, they're left guessing why `search` degrades to FTS or `report` returns empty results.

This command requires no new logic. Every check it performs already exists in other commands or infrastructure code.

### Specification Change

Move `claude-mem status` to Phase 8. Add it to the command tree and implementation.

**Updated command tree (Section 3.2)**:

```
claude-mem
â”œâ”€â”€ report          Session report (default: today)
â”œâ”€â”€ search <query>  Hybrid search over observations
â”œâ”€â”€ mermaid         Call graph as Mermaid diagram
â”œâ”€â”€ status          System health check
â”œâ”€â”€ eval            (Phase 7)
```

**Implementation**:

```python
def status_cmd():
    """Show system health: database, worker, last session."""
    config = _load_config()
    console = Console()

    # 1. Database
    db_path = config.db_path
    if not Path(db_path).exists():
        console.print("[red]Database[/red]: not found")
        console.print(f"  Expected at: {db_path}")
        console.print("  Run a Claude Code session to initialize.")
        raise typer.Exit(1)

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA integrity_check")

        obs_count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        learning_count = conn.execute(
            "SELECT COUNT(*) FROM learnings WHERE is_active = 1"
        ).fetchone()[0]

        last_session = conn.execute(
            "SELECT started_at FROM sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        last_session_ago = _relative_time(last_session[0]) if last_session else "never"

        # Check FTS table exists
        fts_ok = True
        try:
            conn.execute("SELECT COUNT(*) FROM observations_fts")
        except sqlite3.OperationalError:
            fts_ok = False

        conn.close()

        console.print(f"[green]Database[/green]: {obs_count} observations, "
                       f"{session_count} sessions, {learning_count} active learnings")
        console.print(f"  Last session: {last_session_ago}")
        if not fts_ok:
            console.print("  [yellow]FTS index: missing (run migration)[/yellow]")

    except sqlite3.DatabaseError as e:
        console.print(f"[red]Database[/red]: corrupt ({e})")
        raise typer.Exit(1)

    # 2. Worker
    socket = _discover_worker(config)
    if socket:
        console.print(f"[green]Worker[/green]: running ({socket})")
    else:
        console.print("[yellow]Worker[/yellow]: not running (search will use FTS fallback)")

    # 3. Config
    console.print(f"[green]Config[/green]: {config.data_dir}")
```

**Registration**: Add to `main.py`:

```python
app.command(name="status")(status_cmd)
```

**Tests**: Add 2 tests to the test plan (Section 10.1):

| Category | Tests | What it validates |
|----------|-------|-------------------|
| **Status** | 2 | Healthy system shows green, missing DB shows error + exit 1 |

Update total test count from 27 to 29.

**Acceptance criteria**: Add to Section 12:

```markdown
- [ ] `claude-mem status` shows DB stats, worker status, and config path
- [ ] `claude-mem status` with missing DB shows error and exits with code 1
```

**Remove from Section 14 (Future Extensions)**: Delete the `claude-mem status` line.

---

## Amendment 5: Fallback to FTS on Worker Search Errors (Not Just Unreachable)

**Severity**: Medium
**Affects**: Phase 8, Section 5.2 (Search strategy), Section 5.6 (Click/Typer command)

### Problem

The search fallback (Section 5.2, lines 508â€“519) only triggers when the worker is **unreachable** (socket missing, health check fails). If the worker is running but the `/api/search` endpoint returns a 500 error (LanceDB corruption, embedding model OOM, malformed query), the CLI surfaces a raw HTTP error instead of degrading gracefully.

```
# Current behavior when worker returns 500:
$ claude-mem search "JWT auth"
httpx.HTTPStatusError: Server error '500 Internal Server Error'
```

This violates the PRD's own design principle (Section 1.3, line 36): *"Search should always work. If worker is down, fall back to FTS5 search directly against SQLite. Degrade gracefully, don't error."*

### Specification Change

Wrap the worker search call in error handling and fall back to FTS on any failure.

**Updated search command (Section 5.6)**:

```python
def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, help="Max results"),
    query_type: str = typer.Option("observation", "--type",
        help="Search type: observation | code | learning"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Search observations, code, or learnings."""
    config = _load_config()
    db_path = config.db_path

    socket = _discover_worker(config)
    search_type = "fts"
    results = []

    if socket:
        try:
            results, search_type = _search_via_worker(socket, query, limit, query_type)
        except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as e:
            # Worker is up but search failed â€” degrade to FTS
            console.print(
                f"[yellow]Worker search failed ({type(e).__name__}), "
                f"falling back to text search[/yellow]\n"
            )
            results = _search_fts_fallback(db_path, query, limit)
            search_type = "fts"
    else:
        results = _search_fts_fallback(db_path, query, limit)

    if as_json:
        output = {
            "query": query,
            "results": results,
            "count": len(results),
            "search_type": search_type,
        }
        typer.echo(json.dumps(output, indent=2, default=str))
        return

    _render_results(results, query, search_type)
```

**Updated search strategy diagram (Section 5.2)**:

```
User runs: claude-mem search "JWT auth"
    â”‚
    â”œâ”€â”€ Worker running? (check worker.sock / worker.pid / health)
    â”‚   â”œâ”€â”€ YES â†’ HTTP GET /api/search?q=JWT+auth&limit=5
    â”‚   â”‚         â”œâ”€â”€ 200 OK â†’ hybrid results (vector + FTS, best quality)
    â”‚   â”‚         â””â”€â”€ Error (500, timeout, connection reset)
    â”‚   â”‚               â†’ Warning to stderr
    â”‚   â”‚               â†’ Direct SQLite FTS5 fallback
    â”‚   â”‚
    â”‚   â””â”€â”€ NO â†’ Direct SQLite FTS5 fallback
    â”‚            SELECT * FROM observations_fts WHERE observations_fts MATCH ?
    â”‚            (text-only search, no semantic matching)
    â”‚
    â””â”€â”€ Render results with rich
        Badge: [green]hybrid[/green] or [yellow]fts[/yellow]
```

**Event logging**: When fallback triggers due to worker error, log the error type:

```python
# In event_log data field:
{"query": "JWT auth", "search_type": "fts", "fallback_reason": "HTTPStatusError: 500"}
```

This gives `claude-mem eval health` (Phase 7) visibility into how often the worker's search endpoint fails.

---

## Summary

| # | Amendment | Severity | Type |
|---|-----------|----------|------|
| 1 | Create `observations_fts` FTS5 virtual table + triggers + backfill | Blocker | New migration |
| 2 | Standardize on Typer, drop direct Click dependency | High | Dependency + all command rewrites |
| 3 | Define `_row_to_function_info` hydration logic | Low-Medium | Missing spec |
| 4 | Move `claude-mem status` from Phase 9 to Phase 8 | Medium | New command |
| 5 | Fallback to FTS on worker search errors, not just unreachable | Medium | Error handling |

**Net effect on Phase 8 scope**: Amendments 3 and 5 are small additions (< 30 lines each). Amendment 1 is a migration with triggers (~30 lines SQL). Amendment 4 adds one simple command (~40 lines). Amendment 2 is a rewrite of decorator style but not logic. The "~1 session" effort estimate remains realistic.
