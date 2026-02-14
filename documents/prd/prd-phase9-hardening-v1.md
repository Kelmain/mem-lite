# Phase 9 PRD: Hardening + Polish

**Version**: 1.0
**Date**: 2026-02-08
**Depends on**: All phases (0â€“8)
**Nature**: Ongoing â€” not a single deliverable but a prioritized backlog
**Estimated effort**: ~2-3 sessions initially, then continuous

---

## 1. Scope

### 1.1 What this phase is

Phase 9 is the **production-readiness backlog** â€” every deferral, edge case, and optimization opportunity identified during Phases 0â€“8. Unlike previous phases which built new capabilities, Phase 9 hardens existing ones.

This PRD inventories, prioritizes, and specifies every deferred item. Items are grouped into **tiers** by impact on daily usability.

### 1.2 What this phase is NOT

- **New features** â€” no new capabilities. Everything here was already identified as a deferral in Phases 0â€“8.
- **Rewrite** â€” no architectural changes. Hardening means making what exists robust, not redesigning it.
- **Endless** â€” the Tier 1 items make the tool production-ready. Tiers 2â€“3 are "nice to have" improvements.

### 1.3 Corrections to implementation plan

| Item | Implementation Plan | Corrected (Phase 9 PRD) | Rationale |
|------|---------------------|------------------------|-----------|
| Multi-project support | Listed as Phase 9 task | **Already partially solved.** Phase 1 uses `CLAUDE_PROJECT_DIR`. Phase 0's `Config` resolves to per-project DB path. What's missing: CLI project selection, shared cross-project learnings. | Don't overstate the remaining work. |
| ONNX INT8 quantization | "if latency is an issue (~300MB, ~2x CPU speedup)" | **Conditional on Phase 4 profiling data.** Only implement if embedding latency exceeds 150ms in real usage. sentence-transformers has built-in `export_dynamic_quantized_onnx_model()` â€” the work is ~20 lines of config, not a research project. | The plan made this sound harder than it is. |
| Worker watchdog | "5 lines: check PID file, restart if dead" | **Not needed.** Hooks already check worker health and degrade gracefully. If worker dies mid-session, pending_queue preserves everything. Next `claude-mem-worker start` catches up. A watchdog adds complexity for a problem that self-heals. | Watchdogs monitor services that must be continuously available. Our worker is ephemeral by design (30-min idle timeout). |
| `pyproject.toml` finalization | Listed generically | Specific spec: metadata, extras groups, entry points, build system. | Needs precision to be actionable. |

---

## 2. Deferral Inventory

Every item deferred to Phase 9 across all PRDs, organized by source phase.

### From Phase 0 (Storage)

| ID | Item | Severity | Reference |
|----|------|----------|-----------|
| D0-1 | `raw_output` 30-day retention policy + purge job | Medium | Â§2.5: "Keep for 30 days, then purge" |
| D0-2 | SQLite VACUUM after bulk deletes | Low | Implied by D0-1 |

### From Phase 2 (AST Tracker)

| ID | Item | Severity | Reference |
|----|------|----------|-----------|
| D2-1 | Non-Python language support (tree-sitter) | Low | Â§1.2: "tree-sitter support for JS/TS/etc." |
| D2-2 | Cross-file import resolution | Medium | Â§1.2: "Full cross-file resolution deferred" |

### From Phase 3 (Worker/Compression)

| ID | Item | Severity | Reference |
|----|------|----------|-----------|
| D3-1 | Tiered compression (Haiku for obs, Sonnet for summaries) | Medium | Based on Phase 7 benchmark results |
| D3-2 | Bash priority refinement (read-only vs mutation commands) | Low | Phase 1 v3 Â§TODO |
| D3-3 | Mid-session worker crash: stale socket cleanup | Low | Â§2.9 degradation table |

### From Phase 4 (Embeddings/Search)

| ID | Item | Severity | Reference |
|----|------|----------|-----------|
| D4-1 | ONNX INT8 quantization for Qwen3-Embedding | Low | Â§1.2: "deferred to Phase 9" |
| D4-2 | Re-embed missing observations (catch-up mechanism) | Medium | Â§2.7: "embedding failure is non-fatal" |
| D4-3 | IVF_PQ vector index creation | Low | Â§10: "when search latency exceeds 100ms" |
| D4-4 | Tantivy FTS periodic rebuild | Low | Â§8: "create_fts_index(replace=True)" |
| D4-5 | `claude-mem reindex --dim` command | Low | Â§9 open question |

### From Phase 5 (Context Injection)

| ID | Item | Severity | Reference |
|----|------|----------|-----------|
| D5-1 | PreCompact hook to preserve context during compaction | Low | Â§3.4: "Phase 9 may add" |

### From Phase 6 (Learnings)

| ID | Item | Severity | Reference |
|----|------|----------|-----------|
| D6-1 | Prune inactive learnings after N sessions | Low | Â§8: "Monitor DB growth over 50+ sessions" |
| D6-2 | Cross-project learnings | Low | Â§1.2: "Each project has its own set" |
| D6-3 | LLM-assisted call graph edge extraction | Low | Â§4.5: "if pair-matching proves too crude" |

### From Phase 7 (Eval)

| ID | Item | Severity | Reference |
|----|------|----------|-----------|
| D7-1 | Automated eval on every N observations | Medium | Â§9.6 |
| D7-2 | Regression detection (quality threshold alerting) | Low | Â§9.6 |
| D7-3 | Raw output lifecycle management (eval corpus preservation) | Medium | Â§9.6, ties to D0-1 |

### From Phase 8 (CLI Reports)

| ID | Item | Severity | Reference |
|----|------|----------|-----------|
| D8-1 | `claude-mem status` â€” quick one-liner | Medium | Â§14 |
| D8-2 | `claude-mem compress --pending` â€” manual catch-up | Medium | Â§14, architecture doc |
| D8-3 | `claude-mem export` â€” full data dump | Low | Â§14 |
| D8-4 | `claude-mem prune --older-than 30d` â€” cleanup | Medium | Â§14, ties to D0-1 |
| D8-5 | Shell completion for CLI | Low | Â§14 |

### Cross-cutting

| ID | Item | Severity | Reference |
|----|------|----------|-----------|
| X-1 | `pyproject.toml` finalization + packaging | **High** | Implementation plan |
| X-2 | `README.md` + installation docs | **High** | Implementation plan |
| X-3 | Error recovery: corrupt SQLite | Low | Implementation plan |
| X-4 | Error recovery: LanceDB index corruption | Low | Implementation plan |

---

## 3. Priority Tiers

### Tier 1: Must-Have for Daily Use (~1 session)

These items block comfortable daily usage. Without them, the tool works but is rough around the edges.

| ID | Item | Why Tier 1 |
|----|------|-----------|
| **X-1** | `pyproject.toml` finalization | Can't install without it |
| **X-2** | `README.md` + installation docs | Can't onboard without it |
| **D8-1** | `claude-mem status` | Most-used command in any CLI tool â€” "is it working?" |
| **D8-2** | `claude-mem compress --pending` | Manual recovery when worker missed items |
| **D8-4** | `claude-mem prune` | DB grows without bounds otherwise (D0-1 depends on this) |
| **D0-1** | `raw_output` retention + purge | Unbounded growth is the #1 storage risk |

### Tier 2: Quality-of-Life (~1 session)

These improve robustness and performance but aren't blockers.

| ID | Item | Why Tier 2 |
|----|------|-----------|
| **D3-1** | Tiered compression | Data-driven model selection based on Phase 7 benchmark |
| **D4-2** | Re-embed missing observations | Self-healing for embedding failures |
| **D7-1** | Automated eval triggers | Quality monitoring without manual intervention |
| **D7-3** | Raw output lifecycle (eval preservation) | Connects D0-1 purge with eval needs |
| **D2-2** | Cross-file import resolution | Improves call graph accuracy significantly |

### Tier 3: Opportunistic (as needed)

Only implement if a real problem is observed.

| ID | Item | Trigger condition |
|----|------|------------------|
| **D4-1** | ONNX INT8 quantization | Embedding latency >150ms in real usage |
| **D4-3** | IVF_PQ vector index | Search latency >100ms with >1000 observations |
| **D4-4** | Tantivy FTS rebuild | Search quality degrades (zero-result rate >20%) |
| **D2-1** | Non-Python language support | User works heavily in JS/TS |
| **D6-2** | Cross-project learnings | Multiple active projects with shared patterns |
| **D5-1** | PreCompact hook | Context lost during compaction observed |
| **X-3** | Corrupt SQLite recovery | Corruption actually occurs |
| **X-4** | LanceDB corruption recovery | Corruption actually occurs |
| **D6-1** | Prune inactive learnings | Learnings table >5MB |
| **D6-3** | LLM-assisted graph healing | Pair-matching resolves <50% of edges |
| **D3-2** | Bash priority refinement | Noisy low-priority bash observations |
| **D8-3** | Full data export | Requested by user |
| **D8-5** | Shell completion | Requested by user |
| **D4-5** | `reindex --dim` command | Embedding model changed |
| **D0-2** | SQLite VACUUM | DB file >500MB after pruning |
| **D7-2** | Regression detection | Running automated evals regularly |

---

## 4. Tier 1 Specifications

### 4.1 `pyproject.toml` Finalization (X-1)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "claude-mem-lite"
version = "0.1.0"
description = "Local memory for Claude Code sessions"
readme = "README.md"
license = "MIT"
requires-python = ">=3.12"

authors = [
    { name = "Armin" },
]

dependencies = [
    "anthropic>=0.40.0",
    "click>=8.1",
    "rich>=14.0",
    "pydantic>=2.0",
    "httpx>=0.27",
]

[project.optional-dependencies]
worker = [
    "aiosqlite>=0.20",
    "uvicorn>=0.32",
    "starlette>=0.41",
    "sentence-transformers>=4.0",
    "lancedb>=0.17",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
    "coverage>=7.0",
]
onnx = [
    "onnxruntime>=1.20",
    "optimum>=1.23",
]

[project.scripts]
claude-mem = "claude_mem_lite.cli.main:cli"
claude-mem-worker = "claude_mem_lite.worker.lifecycle:main"

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "TCH"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Key decisions**:
- **`hatchling`** as build backend â€” simpler than setuptools, no `setup.py` needed. Standard for modern Python projects.
- **Split dependencies**: Base (hooks + CLI) is lightweight. `[worker]` pulls in heavy ML deps only when worker is needed. This means `pip install claude-mem-lite` is fast for the CLI; `pip install claude-mem-lite[worker]` gets everything.
- **`>=3.12`** minimum, not `>=3.14` â€” no Python 3.14-specific features are used. Wider compatibility.
- **`[onnx]` extra** â€” opt-in for ONNX quantization (Tier 3, D4-1).

### 4.2 `README.md` (X-2)

Structure:

```markdown
# claude-mem-lite

Local memory for Claude Code sessions. Captures tool outputs, compresses them
with Haiku 4.5, embeds for semantic search, and injects relevant context at
session start.

## Quick Start

pip install claude-mem-lite[worker]
claude-mem install-hooks
claude-mem-worker start --daemon

## How It Works

[Architecture diagram â€” simplified version of architecture doc]

## Commands

### claude-mem report
### claude-mem search <query>
### claude-mem mermaid [file]
### claude-mem eval {compression,benchmark,health}
### claude-mem status
### claude-mem prune --older-than 30d

## Configuration

~/.claude-mem/config.toml (auto-created on first run)

## Cost

~$0.50-0.60 per session (Haiku 4.5 compression).
Optional Sonnet for session summaries (~$0.05 extra).

## Development

pip install -e ".[dev,worker]"
pytest
ruff check

## Architecture

See docs/architecture.md for full design documentation.

## License

MIT
```

**Deliverable**: `README.md` at project root + `docs/architecture.md` (copy of architecture doc).

### 4.3 `claude-mem status` (D8-1)

The quickest sanity check: "is everything working?"

```python
@click.command()
@click.pass_context
def status(ctx):
    """Quick system status check."""
    config = ctx.obj["config"]
    db_path = ctx.obj["db_path"]
    console = Console()

    # Database
    if not Path(db_path).exists():
        console.print("[red]âœ—[/red] Database: not found")
    else:
        conn = sqlite3.connect(db_path)
        obs_count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        last_session = conn.execute(
            "SELECT started_at FROM sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        db_size = Path(db_path).stat().st_size / (1024 * 1024)
        console.print(
            f"[green]âœ“[/green] Database: {obs_count} observations, "
            f"{session_count} sessions, {db_size:.1f}MB"
        )
        if last_session:
            age = _relative_time(last_session[0])
            console.print(f"  Last session: {age}")
        conn.close()

    # Worker
    socket_path = _discover_worker(config)
    if socket_path:
        console.print("[green]âœ“[/green] Worker: running")
    else:
        console.print("[yellow]â—‹[/yellow] Worker: not running")

    # Pending queue
    if Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        pending = conn.execute(
            "SELECT COUNT(*) FROM pending_queue WHERE status = 'raw'"
        ).fetchone()[0]
        errors = conn.execute(
            "SELECT COUNT(*) FROM pending_queue WHERE status = 'error'"
        ).fetchone()[0]
        conn.close()
        if pending > 0:
            console.print(f"[yellow]![/yellow] Pending queue: {pending} unprocessed items")
        if errors > 0:
            console.print(f"[red]![/red] Error queue: {errors} failed items")
        if pending == 0 and errors == 0:
            console.print("[green]âœ“[/green] Queue: clear")

    # Hooks
    hooks_installed = _check_hooks_installed(config)
    if hooks_installed:
        console.print("[green]âœ“[/green] Hooks: installed")
    else:
        console.print("[red]âœ—[/red] Hooks: not installed (run: claude-mem install-hooks)")

    # Learnings (if table exists)
    if Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        try:
            active = conn.execute(
                "SELECT COUNT(*) FROM learnings WHERE is_active = 1"
            ).fetchone()[0]
            console.print(f"[green]âœ“[/green] Learnings: {active} active")
        except sqlite3.OperationalError:
            pass  # Table doesn't exist yet
        conn.close()
```

**Output example:**
```
âœ“ Database: 312 observations, 47 sessions, 8.2MB
  Last session: 2h ago
âœ“ Worker: running
âœ“ Queue: clear
âœ“ Hooks: installed
âœ“ Learnings: 23 active
```

### 4.4 `claude-mem compress --pending` (D8-2)

Manual catch-up compression for when the worker missed items.

```python
@click.command()
@click.option("--limit", default=50, help="Max items to process")
@click.option("--dry-run", is_flag=True, help="Show pending items without processing")
@click.pass_context
def compress(ctx, limit, dry_run):
    """Process pending queue items that haven't been compressed yet.

    Use when worker was down or items were missed.

    Examples:
        claude-mem compress --pending          # Process up to 50 items
        claude-mem compress --pending --dry-run # Show what would be processed
    """
    config = ctx.obj["config"]
    db_path = ctx.obj["db_path"]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    pending = conn.execute(
        "SELECT id, session_id, tool_name, created_at FROM pending_queue "
        "WHERE status = 'raw' ORDER BY created_at LIMIT ?",
        (limit,),
    ).fetchall()

    if not pending:
        click.echo("No pending items.")
        return

    if dry_run:
        console = Console()
        table = Table(title=f"Pending items ({len(pending)})")
        table.add_column("ID", style="dim")
        table.add_column("Tool")
        table.add_column("Age")
        for row in pending:
            table.add_row(row["id"][:8], row["tool_name"], _relative_time(row["created_at"]))
        console.print(table)
        return

    # Check if worker is running â€” delegate to worker if possible
    socket = _discover_worker(config)
    if socket:
        click.echo(f"Worker is running â€” triggering catch-up for {len(pending)} items...")
        # POST /api/catch-up on worker
        _trigger_worker_catchup(socket)
    else:
        click.echo(f"Worker not running â€” starting inline compression for {len(pending)} items...")
        # Run compression synchronously (slow but works)
        _compress_inline(config, pending)

    conn.close()
```

**Implementation note**: Inline compression (worker not running) uses synchronous `anthropic.Anthropic()` client. Slower than the async worker but functional for catch-up.

### 4.5 `claude-mem prune` (D8-4, D0-1, D7-3)

The unified cleanup command. Addresses `raw_output` retention, old observation pruning, and eval corpus preservation in one place.

```python
@click.command()
@click.option("--older-than", default="30d",
              help="Delete data older than this. Format: Nd (days), Nw (weeks), Nm (months)")
@click.option("--keep-raw", default=100, type=int,
              help="Always keep the N most recent raw_outputs (for eval). Default: 100")
@click.option("--vacuum", is_flag=True, help="Run VACUUM after pruning to reclaim disk space")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted")
@click.pass_context
def prune(ctx, older_than, keep_raw, vacuum, dry_run):
    """Clean up old data to manage database size.

    Pruning strategy:
    1. Clear raw_output from pending_queue items older than threshold
       (keeps the most recent N for eval benchmarking)
    2. Delete event_log entries older than threshold
    3. Optionally VACUUM the database to reclaim disk space

    Does NOT delete observations (they're small) or learnings (they're valuable).

    Examples:
        claude-mem prune --older-than 30d --dry-run
        claude-mem prune --older-than 60d --vacuum
        claude-mem prune --older-than 90d --keep-raw 50
    """
    db_path = ctx.obj["db_path"]
    cutoff = _parse_duration(older_than)

    conn = sqlite3.connect(db_path)

    # 1. Count what would be affected
    raw_count = conn.execute(
        """SELECT COUNT(*) FROM pending_queue
           WHERE raw_output IS NOT NULL AND created_at < ?
           AND id NOT IN (
               SELECT id FROM pending_queue
               WHERE raw_output IS NOT NULL
               ORDER BY created_at DESC LIMIT ?
           )""",
        (cutoff, keep_raw),
    ).fetchone()[0]

    event_count = conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE timestamp < ?",
        (cutoff,),
    ).fetchone()[0]

    # Size estimate
    raw_bytes = conn.execute(
        """SELECT SUM(LENGTH(raw_output)) FROM pending_queue
           WHERE raw_output IS NOT NULL AND created_at < ?
           AND id NOT IN (
               SELECT id FROM pending_queue
               WHERE raw_output IS NOT NULL
               ORDER BY created_at DESC LIMIT ?
           )""",
        (cutoff, keep_raw),
    ).fetchone()[0] or 0

    console = Console()
    console.print(f"\n[bold]Prune plan[/bold] (data older than {older_than}):")
    console.print(f"  raw_output: {raw_count} items ({raw_bytes / (1024*1024):.1f}MB)")
    console.print(f"  event_log:  {event_count} entries")
    console.print(f"  Preserving: {keep_raw} most recent raw outputs for eval")

    if dry_run:
        console.print("\n[dim]Dry run â€” no changes made.[/dim]")
        conn.close()
        return

    if raw_count == 0 and event_count == 0:
        console.print("\n[dim]Nothing to prune.[/dim]")
        conn.close()
        return

    # 2. Execute
    conn.execute("BEGIN")

    # Clear raw_output (SET NULL, don't delete the queue row)
    conn.execute(
        """UPDATE pending_queue SET raw_output = NULL
           WHERE raw_output IS NOT NULL AND created_at < ?
           AND id NOT IN (
               SELECT id FROM pending_queue
               WHERE raw_output IS NOT NULL
               ORDER BY created_at DESC LIMIT ?
           )""",
        (cutoff, keep_raw),
    )

    # Delete old event_log entries
    conn.execute(
        "DELETE FROM event_log WHERE timestamp < ?",
        (cutoff,),
    )

    conn.execute("COMMIT")

    console.print(f"\n[green]âœ“[/green] Pruned {raw_count} raw outputs + {event_count} event log entries")

    # 3. Optional VACUUM
    if vacuum:
        db_size_before = Path(db_path).stat().st_size
        console.print("[dim]Running VACUUM (this may take a moment)...[/dim]")
        conn.execute("VACUUM")
        db_size_after = Path(db_path).stat().st_size
        saved = (db_size_before - db_size_after) / (1024 * 1024)
        console.print(f"[green]âœ“[/green] VACUUM complete. Reclaimed {saved:.1f}MB")

    conn.close()
```

**Design decisions**:

1. **Don't delete observations** â€” they're small (~500 tokens each, ~200 bytes compressed). 10,000 observations is <2MB. The value of keeping them (searchable history) far outweighs storage cost.

2. **Don't delete learnings** â€” they're valuable long-term knowledge. Inactive learnings with low confidence are invisible to context injection (Phase 5 filters at â‰¥0.5) but kept for potential future reactivation.

3. **Only clear `raw_output`** â€” the biggest storage consumer. A single tool output can be 10KBâ€“500KB. Setting to NULL (not deleting the row) preserves the queue metadata for auditing.

4. **Preserve recent raw outputs** â€” Phase 7's benchmark runner needs `raw_output` to re-compress and compare models. Default: keep 100 most recent.

5. **VACUUM is opt-in** â€” it locks the database and requires temporary disk space equal to the DB size. Not something to do casually during a session.

**Storage math**:
- 50 observations/session Ã— 50KB avg raw_output = 2.5MB/session
- 100 sessions = 250MB in raw_output alone
- After 30-day prune: depends on session frequency. At 2 sessions/day, 60 sessions pruned = ~150MB reclaimed
- VACUUM after prune recovers freed pages

---

## 5. Tier 2 Specifications

### 5.1 Tiered Compression (D3-1)

**Prerequisite**: Phase 7 benchmark results showing Haiku vs Sonnet quality-per-dollar scores.

**Expected outcome** (from Phase 7 research): Haiku wins on quality/dollar for structured observation extraction. Sonnet may win for session summaries where multi-step reasoning matters.

**Implementation**: Add a `model` parameter to the compressor, selected by operation type:

```python
# In config.py
COMPRESSION_MODELS = {
    "observation": "claude-haiku-4-5-20250929",   # Structured extraction â€” Haiku excels
    "summary": "claude-haiku-4-5-20250929",        # Default: Haiku. Switch to Sonnet if benchmark shows benefit
}
```

**Change scope**: One line in `Compressor.__init__` to accept model override. One line in `Summarizer` to use the summary model. Total: <10 lines of code change.

**Decision criteria**: Switch summary model to Sonnet only if:
- Phase 7 benchmark shows Sonnet `info_preservation` â‰¥ 0.15 higher than Haiku on summary-type text
- AND the cost difference is acceptable ($0.05 vs $0.15 per session for summaries)

### 5.2 Re-embed Missing Observations (D4-2)

Catch-up mechanism for observations that failed to embed (LanceDB error, worker restart, etc.).

```python
@click.command()
@click.option("--limit", default=100, help="Max observations to re-embed")
@click.pass_context
def reembed(ctx, limit):
    """Re-embed observations missing from LanceDB vector index.

    Finds observations in SQLite that don't have a corresponding LanceDB entry
    and embeds them.
    """
    config = ctx.obj["config"]

    # This requires the worker's embedder and lance_store
    socket = _discover_worker(config)
    if not socket:
        click.echo("Worker must be running for re-embedding. Start with: claude-mem-worker start")
        raise SystemExit(1)

    # POST /api/reembed?limit=100
    response = _post_worker(socket, "/api/reembed", {"limit": limit})
    embedded = response.get("embedded", 0)
    skipped = response.get("skipped", 0)
    click.echo(f"Re-embedded {embedded} observations ({skipped} skipped/already indexed)")
```

**Worker endpoint** (added to `server.py`):

```python
@app.post("/api/reembed")
async def reembed(request: Request):
    """Find and embed observations missing from LanceDB."""
    body = await request.json()
    limit = body.get("limit", 100)

    # Get all observation IDs from SQLite
    sql_ids = set(r["id"] for r in await db.execute_fetchall(
        "SELECT id FROM observations ORDER BY created_at DESC LIMIT ?", (limit * 2,)
    ))

    # Get all observation IDs from LanceDB
    lance_ids = set()
    try:
        table = lance_store._tables["observations_vec"]
        lance_ids = set(r["obs_id"] for r in table.search().select(["obs_id"]).limit(limit * 2).to_list())
    except Exception:
        pass

    # Find missing
    missing = sql_ids - lance_ids
    if not missing:
        return {"embedded": 0, "skipped": len(sql_ids)}

    # Embed missing observations
    embedded = 0
    for obs_id in list(missing)[:limit]:
        row = await db.execute_fetchone(
            "SELECT id, session_id, title, summary, files_touched, created_at "
            "FROM observations WHERE id = ?", (obs_id,)
        )
        if row:
            try:
                await asyncio.to_thread(
                    lance_store.add_observation,
                    obs_id=row["id"], session_id=row["session_id"],
                    title=row["title"], summary=row["summary"],
                    files_touched=row["files_touched"] or "",
                    functions_changed="",
                    created_at=row["created_at"],
                )
                embedded += 1
            except Exception as e:
                logger.log("reembed.error", {"obs_id": obs_id, "error": str(e)})

    return {"embedded": embedded, "skipped": len(sql_ids) - len(missing)}
```

### 5.3 Automated Eval Triggers (D7-1)

Run deterministic eval scoring automatically every N observations.

```python
# In worker/processor.py â€” after process_item completes

self._compression_count += 1
if self._compression_count % 20 == 0:  # Every 20 compressions
    await self._run_deterministic_eval()

async def _run_deterministic_eval(self):
    """Score last 20 observations with deterministic metrics only.

    No API calls, <50ms total. Logs results to event_log.
    """
    from claude_mem_lite.eval.evaluator import DeterministicEvaluator

    evaluator = DeterministicEvaluator(self.db)
    results = await evaluator.score_recent(limit=20)

    avg_structural = sum(r.structural_validity for r in results) / len(results)
    avg_ratio = sum(r.compression_ratio for r in results) / len(results)
    avg_title = sum(r.title_quality for r in results) / len(results)

    self.logger.log("eval.auto_deterministic", {
        "count": len(results),
        "avg_structural_validity": avg_structural,
        "avg_compression_ratio": avg_ratio,
        "avg_title_quality": avg_title,
    })

    # Alert on quality drop
    if avg_structural < 0.9:
        self.logger.log("eval.quality_alert", {
            "metric": "structural_validity",
            "value": avg_structural,
            "threshold": 0.9,
        })
```

**Cost**: Zero â€” deterministic scoring only. No API calls.

### 5.4 Cross-file Import Resolution (D2-2)

Improve call graph accuracy by following imports across files in the project.

```python
class ProjectIndex:
    """Lightweight project-wide import index.

    Maps module paths to qualified function names for cross-file resolution.
    Built incrementally as files are scanned.
    """

    def __init__(self):
        self._modules: dict[str, set[str]] = {}  # module_path â†’ {qualified_names}

    def register_file(self, file_path: str, functions: list[FunctionInfo]) -> None:
        """Register a scanned file's exports."""
        module = _path_to_module(file_path)  # "auth/service.py" â†’ "auth.service"
        self._modules[module] = {f.qualified_name for f in functions}

    def resolve_import(self, import_path: str) -> str | None:
        """Try to resolve an import to a qualified name.

        Example: "auth.service.AuthService" â†’ "AuthService" in auth/service.py
        """
        parts = import_path.rsplit(".", 1)
        if len(parts) == 2:
            module, name = parts
            if module in self._modules and name in self._modules[module]:
                return f"{module}.{name}"
        return None
```

**Storage**: Add a `project_index` JSON column to the `sessions` table that accumulates the import map. Or simpler: rebuild from `function_map` table on demand (~10ms for 100 files).

**Integration**: The PostToolUse hook (Phase 1) already scans changed files. Add a `project_index` lookup step before persisting `call_graph` edges.

---

## 6. Tier 3 Quick Specs

Brief specifications for conditional items. Only implement when the trigger condition is met.

### 6.1 ONNX INT8 Quantization (D4-1)

**Trigger**: Embedding latency >150ms measured in `eval health`.

sentence-transformers 4.x+ has built-in ONNX support:

```python
from sentence_transformers import SentenceTransformer, export_dynamic_quantized_onnx_model

# One-time export
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", backend="onnx")
export_dynamic_quantized_onnx_model(
    model,
    quantization_config="avx2",  # or "arm64" for Apple Silicon
    model_name_or_path="~/.claude-mem/models/qwen3-int8/",
)

# Runtime loading
model = SentenceTransformer("~/.claude-mem/models/qwen3-int8/", backend="onnx")
```

**Expected results**: ~2-3x CPU speedup (80-150ms â†’ 30-60ms), ~75% model size reduction (2.4GB â†’ ~600MB), ~99% accuracy retention for embedding similarity.

**Implementation**: ~20 lines. Add `claude-mem optimize-embeddings` command that exports the quantized model, then update `Config` to prefer the quantized path if it exists.

### 6.2 IVF_PQ Vector Index (D4-3)

**Trigger**: Search latency >100ms with >1000 observations.

```python
# In lance_store.py
def create_vector_index(self):
    table = self._tables["observations_vec"]
    table.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=16,
        num_sub_vectors=32,
    )
```

LanceDB auto-creates index when table exceeds a threshold. Manual creation forces it earlier.

### 6.3 Corrupt Database Recovery (X-3, X-4)

```python
@click.command()
@click.pass_context
def doctor(ctx):
    """Check and repair database integrity."""
    db_path = ctx.obj["db_path"]

    # SQLite integrity check
    conn = sqlite3.connect(db_path)
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if result == "ok":
        console.print("[green]âœ“[/green] SQLite: integrity OK")
    else:
        console.print(f"[red]âœ—[/red] SQLite: {result}")
        console.print("  Recovery: copy undamaged data with: sqlite3 old.db '.dump' | sqlite3 new.db")

    # LanceDB check
    lance_path = Path(ctx.obj["config"].data_dir) / "lance"
    if lance_path.exists():
        try:
            import lancedb
            db = lancedb.connect(str(lance_path))
            tables = db.table_names()
            console.print(f"[green]âœ“[/green] LanceDB: {len(tables)} tables")
        except Exception as e:
            console.print(f"[red]âœ—[/red] LanceDB: {e}")
            console.print("  Recovery: delete ~/.claude-mem/lance/ and run: claude-mem reembed")
    else:
        console.print("[yellow]â—‹[/yellow] LanceDB: not initialized yet")

    conn.close()
```

---

## 7. CLI Command Tree (Complete)

After Phase 9 Tier 1, the full command tree:

```
claude-mem
â”œâ”€â”€ status              Quick health check (NEW - Phase 9)
â”œâ”€â”€ report              Session report
â”œâ”€â”€ search <query>      Hybrid search
â”œâ”€â”€ mermaid [file]      Call graph diagram
â”œâ”€â”€ eval                Eval framework (Phase 7)
â”‚   â”œâ”€â”€ compression
â”‚   â”œâ”€â”€ benchmark
â”‚   â””â”€â”€ health
â”œâ”€â”€ compress --pending  Manual catch-up compression (NEW - Phase 9)
â”œâ”€â”€ prune               Cleanup old data (NEW - Phase 9)
â”œâ”€â”€ install-hooks       Register Claude Code hooks (Phase 1)
â”œâ”€â”€ doctor              Database integrity check (NEW - Phase 9 Tier 3)
â””â”€â”€ reembed             Re-embed missing observations (NEW - Phase 9 Tier 2)
```

---

## 8. Test Plan

### 8.1 Tier 1 tests

| Category | Tests | What it validates |
|----------|-------|-------------------|
| **status command** | 4 | Empty DB, running worker, pending items, error items |
| **compress --pending** | 3 | Dry-run shows table, worker delegation, inline fallback |
| **prune** | 6 | Dry-run counts, raw_output cleanup, event_log cleanup, keep-raw preservation, vacuum reduces size, empty prune |
| **pyproject.toml** | 2 | `pip install -e .` succeeds, `pip install -e ".[worker,dev]"` succeeds |
| **Total Tier 1** | **15** | |

### 8.2 Tier 2 tests

| Category | Tests | What it validates |
|----------|-------|-------------------|
| **tiered compression** | 2 | Model selection by operation type, config override |
| **reembed** | 3 | Missing detection, batch embedding, idempotency |
| **auto eval** | 2 | Triggers every N observations, logs results |
| **cross-file resolution** | 3 | Import mapping, cross-file edge resolution, incremental update |
| **Total Tier 2** | **10** | |

---

## 9. Acceptance Criteria

### Tier 1 (production-ready)

- [ ] `pip install -e ".[worker,dev]"` succeeds from clean virtualenv
- [ ] `claude-mem --version` displays version from pyproject.toml
- [ ] `claude-mem status` shows DB state, worker state, queue state, hooks state
- [ ] `claude-mem compress --pending --dry-run` shows pending items table
- [ ] `claude-mem compress --pending` processes items (worker or inline)
- [ ] `claude-mem prune --older-than 30d --dry-run` shows deletion plan
- [ ] `claude-mem prune --older-than 30d` clears raw_output, preserves 100 newest
- [ ] `claude-mem prune --vacuum` reclaims disk space
- [ ] `README.md` covers installation, quick start, all commands, cost
- [ ] All 15 Tier 1 tests pass
- [ ] `ruff check` and `ruff format --check` pass

### Tier 2 (quality improvements)

- [ ] Tiered compression uses configured model per operation type
- [ ] `claude-mem reembed` re-indexes missing observations
- [ ] Deterministic eval auto-runs every 20 compressions
- [ ] Quality alerts logged when structural_validity drops below 0.9
- [ ] All 10 Tier 2 tests pass

---

## 10. Performance Targets

| Operation | Target |
|-----------|--------|
| `status` | <200ms (5 SQLite queries + worker health check) |
| `compress --pending` (per item, inline) | <3s (API call) |
| `prune --older-than 30d` (1000 items) | <1s |
| `prune --vacuum` | <10s for 500MB DB |
| `reembed` (per observation) | <200ms (embedding only) |
| Auto deterministic eval (20 obs) | <50ms (no API calls) |

---

## 11. Known Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **VACUUM locks DB during active session** | Medium | Medium | Document: only run prune --vacuum when no session is active. Add warning in CLI. |
| **Inline compression slow without worker** | Medium | Low | Expected: ~3s per item vs ~200ms async. Acceptable for catch-up of <50 items. |
| **pyproject.toml dependency conflicts** | Low | Medium | Pin minimum versions, not exact. Test in clean venv. |
| **ONNX export fails for Qwen3-Embedding** | Low | Low | Tier 3 item â€” only attempt if needed. Fallback: keep PyTorch FP32. |
| **Prune deletes raw_output needed for active benchmark** | Low | Medium | `--keep-raw 100` default preserves eval corpus. Benchmark uses most recent samples. |

---

## 12. Open Questions

| Question | Current assumption | When to resolve |
|----------|-------------------|-----------------|
| **Should prune be scheduled automatically?** | No â€” explicit `claude-mem prune` only. Automatic deletion of user data is risky. | After 3+ months of daily use. If users consistently forget to prune, add a warning in `status`. |
| **Should we support `config.toml` for all settings?** | Currently config is in Python (`config.py`). A TOML file would let users customize without code changes. | Tier 2 if users request it. Most settings (model, budget, thresholds) are fine as defaults. |
| **Should `doctor` auto-repair or just diagnose?** | Diagnose only. Auto-repair of databases is dangerous. | Keep as-is unless corruption becomes a pattern. |
| **Should Tier 1 ship as a single release?** | Yes â€” `v0.1.0` includes all Tier 1 items. | After all Tier 1 acceptance criteria pass. |
