# PRD Amendments: Phase 9 (Post-Review)

**Date**: 2026-02-08
**Triggered by**: External review of Phase 9 PRD (3 comments, 2 actionable, 1 deferred)
**Affects**: Phase 9 v1 (Hardening + Polish)

---

## Amendment 1: Safe Inline Compression â€” Dependency Isolation

**Severity**: High
**Affects**: Phase 9, Section 4.4 (`compress --pending`), Section 4.1 (`pyproject.toml`)

### Problem

When a user installs the CLI-only package (`pip install claude-mem-lite` without `[worker]`), the `compress --pending` command follows this path:

1. Worker is not running (obviously â€” it's not installed).
2. Code falls through to `_compress_inline(config, pending)`.
3. `_compress_inline` is unspecified in the PRD â€” it's referenced but never defined.

The implementation hazard is in the import chain. The Phase 3 processor (`worker/processor.py`) orchestrates compress â†’ store â†’ embed as a single pipeline (see Phase 4, Section 2.7). If `_compress_inline` imports from the processor module, it inherits module-level or init-time imports of:

- `lancedb` (in `[worker]` extras)
- `sentence-transformers` (in `[worker]` extras)
- `aiosqlite` (in `[worker]` extras)

Any of these cause `ImportError` on a CLI-only install.

**Clarification on the crash path**: The `Compressor` class itself (Phase 3, `compressor.py`) only depends on `anthropic` and `pydantic` â€” both in base deps. Compression alone is safe. The danger is the *orchestration after compression*: the processor's `process_item()` calls `self.lance_store.add_observation()` (Phase 4, Section 2.7), which requires `lancedb` and `sentence-transformers`.

A second, subtler issue: the `Compressor` is designed to use `anthropic.AsyncAnthropic()` in the worker's async context. The inline CLI path is synchronous â€” it needs `anthropic.Anthropic()` (sync client). The PRD's "implementation note" (line 438) mentions this but doesn't spec the actual function.

### Specification Change

Add a fully specified `_compress_inline` function to Section 4.4. This function lives in the CLI module, **not** in `worker/processor.py`, to avoid transitive import contamination.

```python
# In cli/compress_cmd.py â€” NOT in worker/

def _compress_inline(config, pending_items):
    """
    Run compression synchronously when worker is unavailable.

    CRITICAL: This function must NOT import from worker.processor,
    search.lance_store, or search.embedder. Those modules pull in
    worker-only dependencies (lancedb, sentence-transformers, torch).

    Compression only â€” embedding is skipped and left for the worker
    to catch up on next startup (via backfill_embeddings).
    """
    import sqlite3
    try:
        from claude_mem_lite.worker.compressor import Compressor
    except ImportError:
        click.echo(
            "[red]Error: Cannot compress â€” 'anthropic' package not found.[/red]\n"
            "Install with: pip install claude-mem-lite[worker]"
        )
        return

    # Sync client â€” Compressor normally uses AsyncAnthropic,
    # but build_compression_prompt + JSON parsing is model-agnostic.
    # We call the Anthropic API directly with the sync client.
    import anthropic
    client = anthropic.Anthropic()

    from claude_mem_lite.worker.prompts import build_compression_prompt

    db_path = config.db_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    processed = 0
    errors = 0

    for item in pending_items:
        item_id = item["id"]
        try:
            # Fetch full raw_output
            row = conn.execute(
                "SELECT raw_output, tool_name, files_touched FROM pending_queue WHERE id = ?",
                (item_id,),
            ).fetchone()

            if not row or not row["raw_output"]:
                continue

            # Build prompt and call API (synchronous)
            prompt = build_compression_prompt(
                raw_output=row["raw_output"],
                tool_name=row["tool_name"],
                files_touched=row["files_touched"],
            )
            response = client.messages.create(
                model=config.compression_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response (reuse compressor's parsing logic)
            import json
            text = response.content[0].text
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            compressed = json.loads(text)

            # Store observation in SQLite
            import hashlib
            obs_id = hashlib.md5(
                f"{item['session_id']}:{item_id}".encode()
            ).hexdigest()

            conn.execute(
                """INSERT OR REPLACE INTO observations
                   (id, session_id, title, summary, detail,
                    tool_name, files_touched, functions_changed,
                    raw_size_bytes, compressed_tokens, embedding_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
                (
                    obs_id,
                    item["session_id"],
                    compressed.get("title", "Untitled"),
                    compressed.get("summary", ""),
                    compressed.get("detail"),
                    row["tool_name"],
                    json.dumps(compressed.get("files_touched", [])),
                    json.dumps(compressed.get("functions_changed", [])),
                    len(row["raw_output"]),
                    response.usage.output_tokens,
                ),
            )

            # Mark queue item as done
            conn.execute(
                "UPDATE pending_queue SET status = 'done' WHERE id = ?",
                (item_id,),
            )
            conn.commit()
            processed += 1
            click.echo(f"  âœ“ {item_id[:8]} â†’ {compressed.get('title', '?')}")

        except Exception as e:
            # Mark as error, don't crash the loop
            conn.execute(
                "UPDATE pending_queue SET status = 'error', attempts = attempts + 1 WHERE id = ?",
                (item_id,),
            )
            conn.commit()
            errors += 1
            click.echo(f"  âœ— {item_id[:8]}: {e}")

    conn.close()
    click.echo(f"\nProcessed {processed}, errors {errors}")
    if processed > 0:
        click.echo(
            "[dim]Note: Embeddings skipped (run worker to index for semantic search).[/dim]"
        )
```

**Key design decisions:**

1. **`embedding_status='pending'`** â€” Observations compressed inline are stored without embeddings. The worker's `backfill_embeddings()` (Phase 4, Section 4) will index them on next startup.

2. **No import from `worker.processor`** â€” The function uses `Compressor` for the prompt template only, and calls the Anthropic API directly with the sync client. No transitive dependency on `lancedb`, `sentence-transformers`, or `aiosqlite`.

3. **`anthropic` import guarded with `try/except`** â€” Although `anthropic` is in base deps per the Phase 9 `pyproject.toml`, this guard is defensive. If the dependency structure changes, the error message tells the user exactly what to do.

4. **Sync `anthropic.Anthropic()`** â€” Not `AsyncAnthropic`. The CLI runs synchronously.

### Test Addition

Add to Section 8.1 test table:

| Category | Tests | What it validates |
|----------|-------|-------------------|
| **compress inline deps** | 1 | `_compress_inline` importable without `[worker]` extras installed |

Test implementation:

```python
def test_compress_inline_no_worker_deps():
    """Verify _compress_inline doesn't import worker-only dependencies."""
    import subprocess
    result = subprocess.run(
        [
            "python", "-c",
            "from claude_mem_lite.cli.compress_cmd import _compress_inline; print('ok')"
        ],
        capture_output=True, text=True,
        # Run in an env without worker deps to truly validate
    )
    assert "ok" in result.stdout
```

**Note**: This test is most meaningful when run in a venv with only base deps installed. In a full `[worker,dev]` install, it'll pass trivially. Consider adding a CI matrix entry for `pip install -e .` (base only).

### Updates to Phase 9 Sections

**Section 8.1**: Update compress --pending test count from 3 to 4 (add inline dependency isolation test).

**Section 11 (Known Risks)**: Add row:

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Inline compression imports worker deps** | High (if not guarded) | High (ImportError crash) | `_compress_inline` lives in CLI module, only imports `compressor` and `prompts`. No processor/lance/embedder imports. |

---

## Amendment 2: VACUUM Failure Handling â€” Reactive Over Proactive

**Severity**: Medium
**Affects**: Phase 9, Section 4.5 (`prune --vacuum`)

### Problem

SQLite `VACUUM` requires an exclusive lock on the database. If the worker is concurrently holding a connection (processing a queue item, running a health check, etc.), the `VACUUM` will fail with `SQLITE_BUSY`.

The PRD identifies this risk in Section 11 ("VACUUM locks DB during active session" â€” Medium/Medium) with the mitigation "Document: only run prune --vacuum when no session is active. Add warning in CLI."

The external review proposed a proactive check: call `_discover_worker()` before `VACUUM`, refuse if the worker is running. This has two problems:

1. **Race condition**: The worker could start between the check and the `VACUUM` call. Or another process (a hook script, a parallel CLI invocation) could hold the DB.

2. **Coupling to `_discover_worker`**: That function has its own cross-platform concerns (UDS socket checking, per Phase 8 Section 5.3). Adding it as a prerequisite for `VACUUM` means VACUUM inherits those platform issues.

A reactive approach â€” attempt `VACUUM`, catch the failure, provide a clear message â€” is both simpler and more robust. It handles *all* lock contention sources, not just the worker.

### Specification Change

Replace the `VACUUM` block in Section 4.5 (lines 545â€“551):

```python
    # BEFORE
    if vacuum:
        db_size_before = Path(db_path).stat().st_size
        console.print("[dim]Running VACUUM (this may take a moment)...[/dim]")
        conn.execute("VACUUM")
        db_size_after = Path(db_path).stat().st_size
        saved = (db_size_before - db_size_after) / (1024 * 1024)
        console.print(f"[green]âœ“[/green] VACUUM complete. Reclaimed {saved:.1f}MB")

    # AFTER
    if vacuum:
        db_size_before = Path(db_path).stat().st_size
        console.print("[dim]Running VACUUM (this may take a moment)...[/dim]")
        try:
            conn.execute("VACUUM")
            db_size_after = Path(db_path).stat().st_size
            saved = (db_size_before - db_size_after) / (1024 * 1024)
            console.print(f"[green]âœ“[/green] VACUUM complete. Reclaimed {saved:.1f}MB")
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                console.print(
                    "[red]âœ—[/red] VACUUM failed â€” database is locked.\n"
                    "  The worker or another process may be using the database.\n"
                    "  Stop the worker first: claude-mem-worker stop\n"
                    "  Then retry: claude-mem prune --vacuum"
                )
            else:
                raise  # Re-raise unexpected OperationalErrors
```

**Why reactive is better than proactive:**

| Approach | Handles worker contention | Handles hook script contention | Handles unknown processes | Race-condition free |
|----------|--------------------------|-------------------------------|--------------------------|---------------------|
| Proactive (`_discover_worker` check) | âœ“ | âœ— | âœ— | âœ— |
| Reactive (`try/except`) | âœ“ | âœ“ | âœ“ | âœ“ |

**WAL mode context**: The project uses WAL mode (Phase 0), which allows concurrent readers during normal operations. `VACUUM` is special â€” it must switch to rollback journal mode temporarily, which requires exclusive access. With WAL mode's `busy_timeout` (configured in Phase 0 at 5000ms), SQLite will wait up to 5 seconds for the lock before raising `SQLITE_BUSY`. In most cases, the worker's brief transactions will complete within that window. The error path is a safety net for sustained contention (e.g., worker processing a large batch).

### Test Addition

Update Section 8.1 prune tests â€” change from 6 to 7:

| Category | Tests | What it validates |
|----------|-------|-------------------|
| **prune** | 7 | Dry-run counts, raw_output cleanup, event_log cleanup, keep-raw preservation, vacuum reduces size, vacuum-while-locked shows error, empty prune |

Test implementation:

```python
def test_prune_vacuum_locked(tmp_db):
    """VACUUM with locked DB shows helpful error, doesn't crash."""
    # Hold an exclusive lock from another connection
    blocker = sqlite3.connect(tmp_db)
    blocker.execute("BEGIN EXCLUSIVE")

    runner = CliRunner()
    result = runner.invoke(prune, ["--vacuum"], obj={"db_path": tmp_db, "config": config})

    assert result.exit_code == 0  # Graceful, not a crash
    assert "database is locked" in result.output
    assert "claude-mem-worker stop" in result.output

    blocker.execute("ROLLBACK")
    blocker.close()
```

### Updates to Phase 9 Sections

**Section 11 (Known Risks)**: Update the VACUUM row:

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **VACUUM locks DB during active session** | Medium | **Low** (graceful error) | `try/except` catches `SQLITE_BUSY`, prints actionable message. WAL mode's `busy_timeout` handles brief contention automatically. |

---

## Amendment 3: Cross-Platform Worker Discovery â€” Deferred, Not Patched

**Severity**: Low (informational â€” no spec change)
**Affects**: Phase 9, Section 4.3 (`_discover_worker`), Phase 3 (`lifecycle.py`)

### Problem Statement (from review)

`_discover_worker` uses UDS socket detection (`Path(socket_path).exists()`), which doesn't work on Windows. The review proposed adding `os.name == "nt"` branching to check PID files on Windows and sockets on Unix.

### Assessment: Patch Is Insufficient, Deferral Is Correct

The `_discover_worker` socket check is one of **at least five** Unix-specific patterns in the codebase:

| Component | Unix-specific pattern | Windows equivalent needed |
|-----------|----------------------|--------------------------|
| `lifecycle.py` (Phase 3) | `subprocess.Popen` with `start_new_session=True` | `CREATE_NEW_PROCESS_GROUP` + `DETACHED_PROCESS` flags |
| `lifecycle.py` (Phase 3) | `os.kill(pid, signal.SIGTERM)` for graceful stop | `TerminateProcess()` or named event signaling |
| `capture-hook.py` (Phase 1) | UDS socket for hook â†’ worker HTTP | Named Pipes or TCP localhost |
| `context-hook.py` (Phase 5) | `socket.AF_UNIX` for context retrieval | Named Pipes or TCP localhost |
| `_discover_worker` (Phase 8) | `Path(socket_path).exists()` | Named Pipe existence check or TCP probe |

Patching only `_discover_worker` gives false confidence â€” the tool would detect the worker but fail to *communicate* with it (hooks still use `AF_UNIX`), and can't *manage* it (`SIGTERM` doesn't exist on Windows).

### Decision

**No specification change.** Windows support, if needed, requires a dedicated effort covering all five components above. This is correctly out of scope for Phase 9's "hardening existing functionality" mandate.

If Windows support becomes a requirement, the approach should be:

1. Replace UDS with TCP `localhost:PORT` (or Named Pipes) across all components
2. Replace signal-based lifecycle with a cross-platform mechanism (TCP command channel or event file)
3. Test on Windows CI

This is a new feature, not a hardening item. Track it as a potential future Phase 10 item if the need arises.

### Section Update

Add to Section 12 (Open Questions):

| Question | Current assumption | When to resolve |
|----------|-------------------|-----------------|
| **Should we support Windows?** | No â€” all UDS/signal/daemon patterns are Unix-specific. The tool targets macOS and Linux development environments. | Only if a concrete Windows use case arises. Requires dedicated effort across 5+ components, not a Phase 9 patch. |

---

## Summary

| Amendment | Severity | Nature | Lines affected |
|-----------|----------|--------|----------------|
| 1. Inline compression dependency isolation | High | New specification (fully defined `_compress_inline`) | Section 4.4, 8.1, 11 |
| 2. VACUUM reactive error handling | Medium | Replace unguarded `VACUUM` with `try/except` | Section 4.5, 8.1, 11 |
| 3. Cross-platform worker discovery | Low | No spec change â€” documented deferral rationale | Section 12 |
