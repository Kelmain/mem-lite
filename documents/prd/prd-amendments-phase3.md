# PRD Amendments: Phase 3 (Post-Review)

**Date**: 2026-02-08
**Triggered by**: External review of Phase 3 PRD (4 comments)
**Affects**: Phase 3 v1 (Worker Service + Compression)

---

## Amendment 1: Replace `os.fork()` with `subprocess.Popen`

**Severity**: Critical
**Affects**: Phase 3, Section 2.7 (`lifecycle.py`)

### Problem

The PRD uses `os.fork()` to daemonize the worker (lines 498â€“508). This is unsafe in a process that uses threaded libraries:

- **aiosqlite** runs a dedicated background thread per connection for its request queue
- **httpx** (used internally by `AsyncAnthropic`) maintains an internal connection pool with threads
- **uvloop** (installed via `uvicorn[standard]`) may initialize threads at import time

When `os.fork()` is called, only the calling thread survives in the child process. All other threads die, but their locks are inherited in their locked state. Any subsequent attempt to acquire those locks deadlocks the child.

This is not a theoretical concern:

1. Python 3.12+ emits `DeprecationWarning` when fork is used with active threads
2. Python 3.14 (our target) no longer uses `fork` as the default multiprocessing start method on any platform â€” the CPython core team formally recognized fork+threads as unsafe
3. aiosqlite v0.22.0 (our pinned version) changed `Connection` to no longer inherit from `threading.Thread`, but still spawns a background thread for its request queue â€” this thread would be killed by fork

The PRD's risk table (Section 8) classifies this as "Low likelihood" â€” this is incorrect. The worker's `lifespan()` creates an aiosqlite connection and an `AsyncAnthropic` client (which initializes httpx) during startup. However, the `os.fork()` call happens in `WorkerLifecycle.start()` which is called from the CLI entry point. The fork happens *before* any of these are initialized (they're created in the FastAPI lifespan, which runs after uvicorn starts in the child). The real danger is more subtle: if `WorkerLifecycle` is ever imported or called from a context where any library has already spawned a thread (e.g., from within an existing async application, or if any import side-effect creates a thread), the fork will deadlock.

Regardless of the exact timing, using `os.fork()` in Python 3.14 for process daemonization is against the language's direction. `subprocess.Popen` is the correct modern approach.

### Specification Change

Replace the `start()` method in Section 2.7:

```python
# BEFORE (os.fork â€” removed)
def start(self, daemon: bool = True) -> int:
    if self.is_running():
        return self.get_pid()
    self._cleanup_stale_files()
    self.config.ensure_dirs()
    if daemon:
        pid = os.fork()
        if pid > 0:
            self._wait_for_socket(timeout=10)
            return pid
        os.setsid()
        self._write_pid(os.getpid())
        run_worker(self.config)
        self._cleanup_stale_files()
        os._exit(0)
    else:
        self._write_pid(os.getpid())
        run_worker(self.config)
        self._cleanup_stale_files()
        return os.getpid()

# AFTER (subprocess.Popen â€” clean interpreter)
def start(self, daemon: bool = True) -> int:
    """Start worker process. Returns PID."""
    if self.is_running():
        return self.get_pid()

    self._cleanup_stale_files()
    self.config.ensure_dirs()

    if daemon:
        proc = subprocess.Popen(
            [sys.executable, "-m", "claude_mem_lite.worker.server"],
            cwd=str(self.config.base_dir),
            start_new_session=True,   # setsid equivalent â€” detach from terminal
            stdin=subprocess.DEVNULL,  # daemon has no terminal input
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._write_pid(proc.pid)
        self._wait_for_socket(timeout=10)
        return proc.pid
    else:
        # Foreground mode (for development/testing)
        self._write_pid(os.getpid())
        run_worker(self.config)
        self._cleanup_stale_files()
        return os.getpid()
```

### Additional Change: `server.py` must write its own PID

With `subprocess.Popen`, the PID is known immediately by the parent. However, the child process should also update the PID file on startup to handle the case where uvicorn's internal process management reassigns PIDs (unlikely but defensive):

```python
# In server.py lifespan, at startup:
config.pid_path.write_text(str(os.getpid()))
```

### Import Change

In `lifecycle.py`, replace:

```python
# BEFORE
import os
import signal
import subprocess
from pathlib import Path

# AFTER
import os
import signal
import subprocess
import sys
from pathlib import Path
```

Remove the `os.fork()` import dependency. `os._exit()` is no longer needed.

### Risk Table Update

Replace row in Section 8:

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ~~`os.fork()` issues on some systems~~ | ~~Low~~ | ~~Medium~~ | ~~Foreground mode as fallback~~ |
| **Worker daemon fails to start** | Low | Medium | `subprocess.Popen` with `start_new_session=True` spawns clean interpreter. Foreground mode (`--foreground`) as debugging fallback. Socket wait timeout provides clear error. |

### Open Questions Update

Remove from Section 9:

> ~~**`os.fork()` vs `subprocess.Popen()` for daemon?** | `os.fork()` â€” simpler, standard Unix daemon pattern. | Switch to subprocess if fork causes issues with asyncio event loop or signal handling.~~

Replace with:

> **`subprocess.Popen()` for daemon** | Resolved: `subprocess.Popen` with `start_new_session=True`. `os.fork()` is unsafe with threaded libraries (aiosqlite, httpx) and deprecated for multi-threaded contexts in Python 3.14. |

### Test Impact

Lifecycle tests (Section 6.1) remain valid â€” they test PID file creation, socket appearance, and SIGTERM behavior, which are unchanged. The test for "Double-start returns existing PID (idempotent)" still works because `is_running()` check precedes spawning.

### Design Rationale

- `subprocess.Popen` spawns a completely new Python interpreter with clean memory â€” no inherited thread locks, no stale event loops
- `start_new_session=True` is the modern equivalent of `os.setsid()` in a forked child
- `stdin/stdout/stderr=DEVNULL` detaches from the terminal properly
- The parent gets the PID immediately from `proc.pid` â€” no need for IPC
- Foreground mode remains unchanged for development/testing

---

## Amendment 2: Use Structured Outputs API for Compression

**Severity**: High
**Affects**: Phase 3, Section 2.5 (`compressor.py`), Section 2.6 (`prompts.py`), Section 2.2 (dependencies)

### Problem

The PRD uses prompt-based JSON extraction: the compression prompt instructs Haiku to "Return ONLY a JSON object" and the parser strips markdown fences. This approach has known failure modes with Haiku models:

1. Haiku adds conversational filler after JSON (e.g., `{"title": "..."} Hope this helps!`)
2. Haiku wraps JSON in markdown fences despite being told not to
3. Haiku occasionally produces unescaped newlines inside JSON string values
4. Missing or extra fields despite explicit schema in prompt

Production users have documented these issues. The PRD's `_parse_response` only handles fence-stripping (lines 360â€“366), which is insufficient.

**However**, the correct fix is not more robust parsing â€” it's eliminating the problem at the source. Structured Outputs is now GA for Claude Haiku 4.5 on the Anthropic API. This feature uses constrained decoding at the token level: the model cannot produce tokens that would violate the JSON schema. It guarantees schema-valid output on every call.

### Specification Change: Compressor

Replace the `compress()` method in Section 2.5:

```python
# BEFORE (prompt-based JSON)
async def compress(self, raw_output: str, tool_name: str, files_touched: str) -> CompressedObservation:
    prompt = build_compression_prompt(raw_output, tool_name, files_touched)
    try:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
    except ...
    text = response.content[0].text
    return self._parse_response(text, response.usage)

# AFTER (structured outputs â€” schema-guaranteed JSON)
async def compress(self, raw_output: str, tool_name: str, files_touched: str) -> CompressedObservation:
    """Compress raw tool output into structured observation.
    
    Uses Anthropic Structured Outputs API (GA) for guaranteed
    schema-valid JSON. Falls back to prompt-based parsing if
    structured outputs fails unexpectedly.
    """
    prompt = build_compression_prompt(raw_output, tool_name, files_touched)

    try:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": COMPRESSION_SCHEMA,
                }
            },
        )
    except (APIConnectionError, RateLimitError) as e:
        raise RetryableError(str(e)) from e
    except APIStatusError as e:
        if e.status_code >= 500:
            raise RetryableError(str(e)) from e
        raise NonRetryableError(str(e)) from e

    text = response.content[0].text
    return self._parse_response(text, response.usage)
```

### Specification Change: Schema Definition

Add to `compressor.py` (or `prompts.py`):

```python
# JSON Schema for structured outputs API â€” mirrors CompressedObservation model
COMPRESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Brief action summary, 5-10 words, imperative mood",
        },
        "summary": {
            "type": "string",
            "description": "What happened and why, 1-3 sentences",
        },
        "detail": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Technical details worth remembering. Null if summary is sufficient.",
        },
        "files_touched": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of file paths actually modified",
        },
        "functions_changed": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "name": {"type": "string"},
                    "action": {"type": "string", "enum": ["new", "modified", "deleted"]},
                },
                "required": ["file", "name", "action"],
                "additionalProperties": False,
            },
            "description": "Functions that were created, modified, or deleted",
        },
    },
    "required": ["title", "summary", "files_touched", "functions_changed"],
    "additionalProperties": False,
}
```

**Note**: The `detail` field uses `anyOf` with null to match the Pydantic model's `str | None` type. Structured outputs requires `additionalProperties: false` on all object types.

### Specification Change: Parser (Fallback)

Replace `_parse_response` with a simplified version that still handles edge cases as defense-in-depth:

```python
def _parse_response(self, text: str, usage) -> CompressedObservation:
    """Parse JSON response from Claude.
    
    With structured outputs, the response is guaranteed to be valid JSON
    matching our schema. The fallback parsing handles the rare case where
    structured outputs is unavailable or returns unexpected format.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from response (structured outputs
        # should prevent this, but defense-in-depth)
        data = self._extract_json_fallback(text)

    return CompressedObservation(
        title=data.get("title", "")[:200],
        summary=data.get("summary", "")[:1000],
        detail=data.get("detail"),
        files_touched=data.get("files_touched", []),
        functions_changed=data.get("functions_changed", []),
        tokens_in=usage.input_tokens,
        tokens_out=usage.output_tokens,
    )

def _extract_json_fallback(self, text: str) -> dict:
    """Last-resort JSON extraction when structured outputs fails.
    
    Strategy: strip markdown fences, then find first { to last }.
    If this fails, raise RetryableError (not NonRetryableError) to
    give the model a second chance.
    """
    cleaned = text.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # Find JSON object boundaries
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise RetryableError(f"Could not extract JSON from response: {text[:200]}")
```

### Error Classification Change

Update error classification table in Section 2.5:

| Error type | Classification | Action |
|-----------|---------------|--------|
| `json.JSONDecodeError` | ~~Non-retryable~~ **Retryable** (1 retry) | Model fluked â€” retry once. Mark error after 2 parse failures. |

This change applies to both the fallback parser and any unexpected parse failure. With structured outputs, `JSONDecodeError` should never occur, but if it does it's likely a transient issue worth retrying.

### Prompt Change

The compression prompt (Section 2.6) remains largely unchanged â€” the instructions still guide the model's content quality. However, remove the "Return ONLY a JSON object (no markdown fences, no explanation)" line since structured outputs enforces this at the token level:

```python
# BEFORE
"""...
Return ONLY a JSON object (no markdown fences, no explanation):
{{
  "title": ...
}}
...
"""

# AFTER
"""...
Return a JSON object with these fields:
{{
  "title": "Brief action summary, 5-10 words. Example: 'Added JWT auth middleware'",
  "summary": "What happened and why, 1-3 sentences. Include key decisions made.",
  "detail": "Technical details worth remembering across sessions. Null if summary is sufficient.",
  "files_touched": ["list", "of", "file/paths"],
  "functions_changed": [
    {{"file": "path/to/file.py", "name": "function_name", "action": "new|modified|deleted"}}
  ]
}}

Rules:
- title: imperative mood, no period. Focus on WHAT changed, not tool mechanics.
- summary: preserve WHY decisions were made, not just WHAT happened.
- detail: only include if there's non-obvious information. Omit for trivial operations.
- files_touched: only files actually modified, not just read.
- functions_changed: only if identifiable from the output. Empty array if unclear.
- If the tool output is a Read operation with no changes, set title to describe what was examined and summary to key findings.
"""
```

The schema in the prompt serves as guidance for content quality; the API schema enforces structural validity.

### Summarizer Impact

The same structured outputs approach should be applied to `Summarizer.summarize_session()` (Section 2.8). The session summary uses a simpler schema:

```python
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_files": {"type": "array", "items": {"type": "string"}},
        "key_decisions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "key_files", "key_decisions"],
    "additionalProperties": False,
}
```

### Changelog Addition

Add row to Phase 3 Changelog (top of PRD):

| Item | Architecture spec (may be outdated) | This PRD (verified Feb 2026) |
|------|-------------------------------------|------------------------------|
| JSON parsing strategy | Prompt-based ("Return ONLY JSON") + fence stripping | **Structured Outputs API** (`output_config.format`) â€” GA for Haiku 4.5 since Jan 2026. Constrained decoding guarantees schema-valid JSON. Fallback parser retained as defense-in-depth. |

### Test Impact

- **Compressor test: "JSON wrapped in fences"** â€” keep as fallback parser test, but note it should never trigger with structured outputs
- **Add test**: Valid structured output response â†’ `CompressedObservation` with correct fields (primary path)
- **Modify test**: `json.JSONDecodeError` â†’ now raises `RetryableError` (not `NonRetryableError`)
- **Add test**: Fallback parser extracts JSON from `{"title":"..."} Hope this helps!` trailing text

Updated test count: 28 â†’ **30** (+2 new compressor tests)

### Design Rationale

- Structured outputs eliminates an entire category of bugs at the API level â€” no amount of parsing can match token-level constrained decoding
- The schema definition is a single source of truth shared between the API call and the Pydantic model
- First request with a new schema incurs ~100-300ms compilation overhead (cached for 24 hours) â€” negligible for our use case
- Fallback parsing is defense-in-depth, not the primary path
- Reclassifying `JSONDecodeError` as retryable is correct because with structured outputs, a parse failure indicates a transient issue, not a fundamental model failure

---

## Amendment 3: Fix Orphan Recovery Race Condition

**Severity**: Medium
**Affects**: Phase 3, Section 2.9 (Graceful Degradation), Section 2.7 (`lifecycle.py`)

### Problem

The `recover_orphaned_items()` function (lines 681â€“692) resets queue items in `processing` state only if they are older than 5 minutes. This creates a race condition on restart:

1. Worker A is processing an item (e.g., a large 500KB tool output, API call takes ~10s)
2. User runs `claude-mem-worker restart`
3. Worker B starts, calls `recover_orphaned_items()`
4. The item is only 10 seconds old â€” doesn't meet the 5-minute threshold
5. Worker A is killed by SIGTERM
6. The item remains in `processing` state forever (until a restart >5 minutes later)

The 5-minute threshold was presumably intended to avoid resetting items that a running worker is actively processing. But on startup of a *new* worker, this concern doesn't apply â€” the old worker is dead (we just killed it or it crashed). Any item in `processing` state is by definition an orphan from the dead process.

### Specification Change: Orphan Recovery

Replace `recover_orphaned_items()` in Section 2.9:

```python
# BEFORE (time-based threshold â€” race-prone)
async def recover_orphaned_items(self):
    """Reset items stuck in 'processing' state from a previous crash."""
    five_min_ago = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
    await self.db.execute(
        """
        UPDATE pending_queue SET status = 'raw'
        WHERE status = 'processing' AND created_at < ?
        """,
        (five_min_ago,),
    )
    await self.db.commit()

# AFTER (reset ALL processing items â€” safe because old worker is dead)
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
        self.logger.log("processor.orphan_recovery", {
            "recovered_count": count,
        })
```

### Specification Change: Lifecycle `start()` Must Stop First

The orphan recovery is safe *only if* the old worker is guaranteed dead before the new one starts. The current `start()` method only checks `is_running()` â€” it doesn't actively stop the old worker. Fix this:

```python
# BEFORE
def start(self, daemon: bool = True) -> int:
    if self.is_running():
        return self.get_pid()
    ...

# AFTER
def start(self, daemon: bool = True) -> int:
    """Start worker process. Returns PID.
    
    If a worker is already running, returns its PID (idempotent).
    If a stale PID file exists but the process is dead, cleans up
    and starts a new worker.
    """
    existing_pid = self.get_pid()
    if existing_pid is not None:
        if self._is_pid_alive(existing_pid):
            return existing_pid  # Already running â€” idempotent
        # Stale PID file â€” process is dead, clean up
        self._cleanup_stale_files()

    self.config.ensure_dirs()
    ...
```

For explicit restart (e.g., `claude-mem-worker restart` CLI command), add a `restart()` method:

```python
def restart(self) -> int:
    """Stop existing worker and start a new one."""
    self.stop()  # Sends SIGTERM, waits up to 5s for exit
    return self.start(daemon=True)
```

And add the CLI command:

```python
elif cmd == "restart":
    pid = lifecycle.restart()
    print(f"Worker restarted (PID: {pid})")
```

### Degradation Table Update

Update the "Worker crashes mid-processing" row in Section 2.9:

| Failure mode | Behavior | Recovery |
|-------------|----------|---------|
| **Worker crashes mid-processing** | Items in `status='processing'` are orphaned | Startup recovery: reset **all** `processing` â†’ `raw`. Safe because `start()` ensures old worker is dead before new worker begins. |

### Acceptance Criteria Update

Replace line in Section 7:

> ~~- [ ] Orphaned `processing` items recovered on startup (reset to `raw` if >5min old)~~

With:

> - [ ] All orphaned `processing` items recovered on startup (reset to `raw` unconditionally â€” old worker guaranteed dead by lifecycle)

### Test Impact

- **Modify test**: "Orphaned 'processing' items from >5min ago â†’ reset to 'raw' on startup" becomes "All 'processing' items â†’ reset to 'raw' on startup" (simpler assertion, no time manipulation needed)
- **Add test**: Restart scenario â€” item in `processing` for <1 minute is still recovered after restart

Updated test count: 30 â†’ **31** (+1 new processor test, cumulative with Amendment 2)

### Design Rationale

- The single-worker constraint (PID file + lifecycle management) makes the time-based threshold unnecessary and harmful
- The race window was real: between SIGTERM and process exit, items could be mid-processing and not yet old enough for the threshold
- Resetting all `processing` items is safe because: (a) only one worker runs at a time, (b) `start()` ensures the old worker is dead, (c) SQLite's WAL mode handles any in-flight writes from the dying process
- The `restart()` method makes the stop-then-start sequence explicit and atomic from the user's perspective

---

## Amendment 4: Define Summarization Trigger in Worker

**Severity**: Medium
**Affects**: Phase 3, Section 2.4 (`processor.py`), Section 2.8 (`summarizer.py`), Phase 1 cross-reference

### Problem

The Phase 3 PRD defines the `Summarizer` class and the `/api/summarize` endpoint, but never specifies **what triggers summarization**. There is an integration gap:

- **Phase 1** `summary.py` (Stop hook) logs a `hook.stop` event and says "Phase 3 wires in POST /api/summarize to worker" (Phase 1 PRD, line 475)
- **Phase 3** implements the summarizer but the `Processor.run()` loop only processes `pending_queue` items (compression). No code path detects "session is ready for summary"
- **Result**: Summaries are never generated unless someone manually calls `POST /api/summarize`

### Design Decision: Keep Hooks Dumb

The hooks should remain fast, synchronous, and side-effect-minimal. Adding an HTTP call to the worker from a hook would:

1. Violate Phase 1's design principle ("direct SQLite writes, no worker dependency")
2. Add latency to Claude Code's stop event (HTTP round-trip over UDS)
3. Create a dependency on the worker being running during hook execution
4. Require error handling for worker unavailability in every hook

Instead, the Worker should detect when a session is ready for summarization during its normal polling loop.

### Specification Change: Add Summary Detection to Processor

Add a `_check_pending_summaries()` method to `Processor` (Section 2.4):

```python
class Processor:
    POLL_INTERVAL = 2.0
    BATCH_SIZE = 5
    MAX_ATTEMPTS = 3
    BACKOFF_BASE = 5.0
    SUMMARY_IDLE_MINUTES = 2  # Summarize session after 2 min of inactivity

    async def run(self):
        """Main processing loop â€” runs as asyncio task."""
        await self.recover_orphaned_items()
        while not self.idle_tracker.should_shutdown:
            items = await self.dequeue_batch()
            if items:
                for item in items:
                    self.idle_tracker.touch()
                    await self.process_item(item)
            else:
                # No compression work â€” check for sessions needing summary
                await self._check_pending_summaries()
                await asyncio.sleep(self.POLL_INTERVAL)

    async def _check_pending_summaries(self):
        """Detect sessions ready for summarization.

        A session is ready for summary when:
        1. It has a 'hook.stop' event logged (session activity ended)
        2. It has no remaining 'raw' or 'processing' items in pending_queue
        3. It has at least one observation (something to summarize)
        4. It hasn't been summarized yet (sessions.summary IS NULL)
        5. The last activity was >SUMMARY_IDLE_MINUTES ago (debounce â€”
           users sometimes trigger multiple stop events in quick succession)
        """
        idle_threshold = (
            datetime.now(timezone.utc) - timedelta(minutes=self.SUMMARY_IDLE_MINUTES)
        ).isoformat()

        cursor = await self.db.execute(
            """
            SELECT DISTINCT s.id
            FROM sessions s
            -- Has at least one observation
            INNER JOIN observations o ON o.session_id = s.id
            -- Has a stop event
            INNER JOIN event_log e ON e.session_id = s.id
                AND json_extract(e.data, '$.event') = 'hook.stop'
            WHERE s.summary IS NULL
              AND s.status != 'closed'
              -- No pending compression work for this session
              AND NOT EXISTS (
                  SELECT 1 FROM pending_queue pq
                  WHERE pq.session_id = s.id
                    AND pq.status IN ('raw', 'processing')
              )
              -- Debounce: last event was >N minutes ago
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
                self.logger.log("processor.summarized", {
                    "session_id": session_id,
                })
            except Exception as e:
                self.logger.log("processor.summarize_error", {
                    "session_id": session_id,
                    "error": str(e),
                })
```

### Specification Change: Processor Constructor

Update the `Processor` class to accept a `Summarizer` dependency:

```python
class Processor:
    def __init__(self, db, compressor, logger, idle_tracker, summarizer):
        self.db = db
        self.compressor = compressor
        self.logger = logger
        self.idle_tracker = idle_tracker
        self.summarizer = summarizer
```

Update the lifespan in Section 2.3 to wire this:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = await aiosqlite.connect(config.db_path)
    app.state.db.row_factory = aiosqlite.Row
    app.state.compressor = Compressor(config)
    app.state.summarizer = Summarizer(app.state.db, app.state.compressor)
    app.state.idle_tracker = IdleTracker(timeout_minutes=30)
    app.state.processor = Processor(
        app.state.db,
        app.state.compressor,
        logger,
        app.state.idle_tracker,
        app.state.summarizer,  # NEW
    )
    processor_task = asyncio.create_task(app.state.processor.run())
    yield
    processor_task.cancel()
    await app.state.compressor.close()
    await app.state.db.close()
```

### Phase 1 Cross-Reference Clarification

Phase 1's `summary.py` comment (line 475) should be updated to reflect the actual integration:

> ~~Phase 1: Just log. Phase 3 wires in POST /api/summarize to worker.~~

Replace with:

> Phase 1: Log `hook.stop` event. Phase 3's Worker polling loop detects this event and triggers summarization automatically when the session has no remaining pending queue items.

The Stop hook does NOT need to be modified. It already writes the `hook.stop` event to `event_log`, which is exactly what the Worker's `_check_pending_summaries()` looks for.

### Query Design Notes

The summary detection query is deliberately conservative:

- **`INNER JOIN observations`**: Don't summarize sessions with zero observations (nothing to summarize)
- **`NOT EXISTS pending_queue`**: Wait for all compression to finish before summarizing â€” the summary should include all observations
- **`e.created_at < idle_threshold`**: 2-minute debounce prevents summarizing too early if the user triggers stop/start rapidly
- **`LIMIT 3`**: Process at most 3 summaries per poll cycle to avoid blocking compression work
- **`s.status != 'closed'`**: Don't re-summarize sessions that were already processed by a `SessionEnd` event

### Test Impact

- **Add test**: Session with `hook.stop` event + completed observations â†’ summary generated
- **Add test**: Session with pending queue items â†’ not summarized yet (waits for completion)
- **Add test**: Session with no observations â†’ not summarized (nothing to summarize)

Updated test count: 31 â†’ **34** (+3 new processor/summarizer tests, cumulative)

### Acceptance Criteria Addition

Add to Section 7:

> - [ ] Worker automatically detects sessions needing summary (via `hook.stop` event + no pending queue items) and generates summary without external trigger
> - [ ] Summarization is debounced (2-minute idle threshold prevents premature summary on rapid stop/start)

---

## Summary of All Changes

### Phase 3 Code Changes

| Module | Change | Type | Amendment |
|--------|--------|------|-----------|
| `lifecycle.py` | Replace `os.fork()` with `subprocess.Popen` | Bug fix (Critical) | #1 |
| `lifecycle.py` | Add `restart()` method and CLI command | Enhancement | #3 |
| `lifecycle.py` | Add `sys` import, remove `os._exit()` usage | Cleanup | #1 |
| `server.py` | Write PID from child process on startup | Defensive | #1 |
| `compressor.py` | Add `COMPRESSION_SCHEMA` for structured outputs | Enhancement | #2 |
| `compressor.py` | Add `output_config` to `messages.create()` call | Enhancement | #2 |
| `compressor.py` | Replace `_parse_response` with structured outputs primary + fallback parser | Enhancement | #2 |
| `compressor.py` | Reclassify `JSONDecodeError` as `RetryableError` | Bug fix | #2 |
| `prompts.py` | Remove "Return ONLY a JSON object" instruction | Cleanup | #2 |
| `processor.py` | Replace time-based orphan recovery with unconditional reset | Bug fix (Medium) | #3 |
| `processor.py` | Add `_check_pending_summaries()` to polling loop | Feature (Medium) | #4 |
| `processor.py` | Accept `Summarizer` as constructor dependency | Enhancement | #4 |
| `summarizer.py` | Add `SUMMARY_SCHEMA` for structured outputs | Enhancement | #2 |

### Phase 3 Documentation Changes

| Section | Change | Amendment |
|---------|--------|-----------|
| Changelog table | Add structured outputs row | #2 |
| 2.4 | Add `_check_pending_summaries()` spec and `SUMMARY_IDLE_MINUTES` constant | #4 |
| 2.5 | Update `compress()` with `output_config`, new `_parse_response`, fallback `_extract_json_fallback` | #2 |
| 2.5 | Update error classification table â€” `JSONDecodeError` becomes retryable | #2 |
| 2.7 | Replace `start()` implementation, add `restart()` | #1, #3 |
| 2.9 | Update orphan recovery spec and degradation table | #3 |
| 6.1 | Update test descriptions for amended behavior | #2, #3 |
| 7 | Update acceptance criteria (orphan recovery, summarization trigger) | #3, #4 |
| 8 | Update risk table (`os.fork()` row) | #1 |
| 9 | Resolve `os.fork()` open question | #1 |

### Phase 1 Cross-Reference

| File | Line | Change | Amendment |
|------|------|--------|-----------|
| `summary.py` | 475 | Update comment: "Phase 3 Worker polling loop detects hook.stop" | #4 |

### Test Count

| Before | After | Delta |
|--------|-------|-------|
| 28 | 34 | +6 |

| New Test | Category | Amendment |
|----------|----------|-----------|
| Structured output response â†’ `CompressedObservation` | Compressor | #2 |
| Fallback parser extracts JSON from trailing text | Compressor | #2 |
| All `processing` items reset on startup (no time threshold) | Processor | #3 |
| Restart: recent `processing` item recovered | Processor | #3 |
| Session with `hook.stop` + observations â†’ auto-summarized | Processor/Summarizer | #4 |
| Session with pending queue items â†’ not summarized | Processor/Summarizer | #4 |
| Session with no observations â†’ not summarized | Processor/Summarizer | #4 |

| Modified Test | Change | Amendment |
|---------------|--------|-----------|
| `JSONDecodeError` error classification | Now expects `RetryableError` instead of `NonRetryableError` | #2 |
| Orphan recovery threshold | Removed time-based assertion, now unconditional | #3 |
