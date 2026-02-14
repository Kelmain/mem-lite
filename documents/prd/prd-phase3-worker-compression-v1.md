# PRD: Phase 3 â€” Worker Service + Compression (v1)

**Project**: claude-mem-lite (fork of claude-mem, Python rewrite)
**Phase**: 3 of 9
**Status**: Draft â€” pending review
**Dependencies**: Phase 0 (Storage Layer â€” SQLiteStore, pending_queue, event_log), Phase 1 (Hook Scripts â€” data flowing into pending_queue)
**Estimated effort**: 2â€“3 sessions (~10-16 hours)
**Python**: 3.14.3 (latest stable)

---

## Changelog from Architecture Spec

| Item | Architecture spec (may be outdated) | This PRD (verified Feb 2026) |
|------|-------------------------------------|------------------------------|
| FastAPI version | `>=0.115` | **>=0.128** (0.128.4 latest, released Feb 7 2026) |
| Anthropic SDK version | `>=0.40` | **>=0.78** (0.78.0 latest, requires Python 3.9+) |
| aiosqlite version | `>=0.20` | **>=0.22** (0.22.1 latest, Dec 2025 â€” note: Connection no longer inherits Thread since v0.22.0) |
| Compression model string | `claude-haiku-4-5` (shorthand) | **`claude-haiku-4-5-20251001`** (pinned snapshot for reproducibility; shorthand `claude-haiku-4-5` also valid) |
| Haiku 4.5 pricing | Not specified | **$1/1M input, $5/1M output** (confirmed Oct 2025 launch) |
| uvicorn UDS | `uvicorn.run(app, uds="...")` | Confirmed: `--uds` flag or `uds=` kwarg in `uvicorn.run()` |
| httpx UDS | "native UDS support" | Confirmed via `httpx.AsyncHTTPTransport(uds="/path/to/socket")` |
| Anthropic SDK async | `anthropic.AsyncAnthropic()` | Confirmed: `AsyncAnthropic().messages.create()`, SDK uses httpx internally |
| Worker process model | "Uvicorn single-process daemon, no Gunicorn" | Confirmed: single worker, `uvicorn.run()` programmatic API |
| aiosqlite thread model | "async wrapper via dedicated background thread" | Confirmed â€” single shared thread per connection, request queue prevents overlapping |
| Prompt engineering scope | "Capture 20-30 real tool outputs manually" | **Deferred to implementation time** â€” PRD defines prompt structure, quality criteria, and evaluation method |

---

## 1. Purpose & Context

### 1.1 What this phase delivers

The background worker that turns raw tool outputs into structured, searchable observations. This is the AI processing pipeline:

- **FastAPI worker service** over Unix Domain Socket â€” daemon with PID file, idle timeout, health endpoint
- **Queue processor** â€” polls `pending_queue`, orchestrates compression, handles retries
- **AI compressor** â€” calls Claude Haiku 4.5 via raw Anthropic SDK to compress 10KB-500KB tool outputs into ~500-token structured observations
- **Worker lifecycle management** â€” start/stop scripts, daemon mode, crash recovery
- **Graceful degradation** â€” API failures leave items in queue for retry; worker absence doesn't break hooks

### 1.2 What this phase does NOT deliver

- **Embedding generation** â€” deferred to Phase 4. Observations are stored in SQLite only.
- **LanceDB writes** â€” deferred to Phase 4. No vector storage yet.
- **Context injection** â€” deferred to Phase 5. Worker has a `/api/context` placeholder that returns empty.
- **Session summarization** â€” the Stop hook (Phase 1) marks sessions for summarization. This phase implements the `/api/summarize` endpoint that generates summaries from a session's observations.
- **A/B testing framework** â€” deferred to Phase 7. Single model (Haiku 4.5) for now.
- **Prompt optimization** â€” this PRD defines the initial compression prompt. Iteration on prompt quality happens during implementation and Phase 7.

### 1.3 Why this is the highest-risk phase

The implementation plan identifies Phase 3 as **highest risk** because compression quality determines the value of the entire system. Bad compression â†’ useless observations â†’ wasted context tokens â†’ Claude ignoring injected context.

The risk is mitigated by:
1. Defining clear quality criteria upfront (this PRD, Â§4)
2. Using structured output (JSON) with validation
3. Starting with Haiku 4.5 ($1/$5 per M tokens â€” cheap enough for iteration)
4. Logging raw inputs alongside compressed outputs for offline evaluation (Phase 7)

### 1.4 Relationship to claude-mem (original)

claude-mem uses:
- Express.js HTTP server on TCP port 37777 â€” we use FastAPI on UDS (no port conflicts)
- Agent SDK for compression â€” caused memory leaks (40GB+), zombie processes. We use raw `client.messages.create()`
- Bun worker process â€” we use uvicorn single-process daemon

### 1.5 Data flow

```
Phase 1 hooks write to pending_queue (status='raw')
         â”‚
         â–¼
Worker polls pending_queue â”€â”€â–º claims batch (status='processing')
         â”‚
         â–¼
Compressor calls Claude Haiku 4.5 API
         â”‚
         â”œâ”€â”€ Success: parse JSON â†’ INSERT observation â†’ UPDATE queue status='done'
         â”‚
         â””â”€â”€ Failure: INCREMENT attempts â†’ status='error' if max_attempts exceeded
                                         â†’ status='raw' if retryable
```

---

## 2. Technical Specification

### 2.1 Module Structure

```
src/claude_mem_lite/worker/
â”œâ”€â”€ __init__.py
â”œâ”€â”€ server.py          # FastAPI app + lifespan + uvicorn runner
â”œâ”€â”€ processor.py       # Queue consumer â€” poll, claim, orchestrate
â”œâ”€â”€ compressor.py      # AI compression â€” prompt, parse, validate
â”œâ”€â”€ prompts.py         # Compression prompt templates
â”œâ”€â”€ lifecycle.py       # Daemon management â€” start, stop, PID, health check
â””â”€â”€ summarizer.py      # Session summarization â€” aggregate observations â†’ summary
```

### 2.2 Dependencies (Phase 3 additions)

| Package | Version | Size | Purpose |
|---------|---------|------|---------|
| `fastapi` | >=0.128 | ~2MB (+ starlette ~1MB) | HTTP endpoints |
| `uvicorn` | >=0.30 | ~500KB | ASGI server with UDS support |
| `anthropic` | >=0.78 | ~2MB (+ httpx, pydantic, anyio) | Claude API for compression |
| `aiosqlite` | >=0.22 | ~50KB (pure Python) | Async SQLite for worker event loop |
| `httpx` | >=0.27 | ~1MB | Hooks â†’ worker HTTP calls over UDS |

**Note**: `httpx` is already a transitive dependency of `anthropic`. We list it explicitly because hooks use it directly for UDS connections to the worker.

**Updated `pyproject.toml` dependencies section**:
```toml
dependencies = [
    "pydantic>=2.0",
    "fastapi>=0.128",
    "uvicorn[standard]>=0.30",
    "anthropic>=0.78",
    "aiosqlite>=0.22",
    "httpx>=0.27",
]
```

**Why `uvicorn[standard]`**: Installs `uvloop` (faster event loop on Linux) and `httptools` (fast HTTP parser). Both are C extensions that provide ~2-4x event loop speedup. No cost for a local tool.

### 2.3 Worker Service (`server.py`)

#### FastAPI app with lifespan

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
import aiosqlite

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.db = await aiosqlite.connect(config.db_path)
    app.state.db.row_factory = aiosqlite.Row
    app.state.compressor = Compressor(config)
    app.state.processor = Processor(app.state.db, app.state.compressor, logger)
    app.state.idle_tracker = IdleTracker(timeout_minutes=30)

    # Start background queue processing
    processor_task = asyncio.create_task(app.state.processor.run())

    yield

    # Shutdown
    processor_task.cancel()
    await app.state.db.close()

app = FastAPI(lifespan=lifespan)
```

#### Endpoints

| Method | Path | Purpose | Response |
|--------|------|---------|----------|
| `GET` | `/api/health` | Liveness + basic stats | `{"status": "ok", "uptime_s": int, "queue_depth": int, "observations_today": int}` |
| `GET` | `/api/context` | Context injection (Phase 5) | `{"context": "", "tokens": 0}` (placeholder) |
| `POST` | `/api/summarize` | Generate session summary | `{"summary": str, "tokens": int}` |
| `GET` | `/api/queue/stats` | Queue status for debugging | `{"raw": int, "processing": int, "done": int, "error": int}` |

**Design decision: No `/api/compress` endpoint.** Compression is triggered by the background processor polling the queue, not by HTTP requests. This decouples hook latency from API call latency. The hooks write directly to SQLite (Phase 1), the worker processes asynchronously.

#### UDS binding

```python
import uvicorn

def run_worker(config: Config):
    uvicorn.run(
        "claude_mem_lite.worker.server:app",
        uds=str(config.socket_path),
        log_level="info",
        lifespan="on",
    )
```

**Socket file cleanup**: uvicorn automatically removes the socket file on clean shutdown. For crash recovery, the lifecycle manager checks for stale socket files on startup (Â§2.7).

#### Idle timeout

```python
class IdleTracker:
    """Track last activity, trigger shutdown after inactivity."""

    def __init__(self, timeout_minutes: int = 30):
        self.timeout = timeout_minutes * 60
        self.last_activity = time.monotonic()
        self._shutdown_event = asyncio.Event()

    def touch(self):
        self.last_activity = time.monotonic()

    async def watch(self):
        """Background task â€” sets shutdown event when idle timeout exceeded."""
        while True:
            await asyncio.sleep(60)  # check every minute
            elapsed = time.monotonic() - self.last_activity
            if elapsed >= self.timeout:
                self._shutdown_event.set()
                return

    @property
    def should_shutdown(self) -> bool:
        return self._shutdown_event.is_set()
```

**How shutdown works**: The idle watcher runs as a background task. When timeout fires, it signals the main process to exit gracefully. The processor's `run()` loop checks `idle_tracker.should_shutdown` on each iteration. When true, it stops polling and allows the lifespan context manager to run cleanup.

**Activity tracking**: Every HTTP request and every queue item processed calls `idle_tracker.touch()`. The queue processor touches on each poll cycle regardless of whether items were found.

### 2.4 Queue Processor (`processor.py`)

```python
class Processor:
    """Poll pending_queue, compress raw observations, store results."""

    POLL_INTERVAL = 2.0      # seconds between polls when queue is empty
    BATCH_SIZE = 5            # items per dequeue
    MAX_ATTEMPTS = 3          # retry limit per item
    BACKOFF_BASE = 5.0        # seconds â€” exponential: 5, 10, 20

    async def run(self):
        """Main processing loop â€” runs as asyncio task."""
        while not self.idle_tracker.should_shutdown:
            items = await self.dequeue_batch()
            if not items:
                await asyncio.sleep(self.POLL_INTERVAL)
                continue

            for item in items:
                self.idle_tracker.touch()
                await self.process_item(item)

    async def dequeue_batch(self) -> list[PendingQueueItem]:
        """Atomically claim items from pending_queue.

        Uses the same UPDATE ... RETURNING pattern as SQLiteStore.dequeue_batch()
        but through aiosqlite's async interface.
        """
        async with self.db.execute("BEGIN IMMEDIATE"):
            pass
        try:
            cursor = await self.db.execute(
                """
                UPDATE pending_queue SET status = 'processing'
                WHERE status = 'raw'
                  AND id IN (
                    SELECT id FROM pending_queue
                    WHERE status = 'raw'
                    ORDER BY created_at
                    LIMIT ?
                  )
                RETURNING *
                """,
                (self.BATCH_SIZE,),
            )
            rows = await cursor.fetchall()
            await self.db.execute("COMMIT")
        except Exception:
            await self.db.execute("ROLLBACK")
            raise
        return [PendingQueueItem(**dict(r)) for r in rows]

    async def process_item(self, item: PendingQueueItem):
        """Compress a single queue item and store the observation."""
        with self.logger.timed("compress.done", session_id=item.session_id,
                               raw_size=len(item.raw_output)):
            try:
                result = await self.compressor.compress(
                    raw_output=item.raw_output,
                    tool_name=item.tool_name,
                    files_touched=item.files_touched,
                )
                await self._store_observation(item, result)
                await self._mark_done(item.id)
            except RetryableError as e:
                await self._handle_retry(item, e)
            except NonRetryableError as e:
                await self._mark_error(item.id, str(e))

    async def _handle_retry(self, item: PendingQueueItem, error: Exception):
        """Exponential backoff retry logic."""
        new_attempts = item.attempts + 1
        if new_attempts >= self.MAX_ATTEMPTS:
            await self._mark_error(item.id, f"Max retries exceeded: {error}")
            return

        backoff = self.BACKOFF_BASE * (2 ** (new_attempts - 1))
        self.logger.log("compress.retry", {
            "item_id": item.id,
            "attempt": new_attempts,
            "backoff_s": backoff,
            "error": str(error),
        }, session_id=item.session_id)

        # Reset to 'raw' with incremented attempt count
        await self.db.execute(
            "UPDATE pending_queue SET status = 'raw', attempts = ? WHERE id = ?",
            (new_attempts, item.id),
        )
        await self.db.commit()
```

**Design decisions:**

1. **Poll interval 2s**: Balances responsiveness with CPU usage. At ~5 req/s peak, items spend <2s in queue. Negligible for a background process.

2. **Batch size 5**: Process up to 5 items per poll cycle. Prevents API rate limiting during bursts while still clearing backlogs efficiently. Haiku 4.5 has generous rate limits but we're courteous.

3. **Retry strategy**: Exponential backoff (5s, 10s, 20s). After 3 attempts, item is marked `error`. Retryable errors: API timeouts, rate limits, transient network. Non-retryable: invalid JSON response, prompt validation failure.

4. **aiosqlite transaction handling**: aiosqlite doesn't support `async with db.execute("BEGIN")` natively â€” we execute BEGIN/COMMIT/ROLLBACK as separate statements. This matches aiosqlite's documented usage pattern.

### 2.5 AI Compressor (`compressor.py`)

#### Anthropic SDK usage

```python
from anthropic import AsyncAnthropic, APIConnectionError, RateLimitError, APIStatusError

class Compressor:
    def __init__(self, config: Config):
        self.client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = config.compression_model  # "claude-haiku-4-5-20251001"

    async def compress(
        self,
        raw_output: str,
        tool_name: str,
        files_touched: str,
    ) -> CompressedObservation:
        """Compress raw tool output into structured observation."""
        prompt = build_compression_prompt(raw_output, tool_name, files_touched)

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
        except (APIConnectionError, RateLimitError) as e:
            raise RetryableError(str(e)) from e
        except APIStatusError as e:
            if e.status_code >= 500:
                raise RetryableError(str(e)) from e
            raise NonRetryableError(str(e)) from e

        text = response.content[0].text
        return self._parse_response(text, response.usage)

    def _parse_response(self, text: str, usage) -> CompressedObservation:
        """Parse and validate JSON response from Claude."""
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise NonRetryableError(f"Invalid JSON response: {e}\nRaw: {text[:200]}")

        # Validate required fields
        return CompressedObservation(
            title=data.get("title", "")[:200],
            summary=data.get("summary", "")[:1000],
            detail=data.get("detail"),
            files_touched=data.get("files_touched", []),
            functions_changed=data.get("functions_changed", []),
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
        )

    async def close(self):
        """Clean up HTTP client."""
        await self.client.close()
```

**Error classification:**

| Error type | Classification | Action |
|-----------|---------------|--------|
| `APIConnectionError` | Retryable | Network issue â€” exponential backoff |
| `RateLimitError` | Retryable | Rate limited â€” exponential backoff |
| `APIStatusError` 5xx | Retryable | Server error â€” exponential backoff |
| `APIStatusError` 4xx | Non-retryable | Bad request, auth failure â€” mark error |
| `json.JSONDecodeError` | Non-retryable | Model returned invalid JSON â€” mark error, log raw response |
| Missing required fields | Non-retryable | Model omitted title/summary â€” mark error |

**Why no streaming**: Compression calls produce small outputs (~500 tokens). Streaming adds complexity with no benefit. A non-streaming call completes in ~500-1500ms for Haiku 4.5.

**Why no system prompt**: System prompts add to input token costs on every call. The user message contains all instructions. For a structured extraction task, this is sufficient and cheaper.

### 2.6 Compression Prompt (`prompts.py`)

```python
COMPRESSION_PROMPT_V1 = """\
Compress this Claude Code tool output into a structured observation.

Tool: {tool_name}
Files: {files_touched}

<tool_output>
{raw_output}
</tool_output>

Return ONLY a JSON object (no markdown fences, no explanation):
{{
  "title": "Brief action summary, 5-10 words. Example: 'Added JWT auth middleware'",
  "summary": "What happened and why, 1-3 sentences. Include key decisions made.",
  "detail": "Technical details worth remembering across sessions. Include function names, patterns used, gotchas encountered. Null if summary is sufficient.",
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

def build_compression_prompt(
    raw_output: str,
    tool_name: str,
    files_touched: str,
) -> str:
    """Build compression prompt, truncating raw_output if too large."""
    # Haiku 4.5 has 200K context. We send at most 8K tokens (~32KB) of raw output.
    # Larger outputs are truncated with a marker.
    MAX_RAW_CHARS = 32_000
    truncated = raw_output
    if len(raw_output) > MAX_RAW_CHARS:
        half = MAX_RAW_CHARS // 2
        truncated = (
            raw_output[:half]
            + f"\n\n[... truncated {len(raw_output) - MAX_RAW_CHARS} chars ...]\n\n"
            + raw_output[-half:]
        )

    return COMPRESSION_PROMPT_V1.format(
        tool_name=tool_name,
        files_touched=files_touched,
        raw_output=truncated,
    )
```

**Prompt design rationale:**

1. **No system prompt**: Saves tokens. For structured extraction, a single user message works fine.
2. **Explicit JSON schema with examples**: Reduces malformed responses. The "Example:" in title field is a concrete guide.
3. **"No markdown fences" instruction**: Haiku occasionally wraps JSON in ` ```json `. The parser handles this, but the instruction reduces frequency.
4. **Imperative mood for title**: Matches git commit conventions. Reads naturally in context injection: "Added JWT auth middleware" vs "JWT auth middleware was added".
5. **WHY over WHAT for summary**: The most useful information for future sessions is decision rationale, not mechanical actions.
6. **Truncation at 32K chars (~8K tokens)**: Keeps input costs bounded. Head+tail truncation preserves the beginning (context) and end (results/errors) of tool outputs.

**Cost estimate**: At $1/1M input tokens, 8K input tokens per call costs $0.008. With 50 tool calls per session: ~$0.40/session. At $5/1M output, 500 output tokens: $0.0025/call â†’ $0.125/session. Total: **~$0.50-0.60 per session**.

### 2.7 Worker Lifecycle (`lifecycle.py`)

```python
import os
import signal
import subprocess
from pathlib import Path

class WorkerLifecycle:
    """Manage worker daemon: start, stop, status, health check."""

    def __init__(self, config: Config):
        self.config = config
        self.pid_path = config.pid_path
        self.socket_path = config.socket_path

    def start(self, daemon: bool = True) -> int:
        """Start worker process. Returns PID."""
        if self.is_running():
            return self.get_pid()

        self._cleanup_stale_files()
        self.config.ensure_dirs()

        if daemon:
            # Fork and detach
            pid = os.fork()
            if pid > 0:
                # Parent: wait briefly for socket to appear, then return
                self._wait_for_socket(timeout=10)
                return pid
            # Child: become session leader, run worker
            os.setsid()
            self._write_pid(os.getpid())
            run_worker(self.config)  # blocks until shutdown
            self._cleanup_stale_files()
            os._exit(0)
        else:
            # Foreground mode (for development/testing)
            self._write_pid(os.getpid())
            run_worker(self.config)
            self._cleanup_stale_files()
            return os.getpid()

    def stop(self) -> bool:
        """Stop worker via SIGTERM. Returns True if stopped."""
        pid = self.get_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait for process to exit (max 5s)
            for _ in range(50):
                try:
                    os.kill(pid, 0)  # check if alive
                    time.sleep(0.1)
                except OSError:
                    break
            self._cleanup_stale_files()
            return True
        except OSError:
            self._cleanup_stale_files()
            return False

    def is_running(self) -> bool:
        """Check if worker process is alive."""
        pid = self.get_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            self._cleanup_stale_files()
            return False

    def get_pid(self) -> int | None:
        """Read PID from file. Returns None if no PID file or invalid."""
        try:
            return int(self.pid_path.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None

    def _write_pid(self, pid: int):
        self.pid_path.write_text(str(pid))

    def _cleanup_stale_files(self):
        """Remove PID file and socket file if process is dead."""
        for path in (self.pid_path, self.socket_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _wait_for_socket(self, timeout: int = 10):
        """Wait for socket file to appear (worker is ready)."""
        for _ in range(timeout * 10):
            if self.socket_path.exists():
                return
            time.sleep(0.1)
        raise TimeoutError(f"Worker did not create socket within {timeout}s")
```

**Shell alias** (from architecture spec):
```bash
alias cc='claude-mem-worker start --daemon && claude'
```

**CLI entry point**:
```python
# In pyproject.toml:
[project.scripts]
claude-mem-worker = "claude_mem_lite.worker.lifecycle:cli_main"
```

```python
def cli_main():
    import sys
    config = Config()
    lifecycle = WorkerLifecycle(config)

    if len(sys.argv) < 2:
        print("Usage: claude-mem-worker {start|stop|status}")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "start":
        daemon = "--foreground" not in sys.argv
        pid = lifecycle.start(daemon=daemon)
        print(f"Worker started (PID: {pid})")
    elif cmd == "stop":
        if lifecycle.stop():
            print("Worker stopped")
        else:
            print("Worker not running")
    elif cmd == "status":
        if lifecycle.is_running():
            print(f"Worker running (PID: {lifecycle.get_pid()})")
        else:
            print("Worker not running")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
```

### 2.8 Session Summarizer (`summarizer.py`)

```python
class Summarizer:
    """Generate session summaries from accumulated observations."""

    SUMMARY_PROMPT = """\
Summarize this development session from its observations.

Session observations (chronological):
{observations}

Return ONLY a JSON object:
{{
  "summary": "2-4 sentence summary of what was accomplished. Focus on outcomes, not process.",
  "key_files": ["list of most important files changed"],
  "key_decisions": ["any significant technical decisions made"]
}}
"""

    async def summarize_session(self, session_id: str) -> SessionSummary:
        """Generate summary from all observations in a session."""
        observations = await self._get_session_observations(session_id)
        if not observations:
            return SessionSummary(summary="No observations captured.", key_files=[], key_decisions=[])

        # Build observation digest â€” titles + summaries, ~50 tokens each
        obs_text = "\n".join(
            f"- [{obs.tool_name}] {obs.title}: {obs.summary}"
            for obs in observations
        )

        response = await self.compressor.client.messages.create(
            model=self.compressor.model,
            max_tokens=512,
            messages=[{"role": "user", "content": self.SUMMARY_PROMPT.format(observations=obs_text)}],
        )

        # Parse, validate, store
        data = json.loads(response.content[0].text)
        summary = SessionSummary(**data)

        await self.db.execute(
            "UPDATE sessions SET summary = ?, status = 'closed' WHERE id = ?",
            (summary.summary, session_id),
        )
        await self.db.commit()
        return summary
```

### 2.9 Graceful Degradation

| Failure mode | Behavior | Recovery |
|-------------|----------|---------|
| **Claude API down** | Queue items stay `status='raw'`, processor logs warning, continues polling | Items auto-processed when API returns |
| **Claude API rate limited** | Exponential backoff (5s, 10s, 20s), item retried up to 3 times | Backoff clears naturally |
| **Worker not running** | Hooks write to SQLite directly (always works). No context injection, no summarization. | Next `claude-mem-worker start` processes backlog |
| **Worker crashes mid-processing** | Items in `status='processing'` are orphaned | Startup recovery: reset `processing` â†’ `raw` for items older than 5 minutes |
| **Invalid API response** | Item marked `status='error'`, raw output preserved | Manual inspection via `claude-mem queue --errors` |
| **SQLite locked** | aiosqlite retries via `busy_timeout=3000` | Automatic â€” WAL mode handles concurrent access |
| **Disk full** | JSONL log write fails â†’ exception propagates, item stays in queue | Manual cleanup needed |

**Startup recovery** (in processor init):
```python
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
```

---

## 3. Data Types

### 3.1 New Pydantic models (additions to `storage/models.py`)

```python
class CompressedObservation(BaseModel):
    """Result of AI compression â€” becomes an Observation row."""
    title: str
    summary: str
    detail: str | None = None
    files_touched: list[str] = []
    functions_changed: list[FunctionChange] = []
    tokens_in: int = 0
    tokens_out: int = 0

class FunctionChange(BaseModel):
    file: str
    name: str
    action: str  # "new" | "modified" | "deleted"

class SessionSummary(BaseModel):
    summary: str
    key_files: list[str] = []
    key_decisions: list[str] = []

class QueueStats(BaseModel):
    raw: int = 0
    processing: int = 0
    done: int = 0
    error: int = 0

class HealthResponse(BaseModel):
    status: str = "ok"
    uptime_s: int = 0
    queue_depth: int = 0
    observations_today: int = 0
```

### 3.2 Custom exceptions

```python
class RetryableError(Exception):
    """Error that should trigger retry with backoff."""
    pass

class NonRetryableError(Exception):
    """Error that should mark item as permanently failed."""
    pass
```

---

## 4. Compression Quality Criteria

### 4.1 What "good compression" means

A compressed observation is good if a future Claude session can:
1. **Understand what happened** from the title alone (~10 tokens)
2. **Decide relevance** from the summary (~50 tokens)
3. **Recover key details** from the detail field (~200 tokens) when needed

### 4.2 Quality dimensions

| Dimension | Description | How to measure |
|-----------|-------------|----------------|
| **Information preservation** | Key facts from raw output survive compression | Generate 5 questions from raw, answer from compressed, score % correct |
| **Decision rationale** | WHY something was done, not just WHAT | Manual review: does summary explain intent? |
| **Actionability** | Could a new session continue the work from just the observation? | Manual review: score 1-5 |
| **Compression ratio** | Tokens saved vs. information lost | `raw_size_bytes / compressed_tokens` â€” target 10:1 to 100:1 |
| **Structural validity** | Valid JSON with all required fields | Automated: JSON parse + field presence check |
| **Latency** | Time from API call to parsed result | `compress.done` event duration_ms â€” target <2000ms |

### 4.3 Prompt iteration strategy (during implementation)

1. Collect 20-30 real tool outputs from actual Claude Code sessions
2. Run compression with v1 prompt
3. Score each on information preservation (automated) and actionability (manual)
4. Identify failure patterns (e.g., "Read tool outputs always get empty detail")
5. Adjust prompt, re-run, compare
6. Lock prompt version when >80% of observations score â‰¥4/5 on actionability

This iteration happens during Phase 3 implementation, not as a separate phase. The Phase 7 A/B framework will enable systematic comparison later.

---

## 5. aiosqlite Usage Notes

### 5.1 Breaking change in v0.22.0

aiosqlite v0.22.0 (Dec 2025) changed `Connection` to no longer inherit from `threading.Thread`. This means:

- **Must use as context manager** or explicitly call `await db.close()`
- **`ResourceWarning` emitted** if connection is garbage collected without closing
- **`stop()` method added** for synchronous cleanup (useful in signal handlers)

Our usage in `lifespan()` handles this correctly â€” the `yield` pattern ensures cleanup on shutdown.

### 5.2 Connection pattern

```python
# Worker uses a single long-lived connection (lifespan-managed)
db = await aiosqlite.connect(config.db_path)
db.row_factory = aiosqlite.Row
await db.execute("PRAGMA journal_mode=WAL")
await db.execute("PRAGMA foreign_keys=ON")
await db.execute("PRAGMA busy_timeout=3000")
```

**Why single connection**: aiosqlite uses one background thread per connection. For our single-user worker with sequential processing, one connection is sufficient. Multiple connections would add thread overhead with no concurrency benefit (SQLite is single-writer anyway).

### 5.3 Transaction handling

aiosqlite proxies `sqlite3` but requires `await` for all operations. Transactions:

```python
# Explicit transaction (for atomic operations like dequeue_batch)
await db.execute("BEGIN IMMEDIATE")
try:
    cursor = await db.execute("UPDATE ... RETURNING *")
    rows = await cursor.fetchall()
    await db.execute("COMMIT")
except Exception:
    await db.execute("ROLLBACK")
    raise

# Simple operations (autocommit)
await db.execute("INSERT INTO observations ...")
await db.commit()
```

---

## 6. Test Plan

### 6.1 Test categories

| Category | Tests | What it validates |
|----------|-------|-------------------|
| **Compressor** | 6 | Prompt builds correctly, JSON parsing with/without fences, truncation at 32K, error classification (retryable vs non-retryable), token counting from usage |
| **Processor** | 7 | Dequeue batch atomic, process item success flow, retry on retryable error, mark error after max attempts, backoff timing, recovery of orphaned items, idle tracker touch on processing |
| **Server endpoints** | 4 | `/api/health` returns stats, `/api/context` returns placeholder, `/api/summarize` generates summary, `/api/queue/stats` returns counts |
| **Summarizer** | 3 | Summary from observations, empty session handling, summary stored in sessions table |
| **Lifecycle** | 5 | Start creates PID + socket, stop sends SIGTERM + cleans up, is_running detects dead process, stale file cleanup, foreground mode |
| **Integration** | 3 | End-to-end: enqueue â†’ process â†’ observation stored, concurrent hooks writing while processor reads, graceful shutdown mid-processing |
| **Total** | **28** | |

### 6.2 Test infrastructure

```python
# conftest.py additions for Phase 3

@pytest.fixture
async def async_db(tmp_config):
    """Async SQLite connection for worker tests."""
    import aiosqlite
    from claude_mem_lite.storage.migrations import migrate
    import sqlite3

    # Run sync migrations first
    sync_conn = sqlite3.connect(tmp_config.db_path)
    migrate(sync_conn)
    sync_conn.close()

    db = await aiosqlite.connect(tmp_config.db_path)
    db.row_factory = aiosqlite.Row
    yield db
    await db.close()

@pytest.fixture
def mock_anthropic():
    """Mock AsyncAnthropic client for compression tests."""
    with unittest.mock.patch("claude_mem_lite.worker.compressor.AsyncAnthropic") as mock:
        client = mock.return_value
        # Default: return valid compressed JSON
        client.messages.create = AsyncMock(return_value=MockResponse(
            content=[MockContent(text='{"title":"Test","summary":"Test summary","detail":null,"files_touched":[],"functions_changed":[]}')],
            usage=MockUsage(input_tokens=100, output_tokens=50),
        ))
        yield client

@pytest.fixture
async def processor(async_db, mock_anthropic, tmp_config):
    """Processor with mocked dependencies."""
    compressor = Compressor(tmp_config)
    compressor.client = mock_anthropic
    logger = MemLogger(tmp_config.log_dir)
    idle = IdleTracker(timeout_minutes=999)
    return Processor(async_db, compressor, logger, idle)
```

### 6.3 Key test scenarios

**Compressor tests** (mock Anthropic API):
- Valid JSON response â†’ `CompressedObservation` with correct fields
- JSON wrapped in ` ```json ` fences â†’ correctly stripped and parsed
- Raw output >32K chars â†’ truncated with head+tail strategy
- `APIConnectionError` â†’ raises `RetryableError`
- `APIStatusError(400)` â†’ raises `NonRetryableError`
- Response missing `title` field â†’ raises `NonRetryableError`

**Processor tests** (mock compressor):
- 3 items in queue â†’ all processed, status='done', observations created
- Retryable error on first attempt â†’ item back to 'raw' with attempts=1
- Item at max_attempts with retryable error â†’ status='error'
- Orphaned 'processing' items from >5min ago â†’ reset to 'raw' on startup
- Empty queue â†’ processor sleeps POLL_INTERVAL, doesn't error

**Lifecycle tests** (real processes):
- Start creates PID file and socket file
- Stop sends SIGTERM, removes PID and socket
- is_running returns False for dead PID, cleans up stale files
- Double-start returns existing PID (idempotent)

### 6.4 Performance targets

| Operation | Target | Rationale |
|-----------|--------|-----------|
| Compression API call | <2000ms | Haiku 4.5 typical latency |
| JSON parse + validate | <1ms | Tiny payloads |
| Dequeue batch (5 items) | <5ms | aiosqlite overhead ~2-3ms |
| Queue poll cycle (empty) | ~2s | POLL_INTERVAL sleep |
| Worker startup | <2s | No model loading in Phase 3 |
| Health check endpoint | <5ms | Simple DB query |

---

## 7. Acceptance Criteria

Phase 3 is complete when:

- [ ] Worker starts as daemon, creates PID file at `~/.claude-mem/worker.pid` and socket at `~/.claude-mem/worker.sock`
- [ ] Worker stops cleanly via `claude-mem-worker stop` (SIGTERM, file cleanup)
- [ ] Worker auto-shuts down after 30 minutes of inactivity
- [ ] Queue processor polls `pending_queue`, claims items atomically, processes sequentially
- [ ] Compressor calls Claude Haiku 4.5 via `AsyncAnthropic().messages.create()` with structured prompt
- [ ] Compressed observations stored in `observations` table with title, summary, detail, files_touched, functions_changed
- [ ] Retry logic: retryable errors â†’ exponential backoff (5s, 10s, 20s), max 3 attempts
- [ ] Non-retryable errors â†’ item marked `status='error'` with error message preserved
- [ ] Orphaned `processing` items recovered on startup (reset to `raw` if >5min old)
- [ ] `/api/health` endpoint returns status, uptime, queue depth
- [ ] `/api/summarize` generates session summary from observations
- [ ] Graceful degradation: API down â†’ items stay in queue; worker down â†’ hooks still write to SQLite
- [ ] All 28 tests pass (pytest + pytest-asyncio, <15s total with mocked API)
- [ ] `ruff check` and `ruff format --check` pass with zero warnings
- [ ] `pip install -e ".[dev]"` installs all Phase 0-3 dependencies cleanly

---

## 8. Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Compression prompt quality** | High | High | Iterate during implementation with real tool outputs. Lock prompt when >80% score â‰¥4/5. Phase 7 adds systematic A/B testing. |
| **Haiku 4.5 returns malformed JSON** | Medium | Low | Parser handles fences, validates fields. Non-retryable error preserves raw output for inspection. |
| **aiosqlite deadlock with sync hooks** | Low | High | Worker uses separate connection from hooks. WAL mode + busy_timeout=3s. Different processes, different connections. |
| **os.fork() issues on some systems** | Low | Medium | Foreground mode as fallback. Could replace with `subprocess.Popen()` if fork causes issues. |
| **Stale socket file prevents restart** | Low | Low | `_cleanup_stale_files()` on startup. Socket removed on clean shutdown by uvicorn. |
| **API cost unexpectedly high** | Low | Medium | Truncate input at 32K chars. Monitor via event_log token counts. Phase 7 adds cost tracking dashboard. |

---

## 9. Open Questions

| Question | Current assumption | When to resolve |
|----------|-------------------|-----------------|
| **Should summarizer use Haiku or Sonnet?** | Haiku for everything initially. Summaries are aggregations of already-compressed observations â€” Haiku should be sufficient. | Phase 7 A/B testing will compare. |
| **Should we batch multiple compressions in a single API call?** | No â€” one tool output per API call. Batching would require combining unrelated outputs into one prompt, risking cross-contamination. | Revisit if API latency becomes a bottleneck. |
| **Should compression be streaming?** | No â€” outputs are <1024 tokens, streaming adds complexity without benefit. | Revisit only if we increase max_tokens significantly. |
| **Should the worker pre-create the aiosqlite connection pool?** | No â€” single connection is sufficient for single-user sequential processing. | Revisit if we add concurrent processing paths. |
| **`os.fork()` vs `subprocess.Popen()` for daemon?** | `os.fork()` â€” simpler, standard Unix daemon pattern. | Switch to subprocess if fork causes issues with asyncio event loop or signal handling. |
