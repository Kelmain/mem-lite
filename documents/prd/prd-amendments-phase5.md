# PRD Amendments: Phase 5 (Post-Review)

**Date**: 2026-02-08
**Triggered by**: External review of Phase 5 PRD (4 comments, all actionable)
**Affects**: Phase 5 v1 (Context Injection)

---

## Amendment 1: Remove Semantic Search from Layer 4 at SessionStart

**Severity**: High
**Affects**: Phase 5, Section 2.2 (`_build_observations`)

### Problem

The `_build_observations` method (lines 326â€“351) uses the **previous session's summary** as the semantic search query when `lance_store` is available. This introduces context pollution when consecutive sessions work on unrelated topics.

**Concrete failure scenario:**

1. Session N: "Refactor database schema for users table" â†’ Summary: "DB schema refactor, Alembic migration, added indexes"
2. Session N+1: "Update frontend CSS for the login page"
3. At SessionStart for N+1, the builder takes "DB schema refactorâ€¦" as the search query
4. LanceDB returns observations about SQL, Alembic, schema design
5. These observations are injected into a CSS/frontend session

**Result**: The context actively misleads Claude by injecting topically irrelevant observations. This is worse than injecting no observations at all â€” it biases Claude toward the *previous* session's domain, not the *current* session's intent.

The core issue: `SessionStart` fires **before any user input exists**. There is no signal about the new session's intent. Using the old session's summary as a proxy assumes topic continuity across sessions, which is often false.

The PRD's own Open Questions table (Section 10, row 3) flags this uncertainty: "Should Layer 4 use the current session's files as search query? Currently uses last session's summary." This confirms the design was not high-confidence.

### Specification Change

Remove the semantic search branch from `_build_observations`. Use recency-based retrieval exclusively.

```python
# BEFORE (semantic search using stale query â€” removed)
async def _build_observations(self, project_path: str) -> Optional[ContextLayer]:
    """Relevant recent observations via hybrid search.

    Strategy:
    - If lance_store available: semantic search using recent session summary as query
    - Fallback: most recent observations from SQLite
    """
    if self.lance_store:
        recent = await self.db.execute_fetchone(
            """
            SELECT summary_text FROM sessions
            WHERE project_path = ? AND summary_text IS NOT NULL
            ORDER BY started_at DESC LIMIT 1
            """,
            (project_path,),
        )
        if recent and recent["summary_text"]:
            query = recent["summary_text"][:200]
            results = await asyncio.to_thread(
                self.lance_store.search_observations,
                query=query,
                limit=10,
            )
            if results:
                return self._format_observations(results)

    # Fallback: most recent observations from SQLite
    rows = await self.db.execute_fetchall(...)
    ...
```

```python
# AFTER (recency only â€” no intent guessing)
async def _build_observations(self, project_path: str) -> Optional[ContextLayer]:
    """Recent high-value observations by recency.

    At SessionStart, there is no user input to infer intent.
    Semantic search is deferred to UserPromptSubmit (future phase)
    where actual user intent is available as a search query.
    """
    rows = await self.db.execute_fetchall(
        """
        SELECT title, summary, created_at
        FROM observations
        WHERE session_id IN (
            SELECT id FROM sessions
            WHERE project_path = ?
            ORDER BY started_at DESC LIMIT 5
        )
        ORDER BY created_at DESC
        LIMIT 10
        """,
        (project_path,),
    )
    if not rows:
        return None

    return self._format_observations_from_rows(rows)
```

### Consequences

- `lance_store` parameter in `ContextBuilder.__init__` is no longer used by `_build_observations`. It stays in the constructor signature for forward compatibility (Phase 9 `UserPromptSubmit` will use it), but is not called.
- The `_format_observations` method (for LanceDB results) can be removed or kept as dead code for Phase 9. Recommendation: keep it, add a `# Used by UserPromptSubmit (Phase 9)` comment.
- Layer 4 section header changes from `## Relevant Past Work` to `## Recent Observations` to reflect recency-based, not relevance-based, retrieval.

### Open Questions Update

Section 10, row 3 is resolved:

| Question | ~~Current Assumption~~ Resolution |
|---|---|
| Should Layer 4 use the current session's files as search query? | **Resolved**: No search at SessionStart. Recency only. Semantic search deferred to `UserPromptSubmit` (Phase 9) where user intent is available. |

### Test Changes

Update `test_observations_with_lance` to verify that `lance_store.search_observations` is **not** called during `build()`. Replace with a test that verifies recency-ordered SQLite retrieval.

```python
@pytest.mark.asyncio
async def test_observations_recency_not_semantic(self, populated_db):
    """Layer 4 uses recency, not semantic search, at SessionStart."""
    mock_lance = MagicMock()
    builder = ContextBuilder(populated_db, lance_store=mock_lance)
    result = await builder.build("/project", "s")

    # lance_store should NOT be called
    mock_lance.search_observations.assert_not_called()
    # observations should still appear (from SQLite recency path)
    if "observations" in result.layers_included:
        assert "## Recent Observations" in result.text
```

---

## Amendment 2: Guard `AF_UNIX` Availability in `_UDSHTTPConnection`

**Severity**: Low
**Affects**: Phase 5, Section 2.4 (`_UDSHTTPConnection`, `_get_worker_context`)

### Problem

`_UDSHTTPConnection.connect()` (line 606) calls `socket.socket(socket.AF_UNIX, ...)` unconditionally. `socket.AF_UNIX` is not available on all platforms:

- **Linux/macOS**: Always available â€” no issue
- **Windows (WSL)**: Available â€” WSL runs a real Linux kernel
- **Windows (native + Git Bash)**: `AF_UNIX` added in Python 3.10 but requires Windows 10 build 17063+. Availability is inconsistent â€” `hasattr(socket, 'AF_UNIX')` can return `False` depending on the Python build and OS version.

### Risk Assessment

**Low severity for this project.** The target OS is Ubuntu (Phase 0, Section 9). Claude Code on Windows runs primarily via WSL where `AF_UNIX` is native. In the native Windows + Git Bash path, Claude Code's own hook transport uses Unix domain sockets â€” if `AF_UNIX` didn't work, the hook would never be invoked in the first place.

However, the fix is a single guard clause that costs nothing and prevents a hard crash in edge cases.

### Specification Change

Add an `AF_UNIX` availability check at the top of `_get_worker_context`:

```python
# BEFORE
def _get_worker_context(project_path: str, session_id: str) -> str:
    import http.client
    try:
        conn = _UDSHTTPConnection(str(WORKER_SOCK))
        ...

# AFTER
def _get_worker_context(project_path: str, session_id: str) -> str:
    if not hasattr(socket, 'AF_UNIX'):
        return ""  # Platform doesn't support UDS â€” fall through to SQLite fallback

    import http.client
    try:
        conn = _UDSHTTPConnection(str(WORKER_SOCK))
        ...
```

No test changes required â€” existing fallback tests cover the empty-string return path.

---

## Amendment 3: Increase SQLite Fallback Timeout from 1s to 3s

**Severity**: Low-Medium
**Affects**: Phase 5, Section 2.4 (`_get_basic_context`, line 619)

### Problem

The SQLite fallback path uses `sqlite3.connect(str(DB_PATH), timeout=1)`. This timeout determines how long the connection waits to acquire a lock before raising `sqlite3.OperationalError`.

**Context**: The fallback fires when the worker is unavailable. It executes a single read-only `SELECT` query. In WAL mode, readers are almost never blocked â€” the SQLite WAL guarantee is "readers do not block writers and a writer does not block readers."

**Edge cases where a read can block:**

1. **WAL checkpoint** â€” `wal_checkpoint(TRUNCATE)` briefly requires an exclusive lock that blocks all access (readers included). Phase 3's worker uses `PASSIVE` checkpoints (non-blocking), but a manual `TRUNCATE` checkpoint or migration could hold the lock.
2. **Schema migration** â€” `ALTER TABLE` or `CREATE TABLE` during migration holds a write lock. If the worker is restarting and running migrations while the hook fires, the read could block.
3. **Heavy WAL recovery** â€” After an unclean shutdown, SQLite may need to replay the WAL on the next connection open, which holds a lock.

At 1 second, transient lock contention from checkpoints or migrations causes a false negative (empty context returned). Python's default `sqlite3.connect` timeout is 5 seconds.

### Specification Change

```python
# BEFORE
conn = sqlite3.connect(str(DB_PATH), timeout=1)

# AFTER
conn = sqlite3.connect(str(DB_PATH), timeout=3)
```

**Why 3 seconds, not 5:**

The hook has a 10-second Claude Code timeout (Section 6.1) and a 5-second internal `CONTEXT_TIMEOUT_S` for the worker path. If the worker path fails (5s) and the SQLite fallback then waits 5s, total elapsed is 10s â€” right at the Claude Code hook timeout. At 3 seconds, worst-case elapsed is 8s with comfortable margin.

The 3-second timeout costs nothing when the DB is free (lock acquired in microseconds). It only matters when contention exists, where it prevents false negatives.

### Config Update

Add the fallback timeout to `ContextConfig` (Section 2.5) for consistency:

```python
@dataclass
class ContextConfig:
    ...
    worker_timeout_s: float = 5.0          # Max wait for worker response
    fallback_db_timeout_s: float = 3.0     # Max wait for SQLite lock in fallback path
    ...
```

Update `_get_basic_context` to use `config.fallback_db_timeout_s` instead of a hardcoded value.

---

## Amendment 4: Enforce UTC in `_relative_time` Helper

**Severity**: Low
**Affects**: Phase 5, Section 2.2 (`_build_session_index`, `_relative_time` helper)

### Problem

The `_relative_time` helper calculates relative timestamps ("2h ago", "yesterday") by comparing a database timestamp string against `datetime.now()`. Two timezone issues:

1. **SQLite stores timestamps without timezone info.** Phase 0 uses `datetime('now')` as defaults, which returns UTC â€” but the stored string has no `+00:00` suffix. `datetime.fromisoformat()` parses it as a naive (timezone-unaware) datetime.
2. **`datetime.now()` returns local time.** On a machine with UTC+2 timezone, `datetime.now()` is 2 hours ahead of the database's UTC timestamps.

**Result**: Relative time calculations are offset by the machine's UTC offset. A session started 2 hours ago could display as "4h ago" (UTC+2) or "in the future" (UTC-negative).

**Practical impact**: Low â€” the output is cosmetic (context display for Claude). Claude's behavior doesn't meaningfully change whether it sees "2h ago" vs "4h ago". However, "in 3 hours" for a past session is visibly wrong and undermines trust in the tool.

### Specification Change

```python
# BEFORE (timezone-naive â€” broken on non-UTC machines)
def _relative_time(db_timestamp: str) -> str:
    now = datetime.now()
    dt = datetime.fromisoformat(db_timestamp)
    delta = now - dt
    ...

# AFTER (explicit UTC â€” correct everywhere)
def _relative_time(db_timestamp: str) -> str:
    now = datetime.now(timezone.utc)
    dt = datetime.fromisoformat(db_timestamp)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # DB stores UTC without suffix
    delta = now - dt
    ...
```

Requires adding `timezone` to the import: `from datetime import datetime, timezone`.

### Test Addition

```python
def test_relative_time_utc_handling(self):
    """_relative_time handles naive UTC timestamps from SQLite."""
    from claude_mem_lite.context.builder import _relative_time

    # Simulate a DB timestamp (naive, UTC) 2 hours ago
    two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2))
    db_str = two_hours_ago.strftime("%Y-%m-%d %H:%M:%S")  # No TZ suffix

    result = _relative_time(db_str)
    assert "ago" in result  # Must not be "in the future"
    assert "2h" in result or "1h" in result  # Approximate
```

---

## Risk Table Update

Section 9 of the Phase 5 PRD should be updated:

| Risk | ~~Original~~ | Updated |
|---|---|---|
| Stale context from old sessions | Medium likelihood, Low impact | **Resolved** by Amendment 1 â€” Layer 4 no longer injects stale semantic search results |

Add new row:

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Timezone mismatch in relative timestamps | Medium (any non-UTC machine) | Low (cosmetic) | Amendment 4 â€” explicit UTC handling in `_relative_time` |

---

## Summary

| # | Amendment | Severity | Change Type |
|---|-----------|----------|-------------|
| 1 | Remove semantic search from Layer 4 at SessionStart | High | Logic change â€” recency replaces semantic search |
| 2 | Guard `AF_UNIX` availability | Low | One-liner guard clause |
| 3 | Increase SQLite fallback timeout to 3s | Low-Medium | Config value change (1s â†’ 3s) |
| 4 | Enforce UTC in `_relative_time` | Low | 3-line defensive fix |
