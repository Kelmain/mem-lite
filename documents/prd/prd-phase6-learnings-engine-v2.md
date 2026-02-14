# PRD: Phase 6 Ã¢â‚¬â€ Learnings Engine + Call Graph Self-Healing (v1)

**Project**: claude-mem-lite (fork of claude-mem, Python rewrite)
**Phase**: 6 of 9
**Status**: Draft Ã¢â‚¬â€ post-review v2
**Dependencies**: Phase 3 (Worker + Compression Ã¢â‚¬â€ summarizer, compressor, processor), Phase 4 (Embeddings + Search Ã¢â‚¬â€ LanceDB, Qwen3 embeddings, hybrid search)
**Estimated effort**: 2 sessions (~10Ã¢â‚¬â€œ14 hours)
**Python**: 3.14

---

## Changelog from Architecture Spec & Implementation Plan

| Item | Architecture / Implementation Plan | This PRD (verified Feb 2026) | Rationale |
|------|-------------------------------------|------------------------------|-----------|
| Learnings table schema | Architecture: `times_seen`, `source_sessions TEXT`, `is_manual BOOLEAN`. Phase 0 PRD: `source_session_id TEXT NOT NULL` (singular), no `times_seen`, no `is_manual` | **Schema migration required** Ã¢â‚¬â€ add `times_seen`, `source_sessions`, `is_manual`. Drop unused `source_session_id` column (SQLite doesn't support DROP COLUMN before 3.35, so recreate table) | Dedup and confidence evolution require tracking multiple source sessions and manual overrides. Phase 0 schema was incomplete for Phase 6's needs. |
| Wiring point | Impl plan: "Wire into Stop hook: after session summary, extract learnings" | **Wire into worker processor** Ã¢â‚¬â€ after session summarization, not the Stop hook | Phase 1 PRD Ã‚Â§5.3 clarified: Stop hook fires on *every Claude response*, not just task completion. Session summarization happens in the worker (Phase 3 `summarizer.py`). Learnings extraction runs after summarization completes, in the same worker pipeline. |
| Call graph columns | Impl plan: "Enhanced call_graph table with confidence, times_confirmed, source columns" | **No migration needed** Ã¢â‚¬â€ columns already exist in Phase 0 schema | Phase 0 PRD already defined `confidence REAL`, `times_confirmed INTEGER`, `source TEXT` on `call_graph`. Phase 6 just *uses* them. |
| Dedup threshold | Architecture: "cosine similarity > 0.85" | **Configurable, default 0.90** | Research shows 0.85 is aggressive for short text with 256-dim Matryoshka embeddings. NVIDIA NeMo uses 0.9Ã¢â‚¬â€œ0.99. SemHash defaults to 0.90. Short text pairs tend to have inflated cosine scores Ã¢â‚¬â€ a higher threshold reduces false positive merges. Configurable via `Config.learning_dedup_threshold`. |
| Confidence formula | Architecture: approximate values (0.5Ã¢â€ â€™0.7Ã¢â€ â€™0.85Ã¢â€ â€™0.95) but no formula | **Concrete formula**: `min(0.95, current + 0.2 Ãƒâ€” (1 - current))` | Diminishing returns: each confirmation adds 20% of remaining headroom. Converges to 0.95 (only manual confirmation reaches 1.0). Produces: 0.5Ã¢â€ â€™0.6Ã¢â€ â€™0.68Ã¢â€ â€™0.744Ã¢â€ â€™0.80Ã¢â€ â€™... Matches the architecture's trajectory without hardcoded lookup tables. |
| Contradiction detection | Architecture: "flag for review" | **LLM-assisted with same API call** Ã¢â‚¬â€ contradiction check embedded in extraction prompt | Separate contradiction detection API calls are wasteful. Instead, the extraction prompt includes existing learnings for the project and asks the model to flag conflicts inline. One API call serves both extraction and conflict detection. |
| Haiku 4.5 pricing | Phase 3 confirmed $1/$5 per MTok | **Confirmed still $1/$5 per MTok** (Feb 2026) | No change from Phase 3. |
| Model string | Not specified for Phase 6 | **`claude-haiku-4-5-20251001`** (same pinned snapshot as Phase 3) | Consistency with Phase 3 compressor. |

---

## Review Amendments (v1 â†’ v2)

| # | Issue | Severity | Fix applied |
|---|-------|----------|-------------|
| 1 | `DROP COLUMN` fallback only mentioned version constraint, not index/FK/trigger restrictions | Low | Updated fallback (Â§2.3) to enumerate all failure modes. Added try/except pattern for implementation. |
| 2 | Ghost learning re-insertion: low-confidence learnings invisible to extraction prompt but findable by `_find_duplicate`, causing zombie resurrection loop | Medium | Added `_get_active_learnings` implementation (Â§2.4) returning ALL active learnings regardless of confidence. Extraction prompt (Â§2.5) marks low-confidence learnings with prefix. Added `learning.revived` log event on merge path. Updated cost estimation for 30 learnings in prompt (~900 tokens). |
| 3 | Contradiction logic penalized old learning but never inserted the new one â€” system forgot the topic entirely | High | Contradiction path in `_process_candidate` (Â§2.4) now: (1) penalizes old, (2) inserts new as fresh entry, (3) embeds in LanceDB. Log event includes both old and new learning IDs. Updated acceptance criteria and test scenarios. |



## 1. Purpose & Context

### 1.1 What this phase delivers

The system that makes claude-mem-lite *smarter over time*. After enough sessions, Claude starts getting injected with project-specific knowledge it extracted from past work Ã¢â‚¬â€ architecture decisions, coding conventions, known gotchas, dependency notes, recurring patterns.

Specifically:

- **`learnings/engine.py`** Ã¢â‚¬â€ Extract learnings from session summaries, deduplicate against existing learnings via semantic similarity, evolve confidence scores
- **`learnings/prompts.py`** Ã¢â‚¬â€ Extraction and conflict-detection prompt
- **`learnings/healer.py`** Ã¢â‚¬â€ Call graph self-healing: confirm or discover edges from compressed observations
- **Schema migration** Ã¢â‚¬â€ Add missing columns to `learnings` table (`times_seen`, `source_sessions`, `is_manual`)
- **Worker integration** Ã¢â‚¬â€ Learnings extraction runs automatically after session summarization
- **LanceDB learning index** Ã¢â‚¬â€ Embed learnings for dedup search (wired into Phase 4's `lance_store`)
- **CLI command** Ã¢â‚¬â€ `claude-mem learnings` to list, add, edit, remove learnings manually

### 1.2 What this phase does NOT deliver

- **Learnings UI** Ã¢â‚¬â€ No web interface. CLI + context injection (Phase 5 already picks up learnings from the `learnings` table).
- **Cross-project learnings** Ã¢â‚¬â€ Each project has its own learning set. Global patterns (Phase 9 consideration).
- **Learnings from user prompts** Ã¢â‚¬â€ We only extract from session summaries + observations, not from what the user typed.
- **Call graph major restructuring** Ã¢â‚¬â€ Self-healing adjusts confidence on existing edges and discovers new ones from observations. It does not re-run AST parsing (that's Phase 2 at capture time).
- **Automated stale learning pruning** Ã¢â‚¬â€ We decay confidence but don't auto-delete. Manual review via CLI.

### 1.3 Why this matters

Without learnings, context injection (Phase 5) only has session summaries, function maps, and recent observations. These are *what happened* Ã¢â‚¬â€ but not *what we learned from it*. Learnings distill patterns:

- "Auth uses JWT with HttpOnly cookies, refresh flow in `auth/refresh.py`" Ã¢â€ â€™ Saves Claude from re-discovering this every session
- "The ORM silently swallows connection errors Ã¢â‚¬â€ always wrap `db.execute()` in try/except" Ã¢â€ â€™ Prevents repeated bugs
- "`pytest fixtures` use `tmp_path`, never manual `/tmp` paths" Ã¢â€ â€™ Convention enforcement

The call graph self-healing is less critical but valuable: it improves accuracy of the function map context (Layer 2) over time, particularly for unresolved edges that Phase 2's static analysis couldn't resolve.

### 1.4 Cost estimation

Learnings extraction adds one Haiku 4.5 API call per session close:

| Input | Tokens | Cost per session |
|-------|--------|------------------|
| System prompt + extraction instructions | ~400 | |
| Session summary (~200 tokens) | ~200 | |
| Recent observations (~5 Ãƒâ€” 100 tokens) | ~500 | |
| Existing learnings for context (~30 Ã— 30 tokens, includes low-confidence) | ~900 | |
| **Total input** | **~2,000** | **$0.002** |
| Output (JSON array, ~5 learnings Ãƒâ€” 50 tokens) | ~300 | **$0.0015** |
| **Total per session** | | **~$0.0035** |

At 10 sessions/day, that's ~$0.03/day or ~$0.90/month. Negligible.

---

## 2. Technical Specification

### 2.1 Module Structure

```
src/claude_mem_lite/learnings/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ __init__.py
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ engine.py          # LearningsEngine Ã¢â‚¬â€ extraction, dedup, confidence evolution
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ prompts.py         # Extraction prompt template
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ healer.py          # CallGraphHealer Ã¢â‚¬â€ edge confirmation from observations
```

### 2.2 Dependencies (Phase 6 additions)

**None.** Phase 6 uses only dependencies already installed in prior phases:

- `anthropic` (Phase 3) Ã¢â‚¬â€ Haiku API calls for extraction
- `aiosqlite` (Phase 3) Ã¢â‚¬â€ async DB access in worker
- `sentence-transformers` (Phase 4) Ã¢â‚¬â€ embedding for dedup
- `lancedb` (Phase 4) Ã¢â‚¬â€ learning index

No new `pyproject.toml` entries.

### 2.3 Schema Migration

Phase 0's `learnings` table is missing columns needed for Phase 6. This requires migration v2 (or whatever the next version is after prior phases).

**Problem**: The Phase 0 schema has `source_session_id TEXT NOT NULL` (singular). Phase 6 needs `source_sessions TEXT` (JSON array). SQLite 3.35+ supports `ALTER TABLE ... DROP COLUMN`, and Python 3.14 ships SQLite 3.47+, so we can use it.

```sql
-- Migration: Phase 6 learnings table update
ALTER TABLE learnings ADD COLUMN times_seen INTEGER NOT NULL DEFAULT 1;
ALTER TABLE learnings ADD COLUMN source_sessions TEXT NOT NULL DEFAULT '[]';
ALTER TABLE learnings ADD COLUMN is_manual INTEGER NOT NULL DEFAULT 0;

-- Migrate existing data: copy source_session_id into source_sessions array
UPDATE learnings
SET source_sessions = json_array(source_session_id)
WHERE source_session_id IS NOT NULL AND source_sessions = '[]';

-- Drop the old singular column (SQLite 3.35+, Python 3.14 ships 3.47+)
ALTER TABLE learnings DROP COLUMN source_session_id;

-- Add index for active learnings query (used by Phase 5 context builder)
CREATE INDEX IF NOT EXISTS idx_learnings_active
ON learnings(is_active, confidence DESC) WHERE is_active = 1;
```

**Fallback**: `DROP COLUMN` fails if the column is indexed, FK-referenced, used in a trigger/view, or if SQLite is pre-3.35. Python 3.14 ships SQLite 3.47+ so the version constraint is met, but a future migration or manual change could theoretically add an index on `source_session_id`. The migration **must** wrap `DROP COLUMN` in a try/except:

```python
try:
    await db.execute("ALTER TABLE learnings DROP COLUMN source_session_id")
except OperationalError:
    # Column stays as dead weight â€” no functional impact.
    # source_sessions is the authoritative field.
    logger.warning("DROP COLUMN source_session_id failed â€” column retained as dead weight")
```

No functional impact either way â€” `source_sessions` becomes the authoritative field and no code reads `source_session_id` after migration.

### 2.4 Learnings Engine (`engine.py`)

#### Core class

```python
class LearningsEngine:
    """Extract, deduplicate, and evolve project learnings.

    Lifecycle:
    1. Session ends Ã¢â€ â€™ summarizer generates summary (Phase 3)
    2. LearningsEngine.extract() called with summary + observations
    3. Each candidate learning is deduped against existing learnings
    4. New learnings inserted; duplicates merged (confidence boosted)
    5. Contradictions flagged in event_log
    """

    # Confidence evolution constants
    INITIAL_CONFIDENCE = 0.5
    MAX_AUTO_CONFIDENCE = 0.95  # Only manual confirmation reaches 1.0
    BOOST_FACTOR = 0.2          # 20% of remaining headroom per confirmation
    CONTRADICTION_PENALTY = 0.3
    MIN_CONFIDENCE = 0.1

    def __init__(
        self,
        db: aiosqlite.Connection,
        client: AsyncAnthropic,
        lance_store: Optional[LanceStore],
        logger: MemLogger,
        config: Config,
    ):
        self.db = db
        self.client = client
        self.lance_store = lance_store
        self.logger = logger
        self.config = config
```

#### Extraction flow

```python
async def extract_from_session(
    self,
    session_id: str,
    summary: str,
    observations: list[dict],
    project_path: str,
) -> list[dict]:
    """Extract learnings from a completed session.

    Args:
        session_id: The session that generated this summary
        summary: AI-generated session summary from Phase 3 summarizer
        observations: Compressed observations from the session
        project_path: For scoping existing learnings query

    Returns:
        List of learning dicts that were inserted or merged
    """
    # 1. Fetch existing learnings for context (used in prompt)
    existing = await self._get_active_learnings(project_path)

    # 2. Call Haiku to extract candidate learnings
    candidates = await self._call_extraction_api(
        summary, observations, existing
    )

    # 3. Process each candidate: dedup, merge, or insert
    results = []
    for candidate in candidates:
        result = await self._process_candidate(
            candidate, session_id, project_path
        )
        results.append(result)

    return results
```

#### Deduplication logic

```python
async def _find_duplicate(
    self,
    content: str,
    category: str,
) -> Optional[dict]:
    """Find semantically similar existing learning.

    Strategy:
    1. If lance_store available: semantic search with cosine threshold
    2. Fallback: exact substring match on content field

    Returns matching learning row or None.
    """
    if self.lance_store:
        results = await asyncio.to_thread(
            self.lance_store.search_learnings,
            query=content,
            limit=3,
        )
        for r in results:
            score = r.get("score", 0)
            # LanceDB returns distance, not similarity for some metrics.
            # With cosine metric, score IS cosine similarity (0-1).
            if score >= self.config.learning_dedup_threshold:  # default 0.90
                return r

    # Fallback: crude substring check
    row = await self.db.execute_fetchone(
        """
        SELECT * FROM learnings
        WHERE category = ? AND is_active = 1
        AND (content LIKE '%' || ? || '%' OR ? LIKE '%' || content || '%')
        LIMIT 1
        """,
        (category, content[:80], content[:80]),
    )
    return dict(row) if row else None
```

**Why 0.90 and not 0.85**: With Qwen3-Embedding-0.6B at `truncate_dim=256` (Matryoshka), the reduced dimensionality compresses the embedding space, inflating cosine similarity between loosely related texts. For short text (~20-50 tokens), this effect is amplified. Research on SemHash (2025) defaults to 0.90 for dedup, and NVIDIA NeMo's semantic dedup uses even stricter thresholds (eps 0.01-0.1 corresponding to cosine 0.9-0.99). Starting at 0.90 with configurability is safer than the architecture's 0.85.

#### Active learnings query (for extraction prompt)

```python
async def _get_active_learnings(self, project_path: str) -> list[dict]:
    """Fetch ALL active learnings for the extraction prompt.

    Unlike Phase 5's context injection (which filters at confidence >= 0.5),
    this returns all active learnings including low-confidence ones.
    This prevents the "ghost learning" problem: if we only show
    high-confidence learnings to the extraction prompt, the model
    can't see penalized/decayed learnings and may re-extract them
    as "new", resurrecting knowledge that was intentionally suppressed.

    Low-confidence learnings are marked so the model knows not to
    re-extract them without strong new evidence.
    """
    rows = await self.db.execute_fetchall(
        """
        SELECT id, category, content, confidence, times_seen
        FROM learnings
        WHERE is_active = 1
        ORDER BY confidence DESC, times_seen DESC
        LIMIT 30
        """,
    )
    return [dict(r) for r in rows] if rows else []
```

**Why return all active learnings, not just high-confidence**: Phase 5 (context injection) correctly filters at `confidence >= 0.5` â€” Claude doesn't need to see dubious learnings during coding. But the *extraction prompt* must see them to avoid re-extracting a penalized learning as "new". Without this, a contradicted learning at confidence 0.2 falls out of the prompt, gets re-extracted, matches via `_find_duplicate`, and gets boosted back up â€” a zombie loop. Showing all active learnings (capped at 30 for token budget) closes this gap.

#### Confidence evolution

```python
def _boost_confidence(self, current: float) -> float:
    """Calculate new confidence after re-observation.

    Formula: min(MAX_AUTO, current + BOOST_FACTOR Ãƒâ€” (MAX_AUTO - current))

    This produces diminishing returns:
        0.50 Ã¢â€ â€™ 0.59 Ã¢â€ â€™ 0.666 Ã¢â€ â€™ 0.733 Ã¢â€ â€™ 0.786 Ã¢â€ â€™ 0.829 Ã¢â€ â€™ ...
    Converges toward MAX_AUTO_CONFIDENCE (0.95) asymptotically.
    Only manual confirmation (`is_manual=True`) sets confidence to 1.0.
    """
    return min(
        self.MAX_AUTO_CONFIDENCE,
        current + self.BOOST_FACTOR * (self.MAX_AUTO_CONFIDENCE - current),
    )

def _penalize_confidence(self, current: float) -> float:
    """Reduce confidence after contradiction.

    Flat penalty capped at MIN_CONFIDENCE.
    """
    return max(self.MIN_CONFIDENCE, current - self.CONTRADICTION_PENALTY)
```

**Why this formula over a lookup table**: The architecture doc showed approximate values (0.5Ã¢â€ â€™0.7Ã¢â€ â€™0.85Ã¢â€ â€™0.95) but no formula. A lookup table is fragile and doesn't handle edge cases (what if current confidence is 0.6 from a manual adjustment?). The diminishing-returns formula is mathematically sound, handles any starting value, and naturally converges.

**Honest assessment of the trajectory**: The architecture's numbers (0.5Ã¢â€ â€™0.7Ã¢â€ â€™0.85Ã¢â€ â€™0.95) imply a larger boost per step than our formula produces. With `BOOST_FACTOR=0.2`, after 5 confirmations from 0.5 we reach ~0.83, not 0.95. The architecture's trajectory requires `BOOST_FACTOR=0.35+`. I chose 0.2 deliberately because the architecture's trajectory is too aggressive Ã¢â‚¬â€ a learning seen 3 times shouldn't already be at 0.85 confidence. 5-8 confirmations to reach 0.85+ feels more appropriate for a single-developer tool where sessions are frequent and the same patterns recur trivially.

If this proves too conservative in practice, bump `BOOST_FACTOR` to 0.25-0.30. It's a config constant, not a schema decision.

#### Candidate processing

```python
async def _process_candidate(
    self,
    candidate: dict,
    session_id: str,
    project_path: str,
) -> dict:
    """Process a single extracted learning candidate.

    Outcomes:
    - 'inserted': New learning, no duplicate found
    - 'merged': Duplicate found, confidence boosted, times_seen incremented
    - 'contradicted': Similar learning found but content conflicts
    - 'skipped': Extraction quality too low (confidence < 0.3)
    """
    content = candidate["content"]
    category = candidate["category"]
    extracted_confidence = candidate.get("confidence", self.INITIAL_CONFIDENCE)

    # Skip low-quality extractions
    if extracted_confidence < 0.3:
        self.logger.log("learning.skipped", {
            "content": content[:100], "reason": "low_extraction_confidence",
        })
        return {"action": "skipped", "content": content}

    # Check for duplicates
    existing = await self._find_duplicate(content, category)

    if existing is None:
        # New learning Ã¢â‚¬â€ insert
        learning_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """
            INSERT INTO learnings
            (id, category, content, confidence, times_seen,
             source_sessions, is_manual, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, 1, ?, 0, 1, ?, ?)
            """,
            (learning_id, category, content, self.INITIAL_CONFIDENCE,
             json.dumps([session_id]), now, now),
        )
        await self.db.commit()

        # Embed in LanceDB for future dedup searches
        if self.lance_store:
            await asyncio.to_thread(
                self.lance_store.add_learning,
                learning_id=learning_id,
                category=category,
                content=content,
            )

        self.logger.log("learning.inserted", {
            "id": learning_id, "category": category,
            "content": content[:100],
        })
        return {"action": "inserted", "id": learning_id, "content": content}

    # Duplicate found Ã¢â‚¬â€ check if contradictory
    if candidate.get("contradicts"):
        # 1. Penalize the OLD learning
        new_confidence = self._penalize_confidence(existing["confidence"])
        await self.db.execute(
            """
            UPDATE learnings
            SET confidence = ?, updated_at = ?
            WHERE id = ?
            """,
            (new_confidence, datetime.now(timezone.utc).isoformat(),
             existing["id"]),
        )

        # 2. Insert the NEW learning as a fresh entry
        #    Without this, contradiction kills old knowledge but loses
        #    the replacement â€” the system forgets the topic entirely.
        new_learning_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """
            INSERT INTO learnings
            (id, category, content, confidence, times_seen,
             source_sessions, is_manual, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, 1, ?, 0, 1, ?, ?)
            """,
            (new_learning_id, category, content, self.INITIAL_CONFIDENCE,
             json.dumps([session_id]), now, now),
        )
        await self.db.commit()

        # 3. Embed new learning in LanceDB for future dedup
        if self.lance_store:
            await asyncio.to_thread(
                self.lance_store.add_learning,
                learning_id=new_learning_id,
                category=category,
                content=content,
            )

        self.logger.log("learning.contradicted", {
            "existing_id": existing["id"],
            "existing_content": existing["content"][:100],
            "new_id": new_learning_id,
            "new_content": content[:100],
            "old_confidence": existing["confidence"],
            "penalized_confidence": new_confidence,
        })
        return {
            "action": "contradicted",
            "existing_id": existing["id"],
            "new_id": new_learning_id,
            "content": content,
        }

    # Same meaning Ã¢â‚¬â€ merge
    new_confidence = self._boost_confidence(existing["confidence"])
    source_sessions = json.loads(existing.get("source_sessions", "[]"))
    if session_id not in source_sessions:
        source_sessions.append(session_id)

    await self.db.execute(
        """
        UPDATE learnings
        SET confidence = ?,
            times_seen = times_seen + 1,
            source_sessions = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (new_confidence, json.dumps(source_sessions),
         datetime.now(timezone.utc).isoformat(), existing["id"]),
    )
    await self.db.commit()

    # Detect revival: a low-confidence learning being boosted back above threshold
        if existing["confidence"] < 0.5 and new_confidence >= 0.5:
            self.logger.log("learning.revived", {
                "existing_id": existing["id"],
                "old_confidence": existing["confidence"],
                "new_confidence": new_confidence,
                "content": existing["content"][:100],
            })

    self.logger.log("learning.merged", {
        "existing_id": existing["id"],
        "old_confidence": existing["confidence"],
        "new_confidence": new_confidence,
        "times_seen": existing.get("times_seen", 1) + 1,
    })
    return {
        "action": "merged",
        "existing_id": existing["id"],
        "new_confidence": new_confidence,
    }
```

### 2.5 Extraction Prompt (`prompts.py`)

```python
LEARNING_EXTRACTION_SYSTEM = """\
You extract reusable project knowledge from development session summaries.

Categories (pick exactly one per learning):
- architecture: System design decisions, module structure, API patterns
- convention: Coding style rules, naming conventions, testing patterns
- gotcha: Non-obvious bugs, silent failures, surprising behaviors
- dependency: Package versions, compatibility notes, API quirks
- pattern: Recurring implementation patterns, common workflows

Rules:
1. Only extract things useful across MULTIPLE sessions
2. Do NOT extract session-specific actions ("fixed bug in auth.py")
3. DO extract patterns ("auth module uses JWT with HttpOnly cookies")
4. Each learning should be 1-2 sentences, specific and actionable
5. If a new learning CONTRADICTS an existing one listed below, set "contradicts" to the existing learning's content
6. confidence: 0.3-0.7 based on how confident you are this is a real pattern (not a one-off)

Respond with a JSON array ONLY (no markdown fences, no preamble):
[{"category": "...", "content": "...", "confidence": 0.5, "contradicts": null}]

Return empty array [] if no learnings are worth extracting.
"""

def build_extraction_prompt(
    summary: str,
    observations: list[dict],
    existing_learnings: list[dict],
) -> str:
    """Build user message for learning extraction.

    Includes existing learnings so the model can detect contradictions
    and avoid re-extracting known knowledge.
    """
    parts = [f"## Session Summary\n{summary}\n"]

    if observations:
        parts.append("## Key Observations")
        for obs in observations[:8]:  # Cap at 8 to stay within token budget
            title = obs.get("title", "untitled")
            obs_summary = obs.get("summary", "")
            parts.append(f"- {title}: {obs_summary}")
        parts.append("")

    if existing_learnings:
        parts.append("## Existing Project Learnings (do NOT re-extract these)")
        parts.append("Learnings marked [low-confidence] have been contradicted or are unconfirmed.")
        parts.append("Do NOT re-extract [low-confidence] learnings unless you have strong new evidence.\n")
        for l in existing_learnings[:30]:  # Cap at 30 (includes low-confidence)
            cat = l.get("category", "unknown")
            content_text = l.get("content", "")
            conf = l.get("confidence", 0.5)
            prefix = "[low-confidence] " if conf < 0.5 else ""
            parts.append(f"- {prefix}[{cat}] {content_text}")
        parts.append("")

    return "\n".join(parts)
```

**Why include existing learnings in the prompt**: This serves two purposes simultaneously Ã¢â‚¬â€ (1) avoids extracting learnings that already exist (the model sees them and skips), and (2) enables contradiction detection without a separate API call. The downside is ~300 extra input tokens per call, which at Haiku 4.5 pricing ($1/MTok) costs $0.0003. Worth it.

### 2.6 Call Graph Self-Healing (`healer.py`)

```python
class CallGraphHealer:
    """Confirm or discover call graph edges from compressed observations.

    After Phase 3 compresses a tool output into an observation, the healer
    scans the observation text for function references and cross-references
    them against the call graph. This provides "evidence from usage" to
    complement Phase 2's "evidence from static analysis."

    Confidence adjustments:
    - Confirmed existing edge: diminishing boost (same formula as learnings)
    - New edge discovered: insert with confidence 0.6, source='observation'
    - Stale edge (10+ sessions without confirmation): decay by 0.05 per session
    """

    CONFIRMATION_BOOST = 0.15  # Smaller than learning boost Ã¢â‚¬â€ less signal
    NEW_EDGE_CONFIDENCE = 0.6
    STALE_DECAY = 0.05
    STALE_SESSION_THRESHOLD = 10

    def __init__(self, db: aiosqlite.Connection, logger: MemLogger):
        self.db = db
        self.logger = logger
```

#### Edge confirmation from observations

```python
async def confirm_edges_from_observation(
    self,
    observation: dict,
    session_id: str,
) -> dict:
    """Parse observation for function references, confirm call graph edges.

    Strategy:
    1. Extract function-like references from observation text
       (title, summary, detail, functions_changed)
    2. For each pair of references in the same observation,
       check if a call graph edge exists
    3. If edge exists: bump confidence + times_confirmed
    4. If edge doesn't exist but both functions are in function_map:
       insert new edge with source='observation'

    This is heuristic, not precise. An observation mentioning both
    `AuthService.authenticate` and `TokenManager.create` doesn't
    *prove* one calls the other Ã¢â‚¬â€ but it's evidence they're related.
    We use lower confidence (0.6) for observation-discovered edges
    vs AST-discovered edges (1.0).
    """
    functions_changed = json.loads(
        observation.get("functions_changed", "[]")
    )
    if len(functions_changed) < 2:
        return {"confirmed": 0, "discovered": 0}

    confirmed = 0
    discovered = 0

    # Check all pairs of functions mentioned in the observation
    for i, caller in enumerate(functions_changed):
        for callee in functions_changed[i + 1:]:
            result = await self._check_or_create_edge(
                caller, callee, session_id
            )
            if result == "confirmed":
                confirmed += 1
            elif result == "discovered":
                discovered += 1

    if confirmed or discovered:
        self.logger.log("callgraph.healed", {
            "session_id": session_id,
            "observation_id": observation.get("id"),
            "confirmed": confirmed,
            "discovered": discovered,
        })

    return {"confirmed": confirmed, "discovered": discovered}
```

#### Stale edge decay

```python
async def decay_stale_edges(self, project_path: str) -> int:
    """Decay confidence on edges not confirmed in recent sessions.

    Called once per session close (after all observations processed).
    Only affects edges with source='observation' (AST edges are
    re-confirmed on every scan and don't need decay).

    An edge is "stale" if it hasn't been confirmed in the last
    STALE_SESSION_THRESHOLD sessions. Decay is gentle Ã¢â‚¬â€ 0.05 per
    session close Ã¢â‚¬â€ so an edge at confidence 0.6 takes 10 session
    closes to reach 0.1 (where it's effectively invisible in
    context injection).
    """
    # Count recent sessions to determine which edges are stale
    recent_sessions = await self.db.execute_fetchall(
        """
        SELECT id FROM sessions
        WHERE project_path = ? AND status IN ('closed', 'summarized')
        ORDER BY started_at DESC
        LIMIT ?
        """,
        (project_path, self.STALE_SESSION_THRESHOLD),
    )
    if len(recent_sessions) < self.STALE_SESSION_THRESHOLD:
        return 0  # Not enough history to judge staleness

    oldest_recent = recent_sessions[-1]["id"]
    oldest_recent_time = await self.db.execute_fetchone(
        "SELECT started_at FROM sessions WHERE id = ?",
        (oldest_recent,),
    )

    # Decay observation-sourced edges not confirmed recently
    result = await self.db.execute(
        """
        UPDATE call_graph
        SET confidence = MAX(0.05, confidence - ?)
        WHERE source = 'observation'
        AND created_at < ?
        AND times_confirmed = 0
        """,
        (self.STALE_DECAY, oldest_recent_time["started_at"]),
    )
    await self.db.commit()

    decayed = result.rowcount
    if decayed > 0:
        self.logger.log("callgraph.decayed", {
            "edges_decayed": decayed,
            "decay_amount": self.STALE_DECAY,
        })
    return decayed
```

**Honest assessment of call graph self-healing value**: For a single-developer local tool, the value is moderate. Here's why:

1. Phase 2's static AST analysis already achieves ~80-95% accuracy on direct/self_method/import calls.
2. The remaining ~5-20% (unresolved calls) are typically dynamic dispatch, `getattr()`, or complex cross-module chains Ã¢â‚¬â€ which observation text is unlikely to clarify precisely.
3. The observation-based "confirmation" is circumstantial: two functions appearing in the same observation doesn't prove a callerÃ¢â€ â€™callee relationship.

The implementation is kept intentionally simple (pair-matching, not NLP extraction) because the ROI doesn't justify complexity. If this proves useful in practice, Phase 9 could add LLM-assisted edge extraction from observation text.

### 2.7 Worker Integration

#### Wiring into the processor pipeline

The Phase 3 processor currently runs this pipeline per queue item:

```
dequeue Ã¢â€ â€™ compress Ã¢â€ â€™ store observation Ã¢â€ â€™ [done]
```

Phase 6 extends it with post-session processing:

```
[Phase 3] dequeue Ã¢â€ â€™ compress Ã¢â€ â€™ store observation
[Phase 6] Ã¢â€ â€™ confirm call graph edges from observation

[On session summarization Ã¢â‚¬â€ Phase 3 summarizer]
[Phase 3] aggregate observations Ã¢â€ â€™ generate summary Ã¢â€ â€™ store summary
[Phase 6] Ã¢â€ â€™ extract learnings from summary + observations
[Phase 6] Ã¢â€ â€™ decay stale call graph edges
```

**Modified `processor.py`** (additions only):

```python
# In Processor.__init__
self.learnings_engine = LearningsEngine(
    db=db, client=compressor.client,
    lance_store=lance_store, logger=logger, config=config,
)
self.call_graph_healer = CallGraphHealer(db=db, logger=logger)

# After observation is stored (in process_item)
async def _post_observation(self, observation: dict, session_id: str):
    """Phase 6 post-processing after observation storage."""
    try:
        await self.call_graph_healer.confirm_edges_from_observation(
            observation, session_id
        )
    except Exception as e:
        self.logger.log("callgraph.heal_error", {
            "error": str(e), "observation_id": observation.get("id"),
        })
        # Non-fatal Ã¢â‚¬â€ don't fail the compression pipeline

# After session summary is generated (in summarize_session)
async def _post_summarization(
    self, session_id: str, summary: str, project_path: str,
):
    """Phase 6 post-processing after session summarization."""
    try:
        # Get this session's observations for the extraction prompt
        observations = await self.db.execute_fetchall(
            """
            SELECT title, summary, detail, files_touched, functions_changed
            FROM observations WHERE session_id = ?
            ORDER BY created_at
            """,
            (session_id,),
        )
        obs_list = [dict(row) for row in observations]

        # Extract learnings
        results = await self.learnings_engine.extract_from_session(
            session_id=session_id,
            summary=summary,
            observations=obs_list,
            project_path=project_path,
        )

        self.logger.log("learning.extraction_complete", {
            "session_id": session_id,
            "inserted": sum(1 for r in results if r["action"] == "inserted"),
            "merged": sum(1 for r in results if r["action"] == "merged"),
            "contradicted": sum(1 for r in results if r["action"] == "contradicted"),
            "skipped": sum(1 for r in results if r["action"] == "skipped"),
        })

        # Decay stale call graph edges
        await self.call_graph_healer.decay_stale_edges(project_path)

    except Exception as e:
        self.logger.log("learning.extraction_error", {
            "error": str(e), "session_id": session_id,
        })
        # Non-fatal Ã¢â‚¬â€ session is already summarized
```

**Critical design decision**: Both learnings extraction and call graph healing are **non-fatal**. If they fail, the observation/summary pipeline is unaffected. This is correct Ã¢â‚¬â€ the core value of claude-mem-lite is capturing and compressing tool outputs, not the learnings layer.

### 2.8 LanceDB Learning Index

Phase 4's `lance_store.py` defined a `LearningIndex` Pydantic model as a placeholder. Phase 6 populates it.

```python
# Addition to LanceStore (Phase 4's lance_store.py)

def add_learning(
    self,
    learning_id: str,
    category: str,
    content: str,
) -> None:
    """Embed and index a learning for dedup search."""
    table = self._get_or_create_table("learnings", LEARNING_SCHEMA)
    table.add([{
        "learning_id": learning_id,
        "category": category,
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }])

def search_learnings(
    self,
    query: str,
    limit: int = 5,
    category: Optional[str] = None,
) -> list[dict]:
    """Semantic search over learnings for dedup.

    Returns results with cosine similarity score.
    """
    table = self._get_or_create_table("learnings", LEARNING_SCHEMA)

    q = table.search(query).limit(limit).metric("cosine")
    if category:
        q = q.where(f"category = '{category}'")

    results = q.to_list()
    return [self._format_learning_result(r) for r in results]
```

### 2.9 CLI Commands

```python
# cli/learnings_cmd.py

@app.command()
def learnings(
    action: str = typer.Argument("list", help="list|add|edit|remove|reset"),
    category: Optional[str] = typer.Option(None, "--category", "-c"),
    content: Optional[str] = typer.Option(None, "--content"),
    learning_id: Optional[str] = typer.Option(None, "--id"),
    confidence: Optional[float] = typer.Option(None, "--confidence"),
):
    """Manage project learnings.

    Examples:
        claude-mem learnings                          # list all active
        claude-mem learnings --category gotcha        # filter by category
        claude-mem learnings add -c convention --content "Use ruff for formatting"
        claude-mem learnings edit --id abc123 --content "Updated text"
        claude-mem learnings remove --id abc123
        claude-mem learnings reset --id abc123        # reset confidence to 0.5
    """
```

Manual additions set `is_manual=True` and `confidence=1.0` Ã¢â‚¬â€ they're never subject to confidence decay or contradiction penalties.

---

## 3. Corrections to Implementation Plan

| Item | Implementation Plan | Corrected (Phase 6 PRD) | Rationale |
|------|---------------------|------------------------|-----------|
| Wiring point | "Wire into Stop hook: after session summary" | Wire into worker processor, after `summarize_session()` | Stop hook fires on every response (Phase 1 PRD Ã‚Â§5.3), not on session end. Learnings extraction is an async worker task, not a sync hook operation. |
| Call graph columns | "Enhanced call_graph table with confidence, times_confirmed, source columns" | **No change needed** Ã¢â‚¬â€ columns already in Phase 0 schema | Phase 0 PRD Ã‚Â§3 already defines these columns. |
| Dedup threshold | "cosine > 0.85" | Default 0.90, configurable | Research shows 0.85 is too aggressive for short text with 256-dim Matryoshka embeddings. See Ã‚Â§2.4 rationale. |
| Confidence values | "new=0.5, seen again=+boost, contradicted=-0.3, manual=1.0" | Formula-based: `min(0.95, c + 0.2 Ãƒâ€” (0.95 - c))` | "+boost" is undefined. Concrete formula with diminishing returns. See Ã‚Â§2.4 rationale. |
| Stale edge threshold | "10+ sessions without confirmation" | 10+ sessions, but only for `source='observation'` edges | AST-sourced edges are re-confirmed on every file scan (Phase 2). Only observation-sourced edges can go stale. |
| Schema | Not flagged | Migration needed for `learnings` table | Phase 0 schema missing `times_seen`, `source_sessions`, `is_manual`. |

---

## 4. Integration Points

### 4.1 Phase 3 (Worker)

**Files modified**: `worker/processor.py`, `worker/summarizer.py`

Processor gains `_post_observation()` hook (call graph healing) and `_post_summarization()` hook (learning extraction + stale edge decay). Both are non-fatal wrappers.

### 4.2 Phase 4 (LanceDB)

**File modified**: `storage/lance_store.py`

Add `add_learning()` and `search_learnings()` methods. The learning embedding table is created on first `add_learning()` call, same pattern as observations.

### 4.3 Phase 5 (Context Injection)

**No changes needed.** Phase 5's `_build_learnings()` layer already queries:

```sql
SELECT category, content, confidence FROM learnings
WHERE is_active = TRUE AND confidence >= 0.5
ORDER BY confidence DESC, times_seen DESC LIMIT 10
```

Phase 6 populates this table. Phase 5 consumes it. Clean separation.

**Note**: Phase 5's query uses `times_seen` which didn't exist in the Phase 0 schema. This is another confirmation that the schema migration in Ã‚Â§2.3 is necessary for Phase 5 to work correctly too.

### 4.4 Future Phases

- **Phase 7 (Eval)**: Learning extraction events are logged to `event_log` with type `learning.*`. Phase 7 can query these for extraction quality metrics.
- **Phase 8 (CLI Reports)**: `claude-mem report` can include a "Top Learnings" section showing highest-confidence entries.
- **Phase 9 (Hardening)**: Cross-project learnings, automated pruning of low-confidence learnings after N sessions.

---

## 5. Test Plan

### 5.1 Test categories

| Category | Tests | What it validates |
|----------|-------|-------------------|
| **Migration** | 3 | Schema migration applies cleanly, data migrates from `source_session_id` to `source_sessions`, rollback doesn't break existing queries |
| **Engine Ã¢â‚¬â€ extraction** | 4 | API call with valid response Ã¢â€ â€™ parsed correctly, empty response Ã¢â€ â€™ no crash, malformed JSON Ã¢â€ â€™ logged and skipped, API error Ã¢â€ â€™ non-fatal |
| **Engine Ã¢â‚¬â€ dedup** | 5 | New learning inserted, duplicate merged with confidence boost, contradiction penalized, dedup threshold respected (below threshold = new), fallback substring match when lance_store=None |
| **Engine Ã¢â‚¬â€ confidence** | 4 | Boost formula produces correct values, penalty clamps at MIN_CONFIDENCE, manual override sets 1.0, multiple boosts converge toward MAX_AUTO |
| **Healer Ã¢â‚¬â€ edge confirmation** | 4 | Existing edge confirmed Ã¢â€ â€™ times_confirmed++, new edge from observation Ã¢â€ â€™ inserted at 0.6, single function observation Ã¢â€ â€™ no-op, error in healing Ã¢â€ â€™ non-fatal |
| **Healer Ã¢â‚¬â€ stale decay** | 3 | Stale observation edges decayed, AST edges untouched, insufficient session history Ã¢â€ â€™ no decay |
| **Worker integration** | 3 | Post-observation hook fires after compression, post-summarization fires after summary, exceptions don't break pipeline |
| **CLI** | 4 | List learnings, add manual learning (confidence=1.0), edit content, remove (soft delete) |
| **LanceDB** | 3 | Add learning embeds correctly, search returns relevant results, category filter works |
| **Total** | **33** | |

### 5.2 Test infrastructure

```python
# conftest.py additions for Phase 6

@pytest.fixture
def mock_haiku_learnings():
    """Mock Haiku response with extracted learnings."""
    return '[{"category":"convention","content":"Uses pytest fixtures with tmp_path","confidence":0.5,"contradicts":null}]'

@pytest.fixture
async def learnings_engine(async_db, mock_anthropic, tmp_config):
    """LearningsEngine with mocked dependencies."""
    engine = LearningsEngine(
        db=async_db,
        client=mock_anthropic,
        lance_store=None,  # Test without LanceDB by default
        logger=MemLogger(tmp_config.log_dir),
        config=tmp_config,
    )
    return engine

@pytest.fixture
async def seeded_learnings(async_db):
    """DB with pre-existing learnings for dedup tests."""
    now = datetime.now(timezone.utc).isoformat()
    learnings = [
        ("l1", "architecture", "Uses FastAPI with SQLAlchemy ORM",
         0.7, 3, '["s1","s2","s3"]', 0, 1, now, now),
        ("l2", "convention", "JWT tokens in HttpOnly cookies",
         0.85, 5, '["s1","s2","s3","s4","s5"]', 0, 1, now, now),
        ("l3", "gotcha", "user_service.get_by_email returns None on errors",
         0.5, 1, '["s1"]', 0, 1, now, now),
    ]
    for l in learnings:
        await async_db.execute(
            """INSERT INTO learnings
            (id, category, content, confidence, times_seen,
             source_sessions, is_manual, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            l,
        )
    await async_db.commit()
    return learnings
```

### 5.3 Key test scenarios

**Dedup tests** (critical path):

- Insert learning "Uses FastAPI with SQLAlchemy" when "Uses FastAPI with SQLAlchemy ORM" exists at cosine 0.93 Ã¢â€ â€™ merge, not insert
- Insert learning "Uses Django with raw SQL" when "Uses FastAPI with SQLAlchemy ORM" exists Ã¢â€ â€™ cosine ~0.6 Ã¢â€ â€™ new insert
- Insert learning "Auth uses session cookies" when "JWT tokens in HttpOnly cookies" exists with `contradicts: "JWT tokens in HttpOnly cookies"` â†’ penalize existing, **insert new learning** as fresh entry with confidence 0.5


**Ghost learning / revival tests** (verify low-confidence learnings don't resurrect):

- Learning L1 at confidence 0.2 (decayed) is included in extraction prompt with `[low-confidence]` marker
- Learning L1 at confidence 0.2 gets re-extracted â†’ `_find_duplicate` matches â†’ merge boosts to 0.29 â†’ `learning.revived` NOT logged (still below 0.5)
- Learning L1 at confidence 0.4 gets re-extracted â†’ boost to 0.49 â†’ still below 0.5, no revival log
- Learning L1 at confidence 0.45 gets re-extracted â†’ boost to 0.54 â†’ `learning.revived` logged

**Confidence tests** (verify formula):

- Starting at 0.5: after 1 boost Ã¢â€ â€™ 0.59, after 2 Ã¢â€ â€™ 0.666, after 5 Ã¢â€ â€™ 0.829
- Starting at 0.8 (already high): after 1 boost Ã¢â€ â€™ 0.83
- After penalty from 0.7 Ã¢â€ â€™ 0.4
- After penalty from 0.2 Ã¢â€ â€™ 0.1 (clamped)

**Call graph healing tests**:

- Observation with `functions_changed: ["AuthService.authenticate", "TokenManager.create"]` and existing edge between them Ã¢â€ â€™ confirmed
- Same pair with no existing edge Ã¢â€ â€™ new edge inserted at 0.6
- Observation with single function Ã¢â€ â€™ no-op (need at least 2 for pair matching)

### 5.4 Performance targets

| Operation | Target | Rationale |
|-----------|--------|-----------|
| Learning extraction API call | <2000ms | Same as Phase 3 compression |
| Dedup search (LanceDB) | <20ms | Small table (<1000 learnings), 256-dim vectors |
| Dedup search (SQLite fallback) | <5ms | Simple LIKE query on small table |
| Confidence update | <2ms | Single UPDATE on indexed table |
| Call graph edge confirmation | <5ms per pair | SELECT + UPDATE per pair, small table |
| Stale edge decay | <10ms | Bulk UPDATE with WHERE clause |
| Full post-summarization pipeline | <3000ms | API call dominates |

---

## 6. Acceptance Criteria

Phase 6 is complete when:

- [ ] Schema migration adds `times_seen`, `source_sessions`, `is_manual` to `learnings` table and migrates existing data
- [ ] LearningsEngine extracts learnings from session summary via Haiku 4.5 API
- [ ] New learnings inserted with `confidence=0.5`, `times_seen=1`, `source_sessions=[session_id]`
- [ ] Duplicate learnings (cosine Ã¢â€°Â¥ 0.90) merged: confidence boosted via diminishing-returns formula, `times_seen` incremented
- [ ] Contradictory learnings penalize existing entry's confidence by 0.3, insert the new (contradicting) learning as a fresh entry with `confidence=0.5`, and log `learning.contradicted` event with both old and new learning IDs
- [ ] Manual learnings via CLI set `is_manual=True`, `confidence=1.0`, immune to automated decay
- [ ] CallGraphHealer confirms existing edges and discovers new ones from observation `functions_changed`
- [ ] Stale observation-sourced edges decay by 0.05 per session close after 10 sessions without confirmation
- [ ] All Phase 6 operations are non-fatal Ã¢â‚¬â€ errors are logged but don't break the compression/summarization pipeline
- [ ] Learnings embedded in LanceDB for semantic dedup search
- [ ] Phase 5 context injection picks up Phase 6 learnings without any Phase 5 code changes
- [ ] `claude-mem learnings` CLI command lists, adds, edits, removes learnings
- [ ] All 33 tests pass (pytest + pytest-asyncio, <20s total with mocked API)
- [ ] `ruff check` and `ruff format --check` pass with zero warnings

---

## 7. Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Extraction prompt produces low-quality learnings** | Medium | Medium | Include existing learnings in prompt so model avoids duplicates. Extraction confidence threshold (Ã¢â€°Â¥0.3) filters garbage. Phase 7 eval can measure extraction quality systematically. |
| **Dedup threshold too strict (0.90) Ã¢â‚¬â€ misses duplicates** | Low | Low | Configurable via `Config.learning_dedup_threshold`. If duplicates accumulate, lower to 0.85. CLI `claude-mem learnings` lets you manually merge. |
| **Dedup threshold too loose Ã¢â‚¬â€ merges distinct learnings** | Low | Medium | Start conservative at 0.90. SemHash research suggests this is a reasonable default. Manual inspection via `claude-mem learnings` catches false merges. |
| **Contradiction detection unreliable** | Medium | Low | LLM-based contradiction detection is approximate. False positives (flagging non-contradictions) reduce confidence unnecessarily but don't delete data. False negatives (missing contradictions) leave stale learnings that eventually decay via low `times_seen`. |
| **Call graph healing too noisy** | Low | Low | Pair-matching is intentionally conservative Ã¢â‚¬â€ only matches functions in the same `functions_changed` array. `source='observation'` edges are separate from `source='ast'` edges and have lower starting confidence (0.6 vs 1.0). |
| **LanceDB learning table grows unbounded** | Low | Low | Learnings are short text (~50 tokens each). Even 10,000 learnings is <5MB in LanceDB. Phase 9 can add pruning for `is_active=False` learnings. |

---

## 8. Open Questions

| Question | Current assumption | When to resolve |
|----------|-------------------|-----------------|
| **Should learnings expire after N sessions without re-confirmation?** | No automatic expiration Ã¢â‚¬â€ only confidence decay. Learnings at confidence 0.1 are effectively invisible (Phase 5 filters at Ã¢â€°Â¥0.5) but remain in DB. | Phase 9 hardening. Monitor DB growth over 50+ sessions first. |
| **Should the extraction prompt use extended thinking?** | No Ã¢â‚¬â€ Haiku 4.5 with plain prompting is sufficient for structured extraction from short text. Extended thinking adds latency and cost for minimal benefit. | Revisit if extraction quality is poor in practice. |
| **Should we batch learning extraction across multiple sessions?** | No Ã¢â‚¬â€ one extraction call per session close. Sessions close at most a few times per day. Batching adds complexity for no meaningful cost savings. | Revisit only if API latency is problematic (unlikely at <2s). |
| **Should `BOOST_FACTOR` be higher (0.25-0.30)?** | 0.20 for conservative confidence growth. Architecture's trajectory implies 0.35+ but that reaches high confidence too quickly for frequent-session usage. | Tune after 20+ real sessions. The factor is a single constant, trivial to change. |
| **Call graph healing: should we use LLM to extract edges from observation text?** | No Ã¢â‚¬â€ pair-matching from `functions_changed` only. Simple, fast, no API cost. | Phase 9 if pair-matching proves too crude. |
