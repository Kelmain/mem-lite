# claude-mem-lite — Implementation Plan

All architecture decisions are resolved. This plan sequences the build with clear dependencies, deliverables, and validation criteria per phase.

---

## Dependency Graph

```
Phase 0: Storage Layer
    │
    ├──► Phase 1: Hook Scripts + Direct Capture
    │       │
    │       ├──► Phase 3: Worker Service + Compression
    │       │       │
    │       │       ├──► Phase 4: Embeddings + Search
    │       │       │       │
    │       │       │       ├──► Phase 5: Context Injection
    │       │       │       │
    │       │       │       └──► Phase 6: Learnings + Call Graph Self-Healing
    │       │       │
    │       │       └──► Phase 7: A/B Testing + Eval
    │       │
    │       └──► Phase 8: CLI Reports
    │
    └──► Phase 2: AST Tracker (independent, can parallelize with Phase 1)
```

---

## Phase 0 — Storage Layer (~1 session)

**Goal**: Foundation that everything plugs into. SQLite schema, LanceDB init, migration system.

**Build**:
- `storage/sqlite_store.py` — CRUD operations via stdlib `sqlite3`
  - WAL mode enabled, `PRAGMA journal_mode=WAL`
  - `PRAGMA user_version` migration system (~30 lines)
  - All tables from architecture spec (sessions, observations, function_map, call_graph, learnings, pending_queue, event_log)
- `storage/lance_store.py` — LanceDB table creation (no embedding model yet, just schema)
  - `ObservationIndex`, `SummaryIndex`, `LearningIndex` Pydantic models
  - Placeholder for embedding function (wired in Phase 4)
- `storage/models.py` — Pydantic models for internal data passing
- `config.py` — Settings, paths (`~/.claude-mem/`), defaults
- `logging/logger.py` — `MemLogger` (JSON file + SQLite event_log, `timed()` context manager)

**Validate**:
- Unit tests: CRUD all tables, migration from v0→v1→v2
- WAL mode concurrent read/write test (simulating hook writes + worker reads)
- Logger writes to both file and SQLite

**Deliverables**: `test_sqlite_store.py`, `test_lance_store.py`, `test_migrations.py`

---

## Phase 1 — Hook Scripts + Direct Capture (~1-2 sessions)

**Goal**: End-to-end data flow with Claude Code. Hooks fire, raw data lands in SQLite. No AI, no worker yet.

**Build**:
- `plugin/hooks.json` — Hook registration for Claude Code
- `plugin/scripts/capture-hook.py` — PostToolUse handler
  - **Direct SQLite write** (<5ms): INSERT into `pending_queue` with `status='raw'`
  - Extract `files_touched` from hook event payload
  - Extract `tool_name` (Write, Edit, Bash, Read, etc.)
  - No HTTP, no worker dependency
- `plugin/scripts/context-hook.py` — SessionStart handler
  - Creates session record in SQLite (direct write)
  - Placeholder for context injection (returns empty string until Phase 5)
  - If worker running: call `GET /api/context` for rich context
  - If worker not running: return basic session list from SQLite
- `plugin/scripts/summary-hook.py` — Stop handler
  - Placeholder: marks session needs summary in SQLite
  - Actual AI summarization wired in Phase 3
- `plugin/scripts/cleanup-hook.py` — SessionEnd handler
  - Marks session `status='closed'` in SQLite (direct write)

**Validate**:
- Install hooks in Claude Code, run a real session
- Verify: observations appear in `pending_queue`, session created/closed
- Measure hook latency (target: <10ms for capture, <5ms for cleanup)

**Deliverables**: Working hooks with Claude Code, `test_hooks.py` (mocking hook events)

---

## Phase 2 — AST Tracker (~2 sessions)

**Goal**: Deterministic, testable code intelligence. No AI dependency.

**Build**:
- `ast_tracker/extractor.py` — `FunctionExtractor` class
  - Extract functions, methods, async variants, classes
  - Build signatures with type annotations
  - Extract decorators, first-line docstrings
  - Body hash via `md5(ast.dump(node))`
- `ast_tracker/call_graph.py` — Call extraction + resolution
  - `_resolve_call()` with 4 resolution types: direct, self_method, import, unresolved
  - Import map building from `ast.Import`/`ast.ImportFrom`
  - Filter noise (builtins, stdlib)
- `ast_tracker/diff.py` — Change detection
  - Compare snapshots by qualified_name + body_hash
  - Classify: new, modified, unchanged, deleted
- `ast_tracker/mermaid.py` — Diagram generation
  - Per-file and cross-file call graphs
  - Confidence-based line styles (solid/dashed)
- Wire into capture hook: scan `.py` files on PostToolUse

**Validate**:
- Unit tests against real Python files (10+ test cases covering classes, async, decorators, nested functions)
- Resolution accuracy test: known codebase with manually verified call graph
- Mermaid output renders correctly
- Change detection: modify function body → detected, move function → detected

**Deliverables**: `test_extractor.py`, `test_call_graph.py`, `test_diff.py`

---

## Phase 3 — Worker Service + Compression (~2-3 sessions)

**Goal**: FastAPI worker processes the queue. AI compression turns raw output into structured observations.

**Build**:
- `worker/server.py` — FastAPI app with uvicorn
  - UDS binding: `uvicorn.run(app, uds="~/.claude-mem/worker.sock")`
  - PID file at `~/.claude-mem/worker.pid`
  - Idle timeout: 30 min auto-shutdown
  - Lifespan: pre-load embedding model (placeholder until Phase 4)
  - Health endpoint: `GET /api/health`
- `worker/processor.py` — Queue consumer
  - Poll `pending_queue` for `status='raw'` entries
  - Orchestrate: compress → store observation → update queue status
  - Retry logic: 3 attempts, exponential backoff
  - Uses `aiosqlite` for all DB operations
- `worker/compressor.py` — AI compression via raw Anthropic API
  - `anthropic.AsyncAnthropic().messages.create()` (NOT Agent SDK)
  - Compression prompt: raw tool output → structured JSON (title, summary, detail, files_touched, functions_changed)
  - Model: claude-haiku-4-5 default
  - **Prompt engineering sub-task**: iterate on compression prompt quality
    - Capture 20-30 real tool outputs manually
    - Compress with different prompt variants
    - Score: does compressed version preserve actionable information?
- Worker startup script: `claude-mem-worker start --daemon`
- Shell alias: `alias cc='claude-mem-worker start --daemon && claude'`

**Validate**:
- Worker starts, accepts connections over UDS, processes queue items
- Compression produces valid structured output for diverse tool types (Write, Edit, Bash, Read)
- Queue retry works on simulated API failures
- Idle timeout shuts worker down cleanly
- Graceful degradation: API down → items stay in queue with `status='raw'`

**Deliverables**: `test_compressor.py`, `test_processor.py`, compression prompt in `worker/prompts.py`

---

## Phase 4 — Embeddings + Search (~2 sessions)

**Goal**: Hybrid search working. Observations are embedded and retrievable.

**Build**:
- Wire Qwen3-Embedding-0.6B into `storage/lance_store.py`
  - `sentence-transformers` with `truncate_dim=256` (Matryoshka)
  - Instruction-aware prefixes per query type (observation/code/learning/document)
  - Pre-load at worker startup (update lifespan in `server.py`)
- `search/hybrid.py` — Hybrid search orchestration
  - LanceDB hybrid search: vector + FTS via Tantivy, RRF reranking
  - Fallback: FTS-only if Qwen3 fails to load
  - Search endpoints on worker:
    - `GET /api/search?q=...&limit=5` — hybrid search observations
    - `GET /api/callgraph?file=...` — call graph for a file
    - `GET /api/function-history?name=...` — function evolution timeline
    - `GET /api/observation/{id}` — progressive disclosure drill-down
    - `GET /api/learnings?category=...` — filtered learnings
- Update processor: after compression, embed observation title → LanceDB
- `plugin/skills/mem-search/SKILL.md` — Search skill for Claude
  - Teaches Claude to `curl` the worker's search endpoints
  - ~30 tokens description cost, ~200-500 tokens when loaded

**Validate**:
- Embed 50+ observations, search returns relevant results
- Hybrid search outperforms FTS-only on semantic queries
- FTS fallback works when embeddings are unavailable
- Search latency: <50ms for typical queries
- Skill activates correctly in Claude Code

**Deliverables**: `test_search.py`, `test_embeddings.py`, working SKILL.md

---

## Phase 5 — Context Injection (~1-2 sessions)

**Goal**: Claude gets useful context from past sessions automatically at SessionStart.

**Build**:
- `context/builder.py` — Progressive disclosure with token budgeting
  - **Layer 1** (~300 tokens): Session index (recent session summaries)
  - **Layer 2** (~500 tokens): Function map of recently changed files
  - **Layer 3** (~300 tokens): Active learnings (top by confidence)
  - **Layer 4** (~600 tokens): Relevant observations via semantic search
  - **Layer 5** (~300 tokens): Call graph context (on-demand only)
  - Token counting: tiktoken or simple word-based estimation
  - Budget: 2000 tokens default, configurable
- Wire into `context-hook.py`:
  - Call `GET /api/context` on worker
  - Worker runs context builder, returns injection string
  - Hook returns injection to Claude Code
- `GET /api/context` endpoint on worker

**Validate**:
- Context injection stays within 2000 token budget
- Layers are prioritized correctly (session index always included, call graph only if budget allows)
- Context is actually useful: does Claude reference injected context in its responses?
- Latency: context injection <500ms total (search + build)

**Deliverables**: `test_context_builder.py`, real-world validation across 5+ sessions

---

## Phase 6 — Learnings Engine + Call Graph Self-Healing (~2 sessions)

**Goal**: System gets smarter over time. Learnings accumulate, call graph accuracy improves.

**Build**:
- `learnings/engine.py` — Learning extraction and evolution
  - Extract learnings from session summaries via Claude API
  - Categories: architecture, convention, gotcha, dependency, pattern
  - Dedup via semantic similarity (cosine > 0.85 against existing learnings)
  - Confidence evolution: new=0.5, seen again=+boost, contradicted=-0.3, manual=1.0
- `learnings/prompts.py` — Extraction prompts
- Wire into Stop hook: after session summary, extract learnings
- Call graph self-healing (from architecture-decisions-round2):
  - `confirm_edges_from_observation()`: parse compressed observations for function references
  - Cross-reference against call graph edges
  - Confirmed edge: confidence += diminishing boost
  - New edge discovered: add with confidence 0.6
  - Stale edge (10+ sessions without confirmation): decay confidence
  - Enhanced `call_graph` table with `confidence`, `times_confirmed`, `source` columns
- Wire into processor: after compression, check for call graph confirmations

**Validate**:
- Learnings deduplicate correctly (same learning from different sessions → single entry with higher confidence)
- Contradictory learnings flagged and logged
- Call graph confidence increases after confirmed observations
- Call graph resolution rate improves over simulated multi-session test

**Deliverables**: `test_learnings.py`, `test_call_graph_healing.py`

---

## Phase 7 — A/B Testing + Eval Framework (~1 session)

**Goal**: Data-driven compression model selection. Measurable quality metrics.

**Build**:
- `worker/compressor.py` — A/B routing
  - `CompressionRouter`: randomized 50/50 Haiku/Sonnet assignment per observation
  - `CompressionScore`: compression_ratio, latency_ms, cost_estimate, info_preservation, semantic_similarity
  - Log `compress.ab_result` events with model assignment + scores
- Offline eval script: `eval/compression_eval.py`
  - Info preservation scoring: generate questions from original, answer from compressed, grade
  - Semantic similarity: embedding cosine between original and compressed
- SQL analysis queries in `eval/queries.sql`
  - Head-to-head model comparison
  - Token budget usage analysis
  - AST resolution rate tracking
  - Search quality (retrieval hit rate)
- `claude-mem eval --compression` CLI command

**Validate**:
- A/B split produces even distribution across models
- Info preservation scoring correlates with human judgment (spot-check 20 observations)
- Eval dashboard renders correctly

**Deliverables**: `eval/compression_eval.py`, `eval/queries.sql`

---

## Phase 8 — CLI Reports (~1 session)

**Goal**: Terminal-based visibility into system health and project state.

**Build**:
- `cli/report.py` — `claude-mem report`
  - Session summary (today's sessions, observation count, function changes)
  - Latest session details
  - Function changes table (NEW/MOD/DEL)
  - Top learnings by confidence
  - Rich terminal output
- `cli/report.py --mermaid` — Call graph as mermaid diagram
  - Per-file subgraphs
  - Confidence-based line styles
  - Output to stdout or file
- `cli/report.py --eval` — Performance dashboard
  - Compression efficiency (ratio, latency, cost)
  - Context injection stats (tokens used vs budget)
  - Search quality (latency, result count)
  - AST resolution rate
  - Learning stats
- `cli/search_cmd.py` — `claude-mem search <query>`
  - CLI wrapper around hybrid search

**Validate**:
- Reports render correctly in terminal
- Mermaid output valid (paste into mermaid.live)
- Eval dashboard shows real data from previous sessions

**Deliverables**: Working CLI commands

---

## Phase 9 — Hardening + Polish (ongoing)

**Goal**: Production-ready for daily use.

**Tasks**:
- Enable tiered compression: Haiku for observations, Sonnet for summaries (based on A/B results)
- ONNX INT8 quantization for Qwen3 if latency is an issue (~300MB, ~2x CPU speedup)
- Error recovery edge cases: mid-session worker crash, corrupt SQLite, LanceDB index corruption
- Worker watchdog (5 lines: check PID file, restart if dead)
- Disk cleanup policy: rotate old observations, cap SQLite size
- Multi-project support: per-project databases in `~/.claude-mem/projects/`
- `pyproject.toml` finalization, `README.md`, installation docs

---

## Summary

| Phase | Effort | Dependencies | Key Risk |
|-------|--------|-------------|----------|
| 0: Storage | ~1 session | None | Low — pure plumbing |
| 1: Hooks + Capture | ~1-2 sessions | Phase 0 | Medium — Claude Code hook mechanics |
| 2: AST Tracker | ~2 sessions | None (can parallel Phase 1) | Low — deterministic, testable |
| 3: Worker + Compression | ~2-3 sessions | Phase 0, 1 | **High — compression prompt quality** |
| 4: Embeddings + Search | ~2 sessions | Phase 3 | Medium — Qwen3 model load, search quality |
| 5: Context Injection | ~1-2 sessions | Phase 4 | Medium — token budgeting, usefulness |
| 6: Learnings + Self-Healing | ~2 sessions | Phase 4 | Medium — dedup accuracy, confidence tuning |
| 7: A/B + Eval | ~1 session | Phase 3 | Low — infrastructure |
| 8: CLI Reports | ~1 session | Phase 0+ | Low — presentation layer |
| 9: Hardening | Ongoing | All | Medium — edge cases |

**Total estimated: ~15-19 sessions**

**Critical path**: Phase 0 → 1 → 3 → 4 → 5 (storage → capture → compress → search → inject)

**Highest risk**: Phase 3 (compression prompt quality). Bad compression = useless observations = wasted context tokens. Budget extra iteration time here.
