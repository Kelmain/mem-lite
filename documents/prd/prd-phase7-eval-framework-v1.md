# Phase 7 PRD: Eval Framework + Compression Benchmarking

**Version**: 1.0
**Date**: 2026-02-08
**Depends on**: Phase 0 (storage), Phase 3 (compressor), Phase 4 (embeddings), Phase 5 (context injection), Phase 6 (learnings)
**Estimated effort**: ~1 session

---

## 1. Scope

### 1.1 What this phase builds

An **offline evaluation framework** for measuring compression quality, search effectiveness, context injection efficiency, and overall system health. The framework answers one central question: **"Is the compressed observation good enough to preserve the information a future session needs?"**

Components:
- `eval/evaluator.py` â€” Compression quality scoring (deterministic + LLM-judge)
- `eval/benchmark.py` â€” A/B model comparison runner (offline replay)
- `eval/queries.sql` â€” Reference SQL queries for system monitoring
- `cli/eval_cmd.py` â€” CLI entry point: `claude-mem eval`

### 1.2 What this phase does NOT build

- **Online A/B routing** â€” deferred (see Â§3 for why)
- **Automated prompt optimization** â€” out of scope. Prompt iteration is manual, informed by eval results.
- **Real-time dashboards** â€” CLI-only. Phase 8 builds richer reporting.
- **Regression testing CI** â€” future hardening (Phase 9). This phase builds the scoring primitives.

### 1.3 Why this matters

Phase 3 introduced compression with a single model (Haiku 4.5) and a v1 prompt. We have no quantitative measure of whether that compression is good. Phase 7 closes this gap:

1. **Quality metrics** â€” objective scoring of every compressed observation
2. **Model comparison** â€” data-driven answer to "should we use Sonnet for summaries?"
3. **Cost-quality tradeoff** â€” normalized comparison: quality-per-dollar
4. **System monitoring** â€” SQL queries to track health across all phases

---

## 2. Corrections to Implementation Plan

The implementation plan proposes an approach that doesn't fit a single-developer local tool. Here are the corrections and their rationale.

| Item | Implementation Plan | Corrected (Phase 7 PRD) | Rationale |
|------|---------------------|------------------------|-----------|
| A/B routing | `CompressionRouter`: randomized 50/50 Haiku/Sonnet per observation in production | **Offline replay**: run both models on stored `raw_output` from `pending_queue` | Online A/B at ~50 obs/day needs weeks for significance. Half your production costs 3Ã— higher during experiment. Offline replay is cheaper, faster, deterministic. |
| Eval method | "Generate questions from original, answer from compressed, grade" | QAG-based scoring with Sonnet 4.5 as judge + deterministic checks as primary metrics | Plan didn't specify judge model. Research confirms QAG (question-answer generation) is best for info preservation. Deterministic checks (JSON validity, field presence, ratio) are cheaper and equally important. |
| `CompressionScore` dimensions | `compression_ratio, latency_ms, cost_estimate, info_preservation, semantic_similarity` | Drop `semantic_similarity` (rawâ†’compressed embedding comparison is unreliable for different-length texts). Add `structural_validity`, `decision_rationale_present`. | Embedding cosine between 500KB raw and 500-token compressed is meaningless â€” they're in completely different semantic spaces. QAG-based info_preservation is the correct automated quality metric. |
| Where it lives | `worker/compressor.py` (CompressionRouter class) | `eval/` directory. No production code changes. | Evaluation is offline analysis, not production routing. Keep compressor simple. |
| Output format | "Eval dashboard renders correctly" | CLI table output + optional JSON export | No dashboard to render. Terminal output via `rich` tables. JSON for scripting/analysis. |
| File location | `eval/compression_eval.py` | `eval/evaluator.py` + `eval/benchmark.py` | Separated: evaluator scores individual observations, benchmark orchestrates model comparison. |

### 2.1 Why not online A/B?

The implementation plan's `CompressionRouter` randomly assigns Haiku or Sonnet per observation during normal usage. This is standard practice for high-volume services (millions of requests/day). For claude-mem-lite:

- **Volume**: ~50 observations/day at most
- **Statistical significance**: At 50/day with 50/50 split, you need ~2 weeks minimum for meaningful p-values on quality dimensions
- **Cost**: During the experiment, half your compressions cost 3Ã— more ($0.008 vs $0.024 per call)
- **Better alternative**: Take 50 stored raw outputs from `pending_queue` (where `raw_output` is preserved), run both models offline in one session, get instant comparison

Offline replay gives you:
- 100% of observations evaluated by both models (paired comparison, not independent samples)
- No production cost increase
- Deterministic: same inputs â†’ reproducible comparison
- Run once, decide, move on

### 2.2 Why Sonnet 4.5 as judge (not self-evaluation)?

Research (LLM-as-judge literature, QAG scoring methodology) shows:

- **Self-evaluation** (Haiku judging Haiku) introduces self-preference bias
- **Stronger-model-as-judge** is standard practice â€” use a model one tier above the evaluated model
- **Sonnet 4.5 at $3/$15 per MTok** as judge: evaluating 50 observations costs ~$0.15 total (cheap)
- **QAG approach** (generate questions from raw, answer from compressed, verify) uses confined yes/no answers, reducing judge hallucination risk

### 2.3 Why drop semantic_similarity?

The implementation plan proposes embedding cosine between raw output and compressed observation. This is a common mistake:

- Raw output: 10KBâ€“500KB of tool output (file contents, terminal output, error logs)
- Compressed observation: ~500 tokens of structured summary
- These are **fundamentally different text types** â€” one is source material, the other is metadata
- Embedding models produce similar-looking vectors but the cosine score is meaningless across genre boundaries
- Qwen3-Embedding-0.6B with 256-dim Matryoshka is optimized for similarity within the same text type (queryâ†”document), not across rawâ†”summary

**What works instead**: QAG-based information preservation. Generate factual questions from the raw output, check if the compressed version can answer them. This measures actual information survival, not vector space proximity.

---

## 3. Evaluation Dimensions

### 3.1 Scoring taxonomy

Each compressed observation receives a `CompressionScore` with these dimensions:

| Dimension | Type | Range | How measured | Cost |
|-----------|------|-------|-------------|------|
| `structural_validity` | Deterministic | 0.0 or 1.0 | Valid JSON, required fields present, types correct | Free |
| `compression_ratio` | Deterministic | float | `len(raw_output) / tokens_compressed` | Free |
| `title_quality` | Deterministic | 0.0â€“1.0 | Length 3â€“15 words, imperative mood (heuristic), no period | Free |
| `info_preservation` | LLM-judge (QAG) | 0.0â€“1.0 | Generate 5 questions from raw, answer from compressed, fraction correct | ~$0.003/obs |
| `decision_rationale` | LLM-judge (binary) | 0.0 or 1.0 | Does summary explain WHY, not just WHAT? | Included in QAG call |
| `latency_ms` | Measured | int | API call duration (from `compress.done` event) | Free |
| `cost_usd` | Calculated | float | `(tokens_in Ã— input_rate + tokens_out Ã— output_rate) / 1_000_000` | Free |

**Composite score** (for model comparison):

```
quality = (0.15 Ã— structural_validity 
         + 0.10 Ã— title_quality
         + 0.50 Ã— info_preservation 
         + 0.25 Ã— decision_rationale)

cost_adjusted_quality = quality / cost_usd  # quality per dollar
```

Weights reflect our priorities: information preservation dominates (0.50), decision rationale is important for future sessions (0.25), structural validity and title quality are hygiene factors.

### 3.2 Deterministic checks (free, run on every observation)

```python
@dataclass
class DeterministicScore:
    structural_validity: float   # 1.0 if valid JSON + required fields
    compression_ratio: float     # raw_chars / compressed_tokens
    title_quality: float         # heuristic scoring
    cost_usd: float              # calculated from token counts
    latency_ms: int              # from event_log

def score_deterministic(
    observation: Observation,
    raw_output: str,
    latency_ms: int,
    model: str,
) -> DeterministicScore:
    """Score an observation without any API calls."""
    
    # Structural validity
    structural = 1.0
    if not observation.title or not observation.summary:
        structural = 0.0
    
    # Compression ratio
    ratio = len(raw_output) / max(observation.tokens_compressed, 1)
    
    # Title quality (heuristic)
    title_score = _score_title(observation.title)
    
    # Cost
    rates = MODEL_RATES[model]  # {"input": 1.0, "output": 5.0} for haiku
    cost = (
        observation.tokens_raw * rates["input"]
        + observation.tokens_compressed * rates["output"]
    ) / 1_000_000
    
    return DeterministicScore(
        structural_validity=structural,
        compression_ratio=ratio,
        title_quality=title_score,
        cost_usd=cost,
        latency_ms=latency_ms,
    )

def _score_title(title: str) -> float:
    """Heuristic title quality scoring."""
    words = title.split()
    score = 1.0
    
    # Length: 3-15 words ideal
    if len(words) < 3:
        score -= 0.3
    elif len(words) > 15:
        score -= 0.3
    
    # No trailing period
    if title.endswith('.'):
        score -= 0.2
    
    # Imperative mood heuristic: starts with verb-like word
    # (not perfect, but catches "Added", "Fixed", "Implemented")
    weak_starts = {"the", "a", "an", "this", "that", "it"}
    if words and words[0].lower() in weak_starts:
        score -= 0.2
    
    return max(0.0, score)
```

### 3.3 QAG-based information preservation (LLM-judge)

The QA-Generation approach is the gold standard for measuring whether a summary preserves key information from the source. Implementation:

**Step 1**: Generate 5 factual questions from the raw tool output
**Step 2**: Attempt to answer each question using only the compressed observation
**Step 3**: Score = fraction of questions answerable correctly

```python
QAG_PROMPT = """\
You are evaluating whether a compressed observation preserves key information from the original tool output.

<raw_tool_output>
{raw_output}
</raw_tool_output>

<compressed_observation>
Title: {title}
Summary: {summary}
Detail: {detail}
Files: {files_touched}
Functions: {functions_changed}
</compressed_observation>

Tasks:
1. Generate exactly 5 factual questions that a developer continuing this work would need answered. Focus on: what changed, why it changed, key file/function names, and decisions made.
2. For each question, determine if the compressed observation contains enough information to answer it correctly.
3. Also assess: does the summary explain the RATIONALE (why something was done), not just the action (what was done)?

Return ONLY a JSON object (no markdown fences):
{{
  "questions": [
    {{"question": "...", "answerable": true/false, "evidence": "brief quote or null"}},
    {{"question": "...", "answerable": true/false, "evidence": "brief quote or null"}},
    {{"question": "...", "answerable": true/false, "evidence": "brief quote or null"}},
    {{"question": "...", "answerable": true/false, "evidence": "brief quote or null"}},
    {{"question": "...", "answerable": true/false, "evidence": "brief quote or null"}}
  ],
  "decision_rationale_present": true/false,
  "rationale_note": "brief explanation of why rationale is/isn't present"
}}
"""
```

**Why 5 questions**: Balance between coverage and cost. Each QAG call uses ~2K input tokens (raw truncated to ~4K chars + compressed ~500 tokens + prompt ~500 tokens) and ~300 output tokens. At Sonnet 4.5 rates: ~$0.003 per observation.

**Why truncate raw to 4K chars for QAG**: The judge doesn't need the full 500KB raw output. It needs enough context to generate meaningful questions. 4K chars (~1K tokens) captures the most information-dense parts (we use the same head+tail truncation as the compressor).

```python
async def score_info_preservation(
    raw_output: str,
    observation: Observation,
    client: AsyncAnthropic,
    model: str = "claude-sonnet-4-5-20250929",
) -> tuple[float, float]:
    """
    Returns (info_preservation_score, decision_rationale_score).
    
    info_preservation: 0.0â€“1.0, fraction of questions answerable
    decision_rationale: 0.0 or 1.0, binary
    """
    # Truncate raw for QAG (4K chars, head+tail)
    truncated = _truncate_for_eval(raw_output, max_chars=4000)
    
    prompt = QAG_PROMPT.format(
        raw_output=truncated,
        title=observation.title,
        summary=observation.summary,
        detail=observation.detail or "(none)",
        files_touched=json.dumps(json.loads(observation.files_touched)),
        functions_changed=json.dumps(json.loads(observation.functions_changed)),
    )
    
    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    
    text = response.content[0].text
    data = _parse_json_response(text)
    
    answerable = sum(1 for q in data["questions"] if q["answerable"])
    info_score = answerable / len(data["questions"])
    rationale_score = 1.0 if data.get("decision_rationale_present") else 0.0
    
    return info_score, rationale_score
```

---

## 4. A/B Benchmark Runner

### 4.1 How it works

The benchmark runner takes stored raw outputs from `pending_queue` (where `raw_output` is preserved after processing), compresses each through two models, scores both, and produces a comparison report.

```python
class BenchmarkRunner:
    """Offline A/B comparison between compression models."""
    
    def __init__(
        self,
        db: aiosqlite.Connection,
        client: AsyncAnthropic,
        logger: MemLogger,
        config: Config,
    ):
        self.db = db
        self.client = client
        self.logger = logger
        self.config = config
    
    async def run(
        self,
        model_a: str = "claude-haiku-4-5-20251001",
        model_b: str = "claude-sonnet-4-5-20250929",
        sample_size: int = 30,
        judge_model: str = "claude-sonnet-4-5-20250929",
    ) -> BenchmarkReport:
        """
        Run head-to-head compression comparison.
        
        1. Sample `sample_size` raw outputs from pending_queue (status='done')
        2. Compress each through model_a and model_b
        3. Score each compression (deterministic + QAG)
        4. Produce paired comparison report
        """
        samples = await self._sample_raw_outputs(sample_size)
        
        results: list[PairedResult] = []
        for sample in samples:
            # Compress through both models
            comp_a = await self._compress_with_model(sample, model_a)
            comp_b = await self._compress_with_model(sample, model_b)
            
            # Deterministic scores (free)
            det_a = score_deterministic(comp_a, sample.raw_output, comp_a.latency_ms, model_a)
            det_b = score_deterministic(comp_b, sample.raw_output, comp_b.latency_ms, model_b)
            
            # QAG scores (costs API calls)
            info_a, rationale_a = await score_info_preservation(
                sample.raw_output, comp_a, self.client, judge_model
            )
            info_b, rationale_b = await score_info_preservation(
                sample.raw_output, comp_b, self.client, judge_model
            )
            
            results.append(PairedResult(
                sample_id=sample.id,
                tool_name=sample.tool_name,
                raw_size=len(sample.raw_output),
                score_a=CompressionScore(
                    model=model_a,
                    deterministic=det_a,
                    info_preservation=info_a,
                    decision_rationale=rationale_a,
                ),
                score_b=CompressionScore(
                    model=model_b,
                    deterministic=det_b,
                    info_preservation=info_b,
                    decision_rationale=rationale_b,
                ),
            ))
            
            # Log each pair
            self.logger.log("eval.ab_pair", {
                "sample_id": sample.id,
                "model_a": model_a,
                "model_b": model_b,
                "quality_a": self._composite_quality(results[-1].score_a),
                "quality_b": self._composite_quality(results[-1].score_b),
                "cost_a": det_a.cost_usd,
                "cost_b": det_b.cost_usd,
            })
        
        return self._build_report(results, model_a, model_b)
```

### 4.2 Sampling strategy

```python
async def _sample_raw_outputs(self, n: int) -> list[QueueSample]:
    """Sample diverse raw outputs for benchmarking."""
    rows = await self.db.execute_fetchall(
        """
        SELECT id, session_id, tool_name, raw_output, files_touched
        FROM pending_queue
        WHERE status = 'done' AND raw_output IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n,),
    )
    return [QueueSample(**row) for row in rows]
```

**Important**: This depends on `pending_queue` retaining `raw_output` after processing. Phase 3's processor marks items as `status='done'` but does not clear `raw_output`. This is by design â€” the architecture doc (Â§1.3) specifies "Logging raw inputs alongside compressed outputs for offline evaluation (Phase 7)."

**If `raw_output` has been cleared** (e.g., by a cleanup job), the benchmark falls back to observations-only mode (deterministic scoring only, no QAG since we need the raw input to generate questions).

### 4.3 Report format

```python
@dataclass
class BenchmarkReport:
    model_a: str
    model_b: str
    sample_size: int
    timestamp: str
    
    # Aggregate scores
    avg_quality_a: float
    avg_quality_b: float
    avg_cost_a: float
    avg_cost_b: float
    avg_latency_a: float
    avg_latency_b: float
    
    # Quality per dollar (the key metric)
    quality_per_dollar_a: float
    quality_per_dollar_b: float
    
    # Win/loss/tie counts
    a_wins: int
    b_wins: int
    ties: int  # within 0.05 quality difference
    
    # Per-dimension breakdown
    info_preservation_a: float
    info_preservation_b: float
    rationale_rate_a: float
    rationale_rate_b: float
    structural_validity_a: float
    structural_validity_b: float
    
    # Individual results for inspection
    pairs: list[PairedResult]
    
    def summary(self) -> str:
        """Human-readable summary for CLI output."""
        winner = self.model_a if self.avg_quality_a > self.avg_quality_b else self.model_b
        cost_winner = (
            self.model_a 
            if self.quality_per_dollar_a > self.quality_per_dollar_b 
            else self.model_b
        )
        return (
            f"Quality: {self.model_a}={self.avg_quality_a:.3f} vs "
            f"{self.model_b}={self.avg_quality_b:.3f} "
            f"(winner: {_short_name(winner)})\n"
            f"Cost-adjusted: {_short_name(cost_winner)} wins "
            f"({self.quality_per_dollar_a:.1f} vs {self.quality_per_dollar_b:.1f} quality/dollar)\n"
            f"Wins: {_short_name(self.model_a)}={self.a_wins}, "
            f"{_short_name(self.model_b)}={self.b_wins}, ties={self.ties}\n"
            f"Avg latency: {self.avg_latency_a:.0f}ms vs {self.avg_latency_b:.0f}ms\n"
            f"Avg cost/obs: ${self.avg_cost_a:.4f} vs ${self.avg_cost_b:.4f}"
        )
```

---

## 5. SQL Analysis Queries

### 5.1 Compression monitoring

```sql
-- Compression efficiency over time
SELECT date(timestamp) as day,
       COUNT(*) as observations,
       AVG(json_extract(data, '$.ratio')) as avg_ratio,
       AVG(duration_ms) as avg_compress_ms,
       SUM(tokens_in) as total_input_tokens,
       SUM(tokens_out) as total_output_tokens,
       ROUND(SUM(tokens_in) * 1.0 / 1000000 + SUM(tokens_out) * 5.0 / 1000000, 4) as cost_usd
FROM event_log 
WHERE event_type = 'compress.done'
GROUP BY day ORDER BY day;

-- Compression failures and error rates
SELECT date(timestamp) as day,
       COUNT(*) as errors,
       json_extract(data, '$.error_type') as error_type
FROM event_log
WHERE event_type = 'compress.error'
GROUP BY day, error_type
ORDER BY day DESC;

-- Cost breakdown by model (if A/B has been run)
SELECT json_extract(data, '$.model') as model,
       COUNT(*) as count,
       AVG(json_extract(data, '$.ratio')) as avg_ratio,
       AVG(duration_ms) as avg_ms,
       ROUND(SUM(tokens_in) * 1.0 / 1000000 + SUM(tokens_out) * 5.0 / 1000000, 4) as total_cost
FROM event_log
WHERE event_type = 'compress.done'
GROUP BY model;
```

### 5.2 Search quality

```sql
-- Search usage and quality
SELECT date(timestamp) as day,
       COUNT(*) as queries,
       AVG(json_extract(data, '$.result_count')) as avg_results,
       AVG(json_extract(data, '$.top_score')) as avg_top_score,
       AVG(duration_ms) as avg_ms
FROM event_log
WHERE event_type LIKE 'search.%'
GROUP BY day ORDER BY day;

-- Zero-result searches (potential quality issues)
SELECT timestamp, json_extract(data, '$.query') as query
FROM event_log
WHERE event_type LIKE 'search.%' 
  AND json_extract(data, '$.result_count') = 0
ORDER BY timestamp DESC LIMIT 20;
```

### 5.3 AST resolution accuracy

```sql
-- Resolution rate over time
SELECT date(timestamp) as day,
       AVG(
           1.0 - CAST(json_extract(data, '$.unresolved_calls') AS REAL) /
           MAX(json_extract(data, '$.call_edges'), 1)
       ) as avg_resolution_rate,
       SUM(json_extract(data, '$.call_edges')) as total_edges
FROM event_log
WHERE event_type = 'ast.scan'
GROUP BY day ORDER BY day;

-- Files with lowest resolution rates
SELECT json_extract(data, '$.file') as file,
       COUNT(*) as scans,
       AVG(
           1.0 - CAST(json_extract(data, '$.unresolved_calls') AS REAL) /
           MAX(json_extract(data, '$.call_edges'), 1)
       ) as avg_resolution
FROM event_log
WHERE event_type = 'ast.scan'
GROUP BY file
HAVING avg_resolution < 0.8
ORDER BY avg_resolution ASC;
```

### 5.4 Context injection efficiency

```sql
-- Token budget usage
SELECT date(timestamp) as day,
       AVG(json_extract(data, '$.total_tokens')) as avg_injected,
       MAX(json_extract(data, '$.total_tokens')) as max_injected,
       AVG(json_extract(data, '$.budget')) as avg_budget,
       AVG(duration_ms) as avg_ms
FROM event_log
WHERE event_type = 'hook.context_inject'
GROUP BY day ORDER BY day;

-- Layer inclusion frequency
SELECT json_extract(data, '$.layers_included') as layers,
       COUNT(*) as count
FROM event_log
WHERE event_type = 'hook.context_inject'
GROUP BY layers
ORDER BY count DESC;
```

### 5.5 Learnings health

```sql
-- Learning confidence distribution
SELECT category,
       COUNT(*) as count,
       AVG(confidence) as avg_confidence,
       SUM(CASE WHEN confidence >= 0.8 THEN 1 ELSE 0 END) as high_confidence,
       SUM(CASE WHEN confidence < 0.3 THEN 1 ELSE 0 END) as low_confidence
FROM learnings
WHERE is_active = 1
GROUP BY category;

-- Learning extraction success rate
SELECT date(timestamp) as day,
       SUM(CASE WHEN event_type = 'learning.extracted' THEN 1 ELSE 0 END) as extracted,
       SUM(CASE WHEN event_type = 'learning.merged' THEN 1 ELSE 0 END) as merged,
       SUM(CASE WHEN event_type = 'learning.contradicted' THEN 1 ELSE 0 END) as contradicted,
       SUM(CASE WHEN event_type = 'learning.error' THEN 1 ELSE 0 END) as errors
FROM event_log
WHERE event_type LIKE 'learning.%'
GROUP BY day ORDER BY day;
```

### 5.6 Overall system health

```sql
-- Daily summary dashboard
SELECT date(el.timestamp) as day,
       (SELECT COUNT(*) FROM sessions WHERE date(started_at) = date(el.timestamp)) as sessions,
       COUNT(DISTINCT CASE WHEN el.event_type = 'compress.done' THEN el.id END) as observations,
       COUNT(DISTINCT CASE WHEN el.event_type = 'compress.error' THEN el.id END) as compress_errors,
       COUNT(DISTINCT CASE WHEN el.event_type LIKE 'search.%' THEN el.id END) as searches,
       COUNT(DISTINCT CASE WHEN el.event_type = 'hook.context_inject' THEN el.id END) as injections,
       ROUND(SUM(CASE WHEN el.event_type = 'compress.done' 
           THEN el.tokens_in * 1.0 / 1000000 + el.tokens_out * 5.0 / 1000000 
           ELSE 0 END), 4) as compress_cost_usd
FROM event_log el
GROUP BY day ORDER BY day DESC LIMIT 14;

-- Cumulative cost tracking
SELECT SUM(tokens_in) * 1.0 / 1000000 + SUM(tokens_out) * 5.0 / 1000000 as total_compress_cost,
       COUNT(*) as total_compressions,
       MIN(timestamp) as first_compression,
       MAX(timestamp) as last_compression
FROM event_log
WHERE event_type = 'compress.done';
```

---

## 6. Data Types

### 6.1 New Pydantic models (`eval/models.py`)

```python
from dataclasses import dataclass
from pydantic import BaseModel

# Model pricing rates ($ per million tokens)
MODEL_RATES: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
}

class DeterministicScore(BaseModel):
    structural_validity: float
    compression_ratio: float
    title_quality: float
    cost_usd: float
    latency_ms: int

class QAGResult(BaseModel):
    questions: list[dict]  # [{question, answerable, evidence}]
    info_preservation: float  # 0.0â€“1.0
    decision_rationale: float  # 0.0 or 1.0
    rationale_note: str

class CompressionScore(BaseModel):
    model: str
    deterministic: DeterministicScore
    info_preservation: float
    decision_rationale: float

class PairedResult(BaseModel):
    sample_id: str
    tool_name: str
    raw_size: int
    score_a: CompressionScore
    score_b: CompressionScore

class BenchmarkReport(BaseModel):
    model_a: str
    model_b: str
    sample_size: int
    timestamp: str
    avg_quality_a: float
    avg_quality_b: float
    avg_cost_a: float
    avg_cost_b: float
    avg_latency_a: float
    avg_latency_b: float
    quality_per_dollar_a: float
    quality_per_dollar_b: float
    a_wins: int
    b_wins: int
    ties: int
    info_preservation_a: float
    info_preservation_b: float
    rationale_rate_a: float
    rationale_rate_b: float
    structural_validity_a: float
    structural_validity_b: float
    pairs: list[PairedResult]

class ObservationEvalResult(BaseModel):
    """Result of evaluating a single existing observation."""
    observation_id: str
    deterministic: DeterministicScore
    qag: QAGResult | None  # None if raw_output unavailable
    composite_quality: float

class QueueSample(BaseModel):
    id: str
    session_id: str
    tool_name: str
    raw_output: str
    files_touched: str
```

---

## 7. CLI Interface

### 7.1 `claude-mem eval compression`

Evaluate existing observations (deterministic scoring, optionally with QAG).

```
Usage: claude-mem eval compression [OPTIONS]

Options:
  --limit INT       Number of recent observations to evaluate (default: 20)
  --with-qag        Run QAG scoring (requires API key, costs ~$0.003/obs)
  --json            Output as JSON instead of table
  --since DATE      Only evaluate observations after this date

Example output:

â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Compression Eval: 20 observations                         â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Title                â”‚ Validity â”‚ Ratio  â”‚ Info    â”‚ Cost  â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Added JWT auth midâ€¦  â”‚ âœ“        â”‚ 42:1   â”‚ 0.80    â”‚$0.008 â”‚
â”‚ Fixed broken imporâ€¦  â”‚ âœ“        â”‚ 28:1   â”‚ 1.00    â”‚$0.006 â”‚
â”‚ Read config documeâ€¦  â”‚ âœ“        â”‚ 15:1   â”‚ 0.60    â”‚$0.004 â”‚
â”‚ ...                  â”‚          â”‚        â”‚         â”‚       â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Avg quality: 0.82  Avg ratio: 34:1  Total cost: $0.12     â”‚
â”‚ Structural validity: 100%  Rationale present: 75%          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 7.2 `claude-mem eval benchmark`

Run offline A/B comparison between two models.

```
Usage: claude-mem eval benchmark [OPTIONS]

Options:
  --model-a STR     First model (default: claude-haiku-4-5-20251001)
  --model-b STR     Second model (default: claude-sonnet-4-5-20250929)
  --samples INT     Number of samples to compare (default: 30)
  --judge STR       Judge model for QAG (default: claude-sonnet-4-5-20250929)
  --json            Output as JSON

Example output:

â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  A/B Benchmark: Haiku 4.5 vs Sonnet 4.5 (30 samples)      â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Metric              â”‚ Haiku 4.5    â”‚ Sonnet 4.5   â”‚ Winner â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Avg Quality         â”‚ 0.81         â”‚ 0.87         â”‚ Sonnet â”‚
â”‚ Info Preservation   â”‚ 0.78         â”‚ 0.86         â”‚ Sonnet â”‚
â”‚ Rationale Present   â”‚ 72%          â”‚ 88%          â”‚ Sonnet â”‚
â”‚ Structural Validity â”‚ 100%         â”‚ 100%         â”‚ Tie    â”‚
â”‚ Avg Latency         â”‚ 620ms        â”‚ 1,340ms      â”‚ Haiku  â”‚
â”‚ Avg Cost/obs        â”‚ $0.008       â”‚ $0.024       â”‚ Haiku  â”‚
â”‚ Quality/Dollar      â”‚ 101          â”‚ 36           â”‚ Haiku  â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Head-to-head wins   â”‚ 8            â”‚ 19           â”‚ Sonnet â”‚
â”‚ Ties                â”‚ 3            â”‚              â”‚        â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                             â”‚
â”‚ RECOMMENDATION: Haiku 4.5 for observations (3Ã— cheaper,    â”‚
â”‚ quality within 7%). Consider Sonnet for session summaries   â”‚
â”‚ where rationale preservation matters more.                  â”‚
â”‚                                                             â”‚
â”‚ Estimated cost impact:                                      â”‚
â”‚   Current (all Haiku): ~$0.50/session                       â”‚
â”‚   All Sonnet:          ~$1.50/session                       â”‚
â”‚   Hybrid (Haiku obs + Sonnet summary): ~$0.55/session       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 7.3 `claude-mem eval health`

Run SQL analysis queries and display system health dashboard.

```
Usage: claude-mem eval health [OPTIONS]

Options:
  --days INT        Period to analyze (default: 7)
  --json            Output as JSON

Example output:

â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  System Health: Last 7 days                                 â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Sessions: 14  Observations: 312  Errors: 2 (0.6%)          â”‚
â”‚ Compression: avg 38:1 ratio, 87ms latency                  â”‚
â”‚ Context injection: avg 1,847 tokens, 42ms build time        â”‚
â”‚ Search: 23 queries, avg 4.2 results, 12ms latency          â”‚
â”‚ AST resolution: 82% avg (worst: utils/config.py at 64%)    â”‚
â”‚ Active learnings: 12 (3 high-confidence, 0 contradicted)   â”‚
â”‚ Total API cost: $6.24                                       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## 8. Module Structure

### 8.1 New files

```
src/claude_mem_lite/
    eval/
        __init__.py
        evaluator.py      # DeterministicScore + QAG scoring
        benchmark.py       # BenchmarkRunner (offline A/B)
        models.py          # Pydantic models + MODEL_RATES
        prompts.py         # QAG_PROMPT
        queries.sql        # Reference SQL queries (embedded as string constants)
    cli/
        eval_cmd.py        # CLI commands: eval compression, eval benchmark, eval health
```

### 8.2 Modified files

None. Phase 7 is purely additive. No production code changes.

**Critical design point**: The eval framework reads from existing tables (`event_log`, `observations`, `pending_queue`, `learnings`, `sessions`) and writes only to `event_log` (via `MemLogger`). It does not modify any production data.

### 8.3 Event types added

| Event Type | When | Data |
|---|---|---|
| `eval.deterministic` | After scoring an observation | `{observation_id, structural_validity, ratio, title_quality, cost_usd}` |
| `eval.qag` | After QAG scoring | `{observation_id, info_preservation, decision_rationale, questions_count}` |
| `eval.ab_pair` | After comparing one sample | `{sample_id, model_a, model_b, quality_a, quality_b, cost_a, cost_b}` |
| `eval.benchmark_complete` | After full benchmark run | `{model_a, model_b, samples, avg_quality_a, avg_quality_b, winner}` |

---

## 9. Integration Points

### 9.1 Phase 0 (Storage)

**No changes.** Reads from `event_log`, `observations`, `pending_queue`, `learnings`, `sessions`. All tables already exist.

### 9.2 Phase 3 (Worker/Compressor)

**No changes.** The compressor already logs `compress.done` and `compress.error` events with token counts, latency, and model name. Phase 7 reads these events.

**Dependency on `raw_output` retention**: The benchmark runner samples from `pending_queue` where `status='done'` and `raw_output IS NOT NULL`. Phase 3 does not clear `raw_output` after processing â€” this is by design. If a future cleanup job is added (Phase 9), it must either: (a) preserve raw outputs for the most recent N observations, or (b) copy them to a separate `eval_corpus` table before deletion.

### 9.3 Phase 5 (Context Injection)

**No changes.** Phase 5 already logs `hook.context_inject` events. Phase 7 queries these.

### 9.4 Phase 6 (Learnings)

**No changes.** Phase 6 already logs `learning.*` events. Phase 7 queries these.

### 9.5 Phase 8 (CLI Reports)

Phase 8 builds on Phase 7's SQL queries. The `eval health` command is a precursor to the `report --eval` dashboard described in the architecture doc.

### 9.6 Phase 9 (Hardening)

Phase 9 can add:
- Automated eval runs on every N observations
- Regression detection (quality drops below threshold â†’ alert)
- Raw output lifecycle management (cleanup with eval corpus preservation)

---

## 10. Cost Analysis

### 10.1 Eval costs

| Operation | Model | Input tokens | Output tokens | Cost per call | Typical usage |
|-----------|-------|-------------|---------------|---------------|---------------|
| QAG scoring | Sonnet 4.5 | ~2,000 | ~300 | ~$0.003 | 20 obs Ã— $0.003 = $0.06 |
| Benchmark (model A compression) | Haiku 4.5 | ~8,000 | ~500 | ~$0.011 | 30 samples = $0.33 |
| Benchmark (model B compression) | Sonnet 4.5 | ~8,000 | ~500 | ~$0.032 | 30 samples = $0.96 |
| Benchmark (QAG for both) | Sonnet 4.5 | ~2,000 | ~300 | ~$0.003 | 60 calls = $0.18 |

**Total cost per benchmark run**: ~$1.47 (30 samples, Haiku vs Sonnet, with QAG)
**Total cost per compression eval**: ~$0.06 (20 observations with QAG) or $0.00 (deterministic only)

This is a one-time (or infrequent) cost. You run the benchmark once to decide the model strategy, then periodically run `eval compression` to spot-check quality.

### 10.2 Expected benchmark results (hypothesis)

Based on research â€” Haiku 4.5 excels at structured extraction and JSON formatting tasks, matching Sonnet quality at 1/3 the cost:

- **Structural validity**: Both ~100% (well-constrained prompt)
- **Info preservation**: Haiku ~0.75â€“0.85, Sonnet ~0.80â€“0.90 (Sonnet slightly better on decision rationale)
- **Compression ratio**: Similar (both follow the same prompt)
- **Latency**: Haiku 2â€“3Ã— faster
- **Quality/dollar**: Haiku wins by 2Ã— or more

**Likely outcome**: Keep Haiku 4.5 for observations. Consider Sonnet for session summaries (Phase 3 summarizer) where rationale preservation is more important. This matches the architecture doc's prediction.

---

## 11. Test Plan

### 11.1 Test categories

| Category | Tests | What it validates |
|----------|-------|-------------------|
| **Deterministic scoring** | 5 | Structural validity catches missing fields, compression ratio calculated correctly, title quality heuristics work, cost calculation matches MODEL_RATES, handles edge cases (zero tokens) |
| **QAG prompt** | 3 | Valid JSON response parsed, malformed response handled gracefully, question count enforced |
| **QAG scoring** | 3 | Info preservation fraction correct, decision rationale binary, API error non-fatal |
| **Benchmark runner** | 4 | Sample selection works, paired results produced, report aggregation correct, empty pending_queue handled |
| **CLI** | 4 | `eval compression` outputs table, `eval benchmark` outputs comparison, `eval health` outputs dashboard, `--json` flag produces valid JSON |
| **SQL queries** | 3 | Compression query runs without error, search query runs, health dashboard query runs |
| **Total** | **22** | |

### 11.2 Test infrastructure

```python
# conftest.py additions for Phase 7

@pytest.fixture
def mock_sonnet_qag():
    """Mock Sonnet response for QAG scoring."""
    return json.dumps({
        "questions": [
            {"question": "What file was modified?", "answerable": True, "evidence": "auth/service.py"},
            {"question": "What function was added?", "answerable": True, "evidence": "refresh_token"},
            {"question": "Why was JWT chosen?", "answerable": True, "evidence": "stateless auth"},
            {"question": "What test was written?", "answerable": False, "evidence": None},
            {"question": "What config changed?", "answerable": False, "evidence": None},
        ],
        "decision_rationale_present": True,
        "rationale_note": "Summary explains JWT was chosen for stateless auth."
    })

@pytest.fixture
def seeded_pending_queue(async_db):
    """DB with pending_queue items that have raw_output preserved."""
    # Insert 10 done items with raw_output for benchmark testing
    ...

@pytest.fixture
def seeded_event_log(async_db):
    """DB with event_log entries for SQL query testing."""
    # Insert compress.done, search.*, hook.context_inject events
    ...

@pytest.fixture
def evaluator(mock_anthropic, tmp_config):
    """Evaluator with mocked API client."""
    return Evaluator(
        client=mock_anthropic,
        logger=MemLogger(tmp_config.log_dir),
        config=tmp_config,
    )
```

### 11.3 Key test scenarios

**Deterministic scoring**:
- Observation with valid title, summary, detail â†’ structural_validity=1.0
- Observation with empty title â†’ structural_validity=0.0
- Title "Added JWT auth middleware" â†’ title_quality=1.0
- Title "The JWT thing." â†’ title_quality < 0.6 (weak start + period)
- Zero tokens_compressed â†’ no division-by-zero error, ratio=0

**QAG scoring** (mock API):
- Valid QAG response â†’ info_preservation=3/5=0.6, decision_rationale=1.0
- Empty questions array â†’ info_preservation=0.0
- API error â†’ returns None, logged as `eval.qag_error`, non-fatal

**Benchmark runner** (mock API):
- 5 samples, both models succeed â†’ 5 PairedResults, report generated
- 0 samples (empty pending_queue) â†’ BenchmarkReport with sample_size=0, graceful message
- Model A fails on 1 sample â†’ that sample skipped, N-1 results reported
- Report quality/dollar calculation correct

**CLI** (mock evaluator):
- `eval compression --limit 5` â†’ table with 5 rows
- `eval benchmark --samples 3 --json` â†’ valid JSON BenchmarkReport
- `eval health --days 1` â†’ dashboard with correct counts

**SQL queries**:
- Each query in `queries.sql` runs without syntax error against seeded DB
- Compression query returns correct aggregates
- Health dashboard query joins correctly

---

## 12. Performance Targets

| Operation | Target | Rationale |
|-----------|--------|-----------|
| Deterministic scoring (1 obs) | <1ms | Pure Python arithmetic |
| QAG scoring (1 obs) | <3000ms | Sonnet 4.5 API call |
| Full eval (20 obs, deterministic) | <100ms | No API calls |
| Full eval (20 obs, with QAG) | <60s | 20 API calls, sequential |
| Benchmark (30 samples) | <10min | 60 compression calls + 60 QAG calls |
| SQL health queries | <50ms each | Indexed event_log |
| CLI render (rich table) | <100ms | Terminal output |

---

## 13. Acceptance Criteria

Phase 7 is complete when:

- [ ] `eval/evaluator.py` scores observations on all deterministic dimensions
- [ ] `eval/evaluator.py` scores observations via QAG (Sonnet 4.5 judge)
- [ ] QAG prompt generates 5 questions, parses JSON response, handles errors
- [ ] `eval/benchmark.py` runs offline A/B comparison between two models
- [ ] Benchmark produces `BenchmarkReport` with quality, cost, and quality/dollar metrics
- [ ] `eval/queries.sql` contains all 6 query categories (compression, search, AST, context, learnings, health)
- [ ] `claude-mem eval compression` displays table of scored observations
- [ ] `claude-mem eval benchmark` displays head-to-head comparison
- [ ] `claude-mem eval health` displays system health dashboard
- [ ] `--json` flag works on all eval subcommands
- [ ] All eval operations log to `event_log` via MemLogger
- [ ] No production code modified (eval is read-only + event logging)
- [ ] All 22 tests pass (pytest + pytest-asyncio, <15s with mocked API)
- [ ] `ruff check` and `ruff format --check` pass with zero warnings

---

## 14. Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **QAG judge quality** | Medium | Medium | Sonnet 4.5 is strong at structured evaluation. Confined yes/no answers reduce hallucination. Spot-check 10 results manually after first run. |
| **raw_output deleted** | Low | High | Phase 3 preserves raw_output by design. If cleanup is added later, must preserve eval corpus. Benchmark gracefully handles missing raw data. |
| **Low sample size** | Medium | Low | 30 samples is minimum for meaningful comparison. With paired design (same inputs), statistical power is better than independent samples. |
| **Title quality heuristic too crude** | Low | Low | Simple scoring â€” catches obvious failures (empty, too long, weak start). Not meant to replace human judgment. |
| **Cost estimates inaccurate** | Low | Low | MODEL_RATES hardcoded. If Anthropic changes pricing, update the dict. Costs are informational, not billing. |
| **Benchmark takes too long** | Low | Low | 30 samples Ã— ~4 API calls each = ~120 calls. At ~1.5s each = ~3 minutes. Acceptable for a one-time operation. |

---

## 15. Open Questions

| Question | Current assumption | When to resolve |
|----------|-------------------|-----------------|
| **Should we add a tiered model strategy based on results?** | Single model (Haiku) for everything. Benchmark provides data but doesn't auto-switch. | After first benchmark run â€” if Sonnet wins decisively on summaries, add `Config.summarization_model`. |
| **Should QAG use extended thinking?** | No â€” plain prompting. CoT rationale is already requested in the prompt ("evidence" field). | Revisit if QAG scores seem unreliable after manual spot-check. |
| **Should eval run automatically on every Nth observation?** | No â€” manual CLI invocation only. | Phase 9 hardening. |
| **Should we cache benchmark results?** | No â€” results are logged to event_log. Re-running produces fresh data. | Only if benchmark runtime becomes problematic. |
| **Is quality/dollar the right meta-metric?** | Yes â€” for cost-conscious local tools, raw quality alone doesn't inform decisions. | Revisit if usage pattern changes (e.g., team usage where cost matters less). |
