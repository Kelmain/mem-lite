# PRD Amendments: Phase 7 (Post-Review)

**Date**: 2026-02-08
**Triggered by**: External review of Phase 7 PRD (3 comments, all actionable)
**Affects**: Phase 7 v1 (Eval Framework + Compression Benchmarking)

---

## Amendment 1: Add `pending_queue` Size Warning to Sampling Strategy

**Severity**: Low
**Affects**: Phase 7, Section 4.2 (`_sample_raw_outputs`)

### Problem

The `ORDER BY RANDOM() LIMIT ?` query on `pending_queue` performs a full table scan. At ~50 observations/day with no cleanup, the table grows ~18k rows/year. SQLite handles this comfortably â€” the concern is not current performance but documenting the degradation path so future developers don't get surprised.

Phase 9's `prune` command already addresses the lifecycle problem (`SET raw_output = NULL` on old items, `--keep-raw N` preserves the most recent for eval). The reviewer's suggestion to archive into a separate `eval_corpus` table adds schema complexity for no practical benefit â€” you'll never benchmark against 6-month-old compression data.

### Specification Change

Add a performance note after the sampling query in Section 4.2:

```markdown
**Performance note**: `ORDER BY RANDOM()` performs a full table scan. This is
negligible for typical table sizes (<50k rows). If `pending_queue` has grown
very large, run `claude-mem prune --keep-raw 100` (Phase 9) first to clear
old `raw_output` blobs. The query filters on `raw_output IS NOT NULL`, so
pruned rows are excluded automatically.
```

No code changes. No schema changes.

### Why not the `eval_corpus` table

The reviewer suggested archiving raw outputs to a compressed `eval_corpus` table before deletion. This adds:
- A new table to the schema (migration, CRUD ops, tests)
- A copy step in the prune workflow
- A second data source for the benchmark runner to query

For a tool where you'll run benchmarks a handful of times total, this is overengineered. Phase 9's `--keep-raw 100` is sufficient â€” it preserves 100 recent raw outputs at all times, which is more than the default `--samples 30` benchmark needs.

---

## Amendment 2: Move `MODEL_RATES` to `config.py`

**Severity**: Low
**Affects**: Phase 7, Section 6.1 (`eval/models.py`)

### Problem

`MODEL_RATES` is hardcoded in `eval/models.py`:

```python
MODEL_RATES: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
}
```

Two risks:
1. **Key mismatch**: If `Config.compression_model` uses a different string (e.g., the shorthand `claude-haiku-4-5`), the lookup raises `KeyError` at runtime.
2. **Stale rates**: Anthropic price changes or model snapshot updates require editing an eval-internal file that's easy to miss.

The reviewer also suggested "fetch dynamically." Anthropic does not expose a public pricing API, so dynamic fetching would mean scraping a web page â€” fragile and inappropriate for a local tool.

### Specification Change

Move `MODEL_RATES` to `config.py` as a class attribute on `Config`, using the same pinned model strings that the rest of the system references:

```python
# config.py
class Config:
    # ... existing fields ...

    # Pricing: $ per million tokens. Update when Anthropic changes rates.
    MODEL_RATES: ClassVar[dict[str, dict[str, float]]] = {
        "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
        "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    }
```

In `eval/models.py`, remove the local `MODEL_RATES` dict. In `eval/evaluator.py`, access rates via `self.config.MODEL_RATES`.

Add a runtime guard in `score_deterministic`:

```python
def score_deterministic(
    observation: Observation,
    raw_output: str,
    latency_ms: int,
    model: str,
    config: Config,
) -> DeterministicScore:
    rates = config.MODEL_RATES.get(model)
    if rates is None:
        raise ValueError(
            f"Unknown model '{model}' â€” add it to Config.MODEL_RATES. "
            f"Known models: {list(config.MODEL_RATES.keys())}"
        )
    # ... rest unchanged
```

This gives a clear error message instead of a bare `KeyError`, and centralizes all model configuration in one place.

### Impact on other sections

- Section 3.2 code sample: update `score_deterministic` signature to accept `config`
- Section 4 (`BenchmarkRunner`): pass `self.config` to `score_deterministic` calls
- Section 11 tests: `DeterministicScore` tests need a `tmp_config` fixture (already exists in conftest)

---

## Amendment 3: Tighten QAG Prompt and Use Dynamic Question Count

**Severity**: Medium
**Affects**: Phase 7, Section 3.3 (QAG prompt), Section 6.1 (`QAGResult`), Section 3.1 (scoring formula)

### Problem

The current QAG prompt says:

> "Generate exactly 5 factual questions that a developer continuing this work would need answered. Focus on: what changed, why it changed, key file/function names, and decisions made."

This creates two failure modes:

**Failure mode 1 â€” Trivial questions on code diffs**: When the raw output is a code diff or tool output, the LLM gravitates toward surface-level questions: "What line number was changed?", "What was the variable renamed to?", "Which import was added?" A good summary intentionally abstracts away these details, so it scores 0 on questions it *should* score 0 on. The `info_preservation` score drops, falsely indicating poor compression.

**Failure mode 2 â€” Padding on small changes**: A 3-line bug fix doesn't have 5 meaningful architectural questions. The model pads with low-quality questions to hit the count, diluting the signal. If 2 questions are meaningful and 3 are padding, a summary that answers both real questions scores 0.4 instead of 1.0.

Both failure modes produce misleading scores that undermine the eval framework's core purpose.

### Specification Change

**Replace the QAG prompt** in Section 3.3:

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
1. Generate UP TO 5 factual questions that a developer continuing this work
   tomorrow would need answered.

   FOCUS ON:
   - Architectural decisions and their rationale
   - Behavioral changes to public APIs or interfaces
   - New dependencies or breaking changes
   - Key trade-offs or alternatives that were considered
   - Error handling or edge case changes

   DO NOT ask about:
   - Specific line numbers or column positions
   - Formatting, whitespace, or import ordering changes
   - Variable or parameter renaming (unless it signals a semantic change)
   - Internal helper functions or private implementation details
   - Trivial boilerplate (license headers, auto-generated code)

   If the change is small (e.g., a single bug fix), generate FEWER questions
   rather than padding with low-value ones. Minimum 2, maximum 5.

2. For each question, determine if the compressed observation contains enough
   information to answer it correctly.

3. Also assess: does the summary explain the RATIONALE (why something was done),
   not just the action (what was done)?

Return ONLY a JSON object (no markdown fences):
{{
  "questions": [
    {{"question": "...", "answerable": true/false, "evidence": "brief quote or null"}}
  ],
  "decision_rationale_present": true/false,
  "rationale_note": "brief explanation of why rationale is/isn't present"
}}
"""
```

**Key changes from original**:
- "exactly 5" â†’ "UP TO 5" with minimum 2
- Explicit inclusion list (architectural, API, dependencies, trade-offs, error handling)
- Explicit exclusion list (line numbers, formatting, renaming, internals, boilerplate)
- "developer continuing this work tomorrow" â€” anchors question quality to practical usefulness

**Update `score_info_preservation`** to use dynamic denominator:

```python
async def score_info_preservation(
    raw_output: str,
    observation: Observation,
    client: AsyncAnthropic,
    model: str = "claude-sonnet-4-5-20250929",
) -> tuple[float, float]:
    """
    Returns (info_preservation_score, decision_rationale_score).
    
    info_preservation: 0.0â€“1.0, fraction of generated questions answerable
    decision_rationale: 0.0 or 1.0, binary
    """
    # ... truncation and API call unchanged ...
    
    data = _parse_json_response(text)
    questions = data["questions"]
    
    # Validate question count (2-5)
    if len(questions) < 2:
        # Model didn't follow instructions â€” treat as eval failure
        return 0.0, 0.0
    if len(questions) > 5:
        questions = questions[:5]  # Truncate silently
    
    answerable = sum(1 for q in questions if q["answerable"])
    info_score = answerable / len(questions)  # Dynamic denominator
    rationale_score = 1.0 if data.get("decision_rationale_present") else 0.0
    
    return info_score, rationale_score
```

**Update `QAGResult` model** in Section 6.1:

```python
class QAGResult(BaseModel):
    questions: list[dict]  # [{question, answerable, evidence}]
    questions_generated: int  # 2-5, actual count (was always 5)
    info_preservation: float  # 0.0â€“1.0, answerable / questions_generated
    decision_rationale: float  # 0.0 or 1.0
    rationale_note: str
```

**Update the scoring table** in Section 3.1:

| Dimension | Change |
|-----------|--------|
| `info_preservation` | Description changes from "Generate 5 questions" to "Generate 2â€“5 questions from raw, answer from compressed, fraction correct" |

The composite quality formula (Section 3.1) is unchanged â€” it already uses `info_preservation` as a 0.0â€“1.0 float regardless of how many questions produced it.

### Impact on tests

Update Section 11.2 `mock_sonnet_qag` fixture to include variable question counts:

```python
@pytest.fixture(params=["full", "minimal"])
def mock_sonnet_qag(request):
    """Mock Sonnet response for QAG scoring â€” variable question count."""
    if request.param == "full":
        return json.dumps({
            "questions": [
                {"question": "What architectural pattern was adopted?", "answerable": True, "evidence": "middleware chain"},
                {"question": "What new dependency was added?", "answerable": True, "evidence": "pyjwt"},
                {"question": "Why was JWT chosen over session tokens?", "answerable": True, "evidence": "stateless auth"},
                {"question": "What breaking API change was introduced?", "answerable": False, "evidence": None},
                {"question": "What error handling was added?", "answerable": False, "evidence": None},
            ],
            "decision_rationale_present": True,
            "rationale_note": "Summary explains JWT was chosen for stateless auth."
        })
    else:  # minimal â€” small change, only 2 questions
        return json.dumps({
            "questions": [
                {"question": "What bug was fixed?", "answerable": True, "evidence": "off-by-one in pagination"},
                {"question": "What was the root cause?", "answerable": True, "evidence": "zero-indexed vs one-indexed"},
            ],
            "decision_rationale_present": False,
            "rationale_note": "Bug fix summary states what, not why the bug existed."
        })
```

Add test case in Section 11.3:

- Variable question count: 2 questions, both answerable â†’ `info_preservation=1.0` (not 0.4)
- Under-minimum: 1 question â†’ `info_preservation=0.0` (eval failure, logged)
- Over-maximum: 7 questions â†’ truncated to 5, scored normally

---

## Summary of Changes

| # | Section | Change | Severity |
|---|---------|--------|----------|
| 1 | Â§4.2 | Add performance note about `pending_queue` size and prune interaction | Low |
| 2 | Â§6.1, Â§3.2, Â§4 | Move `MODEL_RATES` to `Config`, add `KeyError` guard | Low |
| 3 | Â§3.3, Â§6.1, Â§3.1, Â§11 | Rewrite QAG prompt with inclusion/exclusion lists, dynamic 2â€“5 question count, update scoring and tests | Medium |

No new tables. No new dependencies. No changes to production code paths (eval remains read-only + event logging).
