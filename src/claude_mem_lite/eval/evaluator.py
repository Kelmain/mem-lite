"""Compression quality scoring: deterministic checks and QAG-based evaluation."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from claude_mem_lite.eval.models import DeterministicScore
from claude_mem_lite.eval.prompts import QAG_PROMPT

if TYPE_CHECKING:
    from claude_mem_lite.config import Config
    from claude_mem_lite.storage.models import Observation

logger = logging.getLogger(__name__)

# Maximum chars of raw_output to send to QAG judge
_MAX_QAG_CHARS = 4000


def score_deterministic(
    observation: Observation,
    raw_output: str,
    latency_ms: int,
    model: str,
    config: Config,
) -> DeterministicScore:
    """Score an observation without any API calls.

    Args:
        observation: The compressed observation to score.
        raw_output: Original raw tool output.
        latency_ms: Compression latency in milliseconds.
        model: Model identifier used for compression.
        config: Config instance with MODEL_RATES.

    Returns:
        DeterministicScore with structural_validity, compression_ratio,
        title_quality, cost_usd, and latency_ms.
    """
    rates = config.MODEL_RATES.get(model)
    if rates is None:
        msg = (
            f"Unknown model '{model}' -- add it to Config.MODEL_RATES. "
            f"Known models: {list(config.MODEL_RATES.keys())}"
        )
        raise ValueError(msg)

    # Structural validity
    structural = 1.0
    if not observation.title or not observation.summary:
        structural = 0.0

    # Compression ratio (guard against zero)
    if observation.tokens_compressed == 0:
        ratio = 0.0
    else:
        ratio = len(raw_output) / observation.tokens_compressed

    # Title quality (heuristic)
    title_score = _score_title(observation.title)

    # Cost calculation
    cost = (
        observation.tokens_raw * rates["input"] + observation.tokens_compressed * rates["output"]
    ) / 1_000_000

    return DeterministicScore(
        structural_validity=structural,
        compression_ratio=ratio,
        title_quality=title_score,
        cost_usd=cost,
        latency_ms=latency_ms,
    )


def _score_title(title: str) -> float:
    """Heuristic title quality scoring.

    Starts at 1.0 and applies penalties:
    - -0.3 if < 3 words or > 15 words
    - -0.2 if ends with period
    - -0.2 if starts with weak word (the, a, an, this, that, it)
    """
    words = title.split()
    score = 1.0

    if len(words) < 3 or len(words) > 15:
        score -= 0.3

    if title.endswith("."):
        score -= 0.2

    weak_starts = {"the", "a", "an", "this", "that", "it"}
    if words and words[0].lower() in weak_starts:
        score -= 0.2

    return max(0.0, score)


async def score_info_preservation(
    raw_output: str,
    observation: Observation,
    client: object,
    model: str = "claude-sonnet-4-5-20250929",
) -> tuple[float, float]:
    """Score information preservation via QAG (question-answer generation).

    Returns:
        Tuple of (info_preservation_score, decision_rationale_score).
        info_preservation: 0.0-1.0, fraction of questions answerable.
        decision_rationale: 0.0 or 1.0, binary.
        Returns (0.0, 0.0) on any error.
    """
    truncated = _truncate_for_eval(raw_output, max_chars=_MAX_QAG_CHARS)

    prompt = QAG_PROMPT.format(
        raw_output=truncated,
        title=observation.title,
        summary=observation.summary,
        detail=observation.detail or "(none)",
        files_touched=observation.files_touched,
        functions_changed=observation.functions_changed,
    )

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
    except Exception:
        logger.warning("QAG API call failed", exc_info=True)
        return 0.0, 0.0

    data = _parse_json_response(text)
    if data is None:
        return 0.0, 0.0

    questions = data.get("questions", [])

    # Validate question count (2-5 per Amendment 3)
    if len(questions) < 2:
        return 0.0, 0.0
    if len(questions) > 5:
        questions = questions[:5]

    answerable = sum(1 for q in questions if q.get("answerable"))
    info_score = answerable / len(questions)
    rationale_score = 1.0 if data.get("decision_rationale_present") else 0.0

    return info_score, rationale_score


def _truncate_for_eval(raw_output: str, *, max_chars: int = 4000) -> str:
    """Truncate raw_output for QAG evaluation using head+tail strategy."""
    if len(raw_output) <= max_chars:
        return raw_output

    half = max_chars // 2
    return (
        raw_output[:half]
        + f"\n\n[... truncated {len(raw_output) - max_chars} chars ...]\n\n"
        + raw_output[-half:]
    )


def _parse_json_response(text: str) -> dict | None:
    """Parse JSON from LLM response, handling markdown fences."""
    cleaned = text.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # Try direct parse
    try:
        result: dict = json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    else:
        return result

    # Find JSON object boundaries
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass
        else:
            return result

    logger.warning("Could not parse QAG response as JSON: %s", text[:200])
    return None


def compute_composite_quality(
    structural_validity: float,
    title_quality: float,
    info_preservation: float,
    decision_rationale: float,
) -> float:
    """Compute weighted composite quality score.

    Formula: 0.15 * structural + 0.10 * title + 0.50 * info + 0.25 * rationale
    """
    return (
        0.15 * structural_validity
        + 0.10 * title_quality
        + 0.50 * info_preservation
        + 0.25 * decision_rationale
    )
