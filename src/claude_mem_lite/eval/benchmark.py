"""Offline A/B benchmark runner for compression model comparison."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from claude_mem_lite.eval.evaluator import (
    compute_composite_quality,
    score_deterministic,
    score_info_preservation,
)
from claude_mem_lite.eval.models import (
    BenchmarkReport,
    CompressionScore,
    PairedResult,
    QueueSample,
)
from claude_mem_lite.storage.models import Observation

if TYPE_CHECKING:
    import aiosqlite

    from claude_mem_lite.config import Config

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Offline A/B comparison between compression models."""

    def __init__(
        self,
        db: aiosqlite.Connection,
        client: object,
        config: Config,
    ) -> None:
        self.db = db
        self.client = client
        self.config = config

    async def run(
        self,
        model_a: str = "claude-haiku-4-5-20251001",
        model_b: str = "claude-sonnet-4-5-20250929",
        sample_size: int = 30,
        judge_model: str = "claude-sonnet-4-5-20250929",
    ) -> BenchmarkReport:
        """Run head-to-head compression comparison.

        1. Sample raw outputs from pending_queue (status='done')
        2. Compress each through model_a and model_b
        3. Score each compression (deterministic + QAG)
        4. Produce paired comparison report
        """
        samples = await self._sample_raw_outputs(sample_size)

        if not samples:
            return self._build_report([], model_a, model_b)

        results: list[PairedResult] = []
        for sample in samples:
            try:
                pair = await self._evaluate_sample(sample, model_a, model_b, judge_model)
                results.append(pair)
            except Exception:
                logger.warning(
                    "Failed to evaluate sample %s, skipping",
                    sample.id,
                    exc_info=True,
                )

        return self._build_report(results, model_a, model_b)

    async def _sample_raw_outputs(self, n: int) -> list[QueueSample]:
        """Sample diverse raw outputs for benchmarking."""
        cursor = await self.db.execute(
            "SELECT id, session_id, tool_name, raw_output, files_touched "
            "FROM pending_queue "
            "WHERE status = 'done' AND raw_output IS NOT NULL "
            "ORDER BY RANDOM() LIMIT ?",
            (n,),
        )
        rows = await cursor.fetchall()
        return [
            QueueSample(
                id=row["id"],
                session_id=row["session_id"],
                tool_name=row["tool_name"],
                raw_output=row["raw_output"],
                files_touched=row["files_touched"],
            )
            for row in rows
        ]

    async def _evaluate_sample(
        self,
        sample: QueueSample,
        model_a: str,
        model_b: str,
        judge_model: str,
    ) -> PairedResult:
        """Compress a sample through both models and score."""
        comp_a, latency_a = await self._compress_with_model(sample, model_a)
        comp_b, latency_b = await self._compress_with_model(sample, model_b)

        det_a = score_deterministic(comp_a, sample.raw_output, latency_a, model_a, self.config)
        det_b = score_deterministic(comp_b, sample.raw_output, latency_b, model_b, self.config)

        info_a, rationale_a = await score_info_preservation(
            sample.raw_output, comp_a, self.client, judge_model
        )
        info_b, rationale_b = await score_info_preservation(
            sample.raw_output, comp_b, self.client, judge_model
        )

        return PairedResult(
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
        )

    async def _compress_with_model(
        self,
        sample: QueueSample,
        model: str,
    ) -> tuple[Observation, int]:
        """Compress a sample using the given model, return Observation + latency_ms."""
        from claude_mem_lite.worker.prompts import build_compression_prompt

        prompt = build_compression_prompt(sample.raw_output, sample.tool_name, sample.files_touched)

        start = time.monotonic()
        response = await self.client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        text = response.content[0].text
        data = json.loads(text)

        return (
            Observation(
                id=sample.id,
                session_id=sample.session_id,
                tool_name=sample.tool_name,
                title=data.get("title", ""),
                summary=data.get("summary", ""),
                detail=data.get("detail"),
                files_touched=json.dumps(data.get("files_touched", [])),
                functions_changed=json.dumps(data.get("functions_changed", [])),
                tokens_raw=response.usage.input_tokens,
                tokens_compressed=response.usage.output_tokens,
                created_at=datetime.now(UTC).isoformat(),
            ),
            latency_ms,
        )

    def _build_report(
        self,
        results: list[PairedResult],
        model_a: str,
        model_b: str,
    ) -> BenchmarkReport:
        """Build aggregate report from paired results."""
        n = len(results)

        if n == 0:
            return BenchmarkReport(
                model_a=model_a,
                model_b=model_b,
                sample_size=0,
                timestamp=datetime.now(UTC).isoformat(),
                avg_quality_a=0.0,
                avg_quality_b=0.0,
                avg_cost_a=0.0,
                avg_cost_b=0.0,
                avg_latency_a=0.0,
                avg_latency_b=0.0,
                quality_per_dollar_a=0.0,
                quality_per_dollar_b=0.0,
                a_wins=0,
                b_wins=0,
                ties=0,
                info_preservation_a=0.0,
                info_preservation_b=0.0,
                rationale_rate_a=0.0,
                rationale_rate_b=0.0,
                structural_validity_a=0.0,
                structural_validity_b=0.0,
                pairs=[],
            )

        qualities_a = []
        qualities_b = []
        a_wins = 0
        b_wins = 0
        ties = 0

        for r in results:
            qa = compute_composite_quality(
                r.score_a.deterministic.structural_validity,
                r.score_a.deterministic.title_quality,
                r.score_a.info_preservation,
                r.score_a.decision_rationale,
            )
            qb = compute_composite_quality(
                r.score_b.deterministic.structural_validity,
                r.score_b.deterministic.title_quality,
                r.score_b.info_preservation,
                r.score_b.decision_rationale,
            )
            qualities_a.append(qa)
            qualities_b.append(qb)

            diff = qa - qb
            if abs(diff) < 0.05:
                ties += 1
            elif diff > 0:
                a_wins += 1
            else:
                b_wins += 1

        avg_quality_a = sum(qualities_a) / n
        avg_quality_b = sum(qualities_b) / n
        avg_cost_a = sum(r.score_a.deterministic.cost_usd for r in results) / n
        avg_cost_b = sum(r.score_b.deterministic.cost_usd for r in results) / n
        avg_latency_a = sum(r.score_a.deterministic.latency_ms for r in results) / n
        avg_latency_b = sum(r.score_b.deterministic.latency_ms for r in results) / n

        qpd_a = avg_quality_a / avg_cost_a if avg_cost_a > 0 else 0.0
        qpd_b = avg_quality_b / avg_cost_b if avg_cost_b > 0 else 0.0

        return BenchmarkReport(
            model_a=model_a,
            model_b=model_b,
            sample_size=n,
            timestamp=datetime.now(UTC).isoformat(),
            avg_quality_a=avg_quality_a,
            avg_quality_b=avg_quality_b,
            avg_cost_a=avg_cost_a,
            avg_cost_b=avg_cost_b,
            avg_latency_a=avg_latency_a,
            avg_latency_b=avg_latency_b,
            quality_per_dollar_a=qpd_a,
            quality_per_dollar_b=qpd_b,
            a_wins=a_wins,
            b_wins=b_wins,
            ties=ties,
            info_preservation_a=sum(r.score_a.info_preservation for r in results) / n,
            info_preservation_b=sum(r.score_b.info_preservation for r in results) / n,
            rationale_rate_a=sum(r.score_a.decision_rationale for r in results) / n,
            rationale_rate_b=sum(r.score_b.decision_rationale for r in results) / n,
            structural_validity_a=sum(r.score_a.deterministic.structural_validity for r in results)
            / n,
            structural_validity_b=sum(r.score_b.deterministic.structural_validity for r in results)
            / n,
            pairs=results,
        )
