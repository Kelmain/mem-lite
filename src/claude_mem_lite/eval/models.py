"""Pydantic models for the eval framework."""

from __future__ import annotations

from pydantic import BaseModel


class DeterministicScore(BaseModel):
    """Deterministic scoring of a compressed observation (no API calls)."""

    structural_validity: float
    compression_ratio: float
    title_quality: float
    cost_usd: float
    latency_ms: int


class QAGResult(BaseModel):
    """Result of QAG-based information preservation scoring."""

    questions: list[dict]
    questions_generated: int
    info_preservation: float
    decision_rationale: float
    rationale_note: str


class CompressionScore(BaseModel):
    """Combined score for a single compression."""

    model: str
    deterministic: DeterministicScore
    info_preservation: float
    decision_rationale: float


class PairedResult(BaseModel):
    """Head-to-head comparison result for one sample."""

    sample_id: str
    tool_name: str
    raw_size: int
    score_a: CompressionScore
    score_b: CompressionScore


class BenchmarkReport(BaseModel):
    """Aggregate report from an A/B benchmark run."""

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
    qag: QAGResult | None = None
    composite_quality: float


class QueueSample(BaseModel):
    """A pending_queue row sampled for benchmarking."""

    id: str
    session_id: str
    tool_name: str
    raw_output: str
    files_touched: str
