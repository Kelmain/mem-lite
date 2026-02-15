"""CLI functions for eval commands (Phase 7) + Typer wrappers (Phase 8)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console

from claude_mem_lite.eval.evaluator import (
    compute_composite_quality,
    score_deterministic,
)
from claude_mem_lite.eval.queries import (
    COMPRESSION_MONITORING,
    SEARCH_QUALITY,
    SYSTEM_HEALTH_DAILY,
)

if TYPE_CHECKING:
    import aiosqlite

    from claude_mem_lite.config import Config
    from claude_mem_lite.eval.models import BenchmarkReport

eval_app = typer.Typer(help="Evaluate compression quality and system health.")
eval_console = Console()


def eval_compression(
    *,
    conn: sqlite3.Connection,
    config: Config,
    limit: int = 20,
    _with_qag: bool = False,
    as_json: bool = False,
    since: str | None = None,
) -> list[dict] | str:
    """Evaluate existing observations with deterministic scoring.

    Args:
        conn: Sync SQLite connection with migrated schema.
        config: Config with MODEL_RATES.
        limit: Number of recent observations to evaluate.
        _with_qag: Reserved for future QAG scoring support.
        as_json: Return JSON string instead of list of dicts.
        since: Only evaluate observations after this date.

    Returns:
        List of result dicts, or JSON string if as_json=True.
    """
    query = (
        "SELECT o.id, o.title, o.summary, o.detail, o.files_touched, "
        "o.functions_changed, o.tokens_raw, o.tokens_compressed, o.created_at, "
        "o.session_id, o.tool_name, "
        "p.raw_output "
        "FROM observations o "
        "LEFT JOIN pending_queue p ON o.id = p.id "
        "WHERE 1=1"
    )
    params: list = []

    if since:
        query += " AND o.created_at >= ?"
        params.append(since)

    query += " ORDER BY o.created_at DESC LIMIT ?"
    params.append(limit)

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    results = []
    for row in rows:
        from claude_mem_lite.storage.models import Observation

        obs = Observation(
            id=row[0],
            session_id=row[9],
            tool_name=row[10],
            title=row[1],
            summary=row[2],
            detail=row[3],
            files_touched=row[4] or "[]",
            functions_changed=row[5] or "[]",
            tokens_raw=row[6],
            tokens_compressed=row[7],
            created_at=row[8],
        )
        raw_output = row[11] or ""
        model = config.compression_model

        det = score_deterministic(obs, raw_output, 0, model, config)
        quality = compute_composite_quality(
            det.structural_validity,
            det.title_quality,
            0.0,  # info_preservation (requires QAG)
            0.0,  # decision_rationale (requires QAG)
        )

        results.append(
            {
                "observation_id": obs.id,
                "title": obs.title,
                "deterministic": det.model_dump(),
                "composite_quality": quality,
            }
        )

    # Log eval event (best-effort)
    with contextlib.suppress(Exception):
        conn.execute(
            "INSERT INTO event_log (id, event_type, data) VALUES (?, ?, ?)",
            (
                str(uuid.uuid4()),
                "eval.compression",
                json.dumps({"count": len(results), "limit": limit, "since": since}),
            ),
        )

    if as_json:
        return json.dumps(results, indent=2)

    return results


async def eval_benchmark(
    *,
    db: aiosqlite.Connection,
    client: object,
    config: Config,
    model_a: str = "claude-haiku-4-5-20251001",
    model_b: str = "claude-sonnet-4-5-20250929",
    samples: int = 30,
    judge_model: str = "claude-sonnet-4-5-20250929",
    as_json: bool = False,
) -> BenchmarkReport | str:
    """Run offline A/B benchmark between two models.

    Args:
        db: Async SQLite connection.
        client: Anthropic client (or mock).
        config: Config instance.
        model_a: First model for comparison.
        model_b: Second model for comparison.
        samples: Number of samples to compare.
        judge_model: Model to use for QAG judging.
        as_json: Return JSON string instead of report.

    Returns:
        BenchmarkReport, or JSON string if as_json=True.
    """
    from claude_mem_lite.eval.benchmark import BenchmarkRunner

    runner = BenchmarkRunner(db=db, client=client, config=config)
    report = await runner.run(
        model_a=model_a,
        model_b=model_b,
        sample_size=samples,
        judge_model=judge_model,
    )

    if as_json:
        return report.model_dump_json(indent=2)

    return report


def eval_health(
    *,
    conn: sqlite3.Connection,
    days: int = 7,
    as_json: bool = False,
) -> dict | str:
    """Run SQL analysis queries and return system health data.

    Args:
        conn: Sync SQLite connection with migrated schema.
        days: Period to analyze.
        as_json: Return JSON string instead of dict.

    Returns:
        Dict with health data, or JSON string if as_json=True.
    """
    compression_rows = conn.execute(COMPRESSION_MONITORING).fetchall()
    search_rows = conn.execute(SEARCH_QUALITY).fetchall()
    health_rows = conn.execute(SYSTEM_HEALTH_DAILY).fetchall()

    result = {
        "compression": [
            {
                "day": row[0],
                "observations": row[1],
                "avg_ratio": row[2],
                "avg_compress_ms": row[3],
                "cost_usd": row[6],
            }
            for row in compression_rows
        ],
        "search": [
            {
                "day": row[0],
                "queries": row[1],
                "avg_results": row[2],
                "avg_top_score": row[3],
                "avg_ms": row[4],
            }
            for row in search_rows
        ],
        "health": [
            {
                "day": row[0],
                "sessions": row[1],
                "observations": row[2],
                "compress_errors": row[3],
                "searches": row[4],
                "injections": row[5],
                "compress_cost_usd": row[6],
            }
            for row in health_rows
        ],
        "period_days": days,
    }

    if as_json:
        return json.dumps(result, indent=2)

    return result


# -----------------------------------------------------------------------
# Typer CLI wrappers (Phase 8)
# -----------------------------------------------------------------------


@eval_app.command(name="compression")
def compression_cmd(
    limit: Annotated[int, typer.Option(help="Number of recent observations.")] = 20,
    output_json: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
    since: Annotated[str | None, typer.Option(help="Only after this date.")] = None,
) -> None:
    """Evaluate compression quality with deterministic scoring."""
    from claude_mem_lite.config import Config
    from claude_mem_lite.storage.migrations import migrate

    config = Config()
    db_path = str(config.db_path)

    if not Path(db_path).exists():
        eval_console.print("[red]Database not found.[/red]")
        raise typer.Exit(code=1)

    conn = sqlite3.connect(db_path)
    migrate(conn)
    try:
        result = eval_compression(
            conn=conn, config=config, limit=limit, as_json=output_json, since=since
        )
        eval_console.print(result)
    finally:
        conn.close()


@eval_app.command(name="health")
def health_cmd(
    days: Annotated[int, typer.Option(help="Number of days to analyze.")] = 7,
    output_json: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """Show system health dashboard from event log."""
    from claude_mem_lite.config import Config
    from claude_mem_lite.storage.migrations import migrate

    config = Config()
    db_path = str(config.db_path)

    if not Path(db_path).exists():
        eval_console.print("[red]Database not found.[/red]")
        raise typer.Exit(code=1)

    conn = sqlite3.connect(db_path)
    migrate(conn)
    try:
        result = eval_health(conn=conn, days=days, as_json=output_json)
        eval_console.print(result if isinstance(result, str) else json.dumps(result, indent=2))
    finally:
        conn.close()


@eval_app.command(name="benchmark")
def benchmark_cmd(
    model_a: Annotated[
        str, typer.Option(help="First model for comparison.")
    ] = "claude-haiku-4-5-20251001",
    model_b: Annotated[
        str, typer.Option(help="Second model for comparison.")
    ] = "claude-sonnet-4-5-20250929",
    samples: Annotated[int, typer.Option(help="Number of samples to compare.")] = 30,
    judge_model: Annotated[
        str, typer.Option(help="Model for QAG judging.")
    ] = "claude-sonnet-4-5-20250929",
    output_json: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """Run offline A/B benchmark comparing two compression models."""
    import anthropic

    from claude_mem_lite.config import Config
    from claude_mem_lite.storage.migrations import migrate

    config = Config()
    db_path = str(config.db_path)

    if not Path(db_path).exists():
        eval_console.print("[red]Database not found.[/red]")
        raise typer.Exit(code=1)

    async def _run() -> BenchmarkReport | str:
        import aiosqlite

        db = await aiosqlite.connect(db_path)
        db.row_factory = aiosqlite.Row
        client = anthropic.AsyncAnthropic()
        try:
            return await eval_benchmark(
                db=db,
                client=client,
                config=config,
                model_a=model_a,
                model_b=model_b,
                samples=samples,
                judge_model=judge_model,
                as_json=output_json,
            )
        finally:
            await db.close()

    # Sync migration before async run
    conn = sqlite3.connect(db_path)
    migrate(conn)
    conn.close()

    result = asyncio.run(_run())
    eval_console.print(result)
