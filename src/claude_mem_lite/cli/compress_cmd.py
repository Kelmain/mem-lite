"""CLI compress command: compress pending queue items offline."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

# Guard anthropic import — it's an optional dep at CLI level
try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]


def _discover_worker(config: object) -> str | None:
    """Check if worker is running and healthy, return socket path or None."""
    socket_path = config.socket_path
    if not socket_path.exists():
        return None

    try:
        import httpx

        transport = httpx.HTTPTransport(uds=str(socket_path))
        with httpx.Client(transport=transport, timeout=2.0) as client:
            resp = client.get("http://localhost/api/health")
            if resp.status_code == 200:
                return str(socket_path)
    except Exception:
        pass
    return None


def _get_pending_items(
    conn: sqlite3.Connection,
    limit: int,
) -> list[dict]:
    """Fetch pending queue items."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, session_id, tool_name, raw_output, files_touched, created_at "
        "FROM pending_queue WHERE status = 'raw' "
        "ORDER BY created_at LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def _compress_inline(
    conn: sqlite3.Connection,
    items: list[dict],
    model: str,
) -> tuple[int, int]:
    """Compress items using sync Anthropic client.

    Returns (success_count, error_count).
    Only imports anthropic and worker.prompts -- no lancedb, sentence-transformers, aiosqlite.
    """
    if anthropic is None:
        console.print("[red]anthropic package not installed.[/red] Install with: uv add anthropic")
        raise typer.Exit(code=1)

    from claude_mem_lite.worker.prompts import COMPRESSION_SCHEMA, build_compression_prompt

    client = anthropic.Anthropic()
    success = 0
    errors = 0

    for item in items:
        try:
            prompt = build_compression_prompt(
                raw_output=item["raw_output"],
                tool_name=item["tool_name"],
                files_touched=item["files_touched"],
            )

            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                extra_body={
                    "output_config": {
                        "format": {
                            "type": "json_schema",
                            "schema": COMPRESSION_SCHEMA,
                        }
                    }
                },
            )

            text_block = response.content[0]
            data = json.loads(text_block.text)

            # Store observation
            obs_id = str(uuid.uuid4())
            files_json = json.dumps(data.get("files_touched", []))
            functions_json = json.dumps(data.get("functions_changed", []))

            conn.execute(
                "INSERT INTO observations "
                "(id, session_id, tool_name, title, summary, detail, "
                "files_touched, functions_changed, tokens_raw, tokens_compressed, "
                "embedding_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    obs_id,
                    item["session_id"],
                    item["tool_name"],
                    data.get("title", "")[:200],
                    data.get("summary", "")[:1000],
                    data.get("detail"),
                    files_json,
                    functions_json,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    "pending",
                ),
            )
            conn.execute(
                "UPDATE sessions SET observation_count = observation_count + 1 WHERE id = ?",
                (item["session_id"],),
            )
            conn.execute(
                "UPDATE pending_queue SET status = 'done' WHERE id = ?",
                (item["id"],),
            )
            conn.commit()
            success += 1
            console.print(f"  [green]OK[/green] {item['id']}: {data.get('title', '?')}")

        except Exception as e:
            conn.execute(
                "UPDATE pending_queue SET status = 'error', attempts = attempts + 1 WHERE id = ?",
                (item["id"],),
            )
            conn.commit()
            errors += 1
            console.print(f"  [red]ERR[/red] {item['id']}: {e}")

    return success, errors


def compress_cmd(
    pending: Annotated[
        bool, typer.Option("--pending", help="Compress pending queue items.")
    ] = False,
    limit: Annotated[int, typer.Option(help="Maximum items to process.")] = 50,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show pending items without processing.")
    ] = False,
) -> None:
    """Compress pending queue items into observations."""
    if not pending:
        console.print("Use --pending to compress queued items.")
        console.print("Example: claude-mem compress --pending")
        return

    from claude_mem_lite.config import Config

    config = Config()
    db_path = str(config.db_path)

    if not Path(db_path).exists():
        console.print("[red]Database not found.[/red] Run claude-mem first to create it.")
        raise typer.Exit(code=1)

    conn = sqlite3.connect(db_path)
    try:
        items = _get_pending_items(conn, limit)

        if not items:
            console.print("[dim]No pending items to compress.[/dim]")
            return

        if dry_run:
            console.print(f"[bold]Dry run:[/bold] {len(items)} pending items found\n")
            table = Table(title="Pending Items")
            table.add_column("ID", style="dim")
            table.add_column("Tool")
            table.add_column("Created")
            table.add_column("Output Size", justify="right")

            for item in items:
                table.add_row(
                    item["id"],
                    item["tool_name"],
                    item["created_at"],
                    str(len(item["raw_output"])),
                )
            console.print(table)
            return

        # Check if worker is running — inform user
        worker_socket = _discover_worker(config)
        if worker_socket:
            console.print(
                "[yellow]Worker is running.[/yellow] "
                "The worker will process pending items automatically.\n"
                "To force inline compression, stop the worker first."
            )
            return

        console.print(f"Compressing {len(items)} pending items inline...\n")
        success, errors = _compress_inline(conn, items, config.compression_model)
        console.print(f"\nDone: [green]{success} ok[/green], [red]{errors} errors[/red]")

    finally:
        conn.close()
