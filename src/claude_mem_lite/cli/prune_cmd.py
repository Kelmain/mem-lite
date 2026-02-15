"""CLI prune command: clean up old data from the database."""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()


def _parse_duration(s: str) -> str:
    """Parse duration string into ISO datetime cutoff.

    Supports formats: "30d" (days), "4w" (weeks), "2m" (months).
    Returns ISO datetime string for the cutoff point.
    """
    match = re.match(r"^(\d+)([dwm])$", s.strip().lower())
    if not match:
        msg = f"Invalid duration: {s!r}. Use format: 30d, 4w, 2m"
        raise typer.BadParameter(msg)

    value = int(match.group(1))
    unit = match.group(2)

    now = datetime.now(UTC)
    if unit == "d":
        cutoff = now - timedelta(days=value)
    elif unit == "w":
        cutoff = now - timedelta(weeks=value)
    elif unit == "m":
        cutoff = now - timedelta(days=value * 30)
    else:
        msg = f"Unknown unit: {unit}"
        raise typer.BadParameter(msg)

    return cutoff.isoformat()


def _count_prunable(
    conn: sqlite3.Connection,
    cutoff: str,
    keep_raw: int,
) -> tuple[int, int]:
    """Count items that would be pruned.

    Returns (raw_output_count, event_log_count).
    keep_raw preserves the N newest among items older than cutoff.
    """
    raw_count = conn.execute(
        "SELECT COUNT(*) FROM pending_queue "
        "WHERE created_at < ? AND status = 'done' "
        "AND raw_output IS NOT NULL AND raw_output != '' "
        "AND id NOT IN ("
        "  SELECT id FROM pending_queue "
        "  WHERE created_at < ? AND status = 'done' "
        "  AND raw_output IS NOT NULL AND raw_output != '' "
        "  ORDER BY created_at DESC LIMIT ?"
        ")",
        (cutoff, cutoff, keep_raw),
    ).fetchone()[0]

    event_count = conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE created_at < ?",
        (cutoff,),
    ).fetchone()[0]

    return raw_count, event_count


def _prune_raw_output(
    conn: sqlite3.Connection,
    cutoff: str,
    keep_raw: int,
) -> int:
    """Clear raw_output from old pending_queue items, preserving newest N.

    keep_raw preserves the N newest among items older than cutoff.
    Returns number of items cleaned.
    """
    cursor = conn.execute(
        "UPDATE pending_queue SET raw_output = '' "
        "WHERE created_at < ? AND status = 'done' "
        "AND raw_output IS NOT NULL AND raw_output != '' "
        "AND id NOT IN ("
        "  SELECT id FROM pending_queue "
        "  WHERE created_at < ? AND status = 'done' "
        "  AND raw_output IS NOT NULL AND raw_output != '' "
        "  ORDER BY created_at DESC LIMIT ?"
        ")",
        (cutoff, cutoff, keep_raw),
    )
    return cursor.rowcount


def _prune_event_log(conn: sqlite3.Connection, cutoff: str) -> int:
    """Delete event_log entries older than cutoff. Returns count deleted."""
    cursor = conn.execute(
        "DELETE FROM event_log WHERE created_at < ?",
        (cutoff,),
    )
    return cursor.rowcount


def prune_cmd(
    older_than: Annotated[
        str, typer.Option("--older-than", help="Age threshold (e.g. 30d, 4w, 2m).")
    ] = "30d",
    keep_raw: Annotated[
        int, typer.Option("--keep-raw", help="Number of newest raw_outputs to preserve.")
    ] = 100,
    vacuum: Annotated[
        bool, typer.Option("--vacuum", help="Run VACUUM to reclaim disk space.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be pruned without changes.")
    ] = False,
) -> None:
    """Prune old data from the database to reclaim space."""
    from claude_mem_lite.config import Config

    config = Config()
    db_path = str(config.db_path)

    if not Path(db_path).exists():
        console.print("[red]Database not found.[/red]")
        raise typer.Exit(code=1)

    cutoff = _parse_duration(older_than)

    conn = sqlite3.connect(db_path)
    try:
        raw_count, event_count = _count_prunable(conn, cutoff, keep_raw)

        if raw_count == 0 and event_count == 0:
            console.print("[dim]Nothing to prune.[/dim]")
            if vacuum:
                _do_vacuum(conn)
            return

        if dry_run:
            console.print("[bold]Dry run:[/bold] No changes will be made.\n")
            console.print(f"  Raw outputs to clear: {raw_count}")
            console.print(f"  Event log entries to delete: {event_count}")
            console.print(f"  Cutoff: {cutoff}")
            console.print(f"  Keep newest: {keep_raw}")
            return

        # Execute pruning
        cleaned = _prune_raw_output(conn, cutoff, keep_raw)
        deleted = _prune_event_log(conn, cutoff)
        conn.commit()

        console.print(f"Pruned [green]{cleaned}[/green] raw outputs")
        console.print(f"Deleted [green]{deleted}[/green] event log entries")

        if vacuum:
            _do_vacuum(conn)

    finally:
        conn.close()


def _do_vacuum(conn: sqlite3.Connection) -> None:
    """Run VACUUM with reactive error handling for locked database."""
    try:
        conn.execute("VACUUM")
        console.print("[green]VACUUM completed.[/green]")
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            console.print("[red]VACUUM failed -- database is locked.[/red]")
            console.print("  Stop the worker first: claude-mem-worker stop")
        else:
            raise
