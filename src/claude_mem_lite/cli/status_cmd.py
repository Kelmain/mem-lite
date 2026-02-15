"""CLI status command: system health check."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import typer
from rich.console import Console

console = Console()


def _check_fts_exists(conn: sqlite3.Connection) -> bool:
    """Check if observations_fts virtual table exists."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='observations_fts'"
    ).fetchone()
    return row is not None


def _check_worker(config: object) -> bool:
    """Check if worker is running via socket health check."""
    socket_path = config.socket_path
    if not socket_path.exists():
        return False
    try:
        import httpx

        transport = httpx.HTTPTransport(uds=str(socket_path))
        with httpx.Client(transport=transport, timeout=2.0) as client:
            resp = client.get("http://localhost/api/health")
            return resp.status_code == 200
    except Exception:
        return False


def status_cmd() -> None:
    """Show system health status."""
    from claude_mem_lite.config import Config

    config = Config()
    db_path = str(config.db_path)

    console.print("[bold]claude-mem-lite status[/bold]\n")

    # Config path
    console.print(f"Config dir: {config.base_dir}")
    console.print(f"Database:   {config.db_path}")
    console.print(f"Socket:     {config.socket_path}")
    console.print()

    # Database check
    if not Path(db_path).exists():
        console.print("[red]Database: NOT FOUND[/red]")
        raise typer.Exit(code=1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Counts
        obs_count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        learning_count = conn.execute(
            "SELECT COUNT(*) FROM learnings WHERE is_active = 1"
        ).fetchone()[0]

        console.print("[green]Database: OK[/green]")
        console.print(f"  Sessions:     {session_count}")
        console.print(f"  Observations: {obs_count}")
        console.print(f"  Learnings:    {learning_count}")

        # Last session age
        last_session = conn.execute(
            "SELECT started_at FROM sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if last_session:
            console.print(f"  Last session: {last_session['started_at']}")

        # FTS check
        if _check_fts_exists(conn):
            console.print("[green]FTS index: OK[/green]")
        else:
            console.print("[yellow]FTS index: NOT FOUND[/yellow]")

    finally:
        conn.close()

    # Worker check
    if _check_worker(config):
        console.print("[green]Worker: RUNNING[/green]")
    else:
        console.print("[yellow]Worker: NOT RUNNING[/yellow]")
