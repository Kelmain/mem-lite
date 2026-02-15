"""CLI search command: search observations with worker/FTS fallback."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()


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


def _search_worker(query: str, limit: int, socket_path: str) -> list[dict] | None:
    """Search via running worker. Returns results or None on error."""
    try:
        import httpx

        transport = httpx.HTTPTransport(uds=socket_path)
        with httpx.Client(transport=transport, timeout=5.0) as client:
            resp = client.get(
                "http://localhost/api/search",
                params={"q": query, "limit": limit},
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("results", [])
    except Exception:
        pass
    return None


def _search_fts(
    query: str,
    limit: int,
    db_path: str,
    type_filter: str | None = None,
) -> list[dict]:
    """Fallback FTS5 search directly against SQLite."""
    if not Path(db_path).exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Check if FTS table exists
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='observations_fts'"
        ).fetchall()
        if not tables:
            return []

        base_query = (
            "SELECT o.id, o.title, o.summary, o.created_at, o.tool_name, o.files_touched "
            "FROM observations o "
            "JOIN observations_fts fts ON fts.rowid = o.rowid "
            "WHERE observations_fts MATCH ?"
        )
        params: list = [query]

        if type_filter:
            base_query += " AND o.tool_name = ?"
            params.append(type_filter)

        base_query += " ORDER BY rank LIMIT ?"
        params.append(limit)

        rows = conn.execute(base_query, params).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "summary": r["summary"],
                "created_at": r["created_at"],
                "tool_name": r["tool_name"],
                "files_touched": r["files_touched"],
            }
            for r in rows
        ]
    except Exception:
        return []
    finally:
        conn.close()


def search_cmd(
    query: Annotated[str, typer.Argument(help="Search query string.")],
    limit: Annotated[int, typer.Option(help="Maximum results to return.")] = 5,
    type_filter: Annotated[
        str | None, typer.Option("--type", help="Filter by type (observation/code/learning).")
    ] = None,
    output_json: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """Search observations using worker or FTS fallback."""
    from claude_mem_lite.config import Config

    config = Config()
    db_path = str(config.db_path)

    if not Path(db_path).exists():
        console.print("[red]Database not found.[/red] Run claude-mem first to create it.")
        raise typer.Exit(code=1)

    results = None
    source = "worker"

    # Try worker first
    socket_path = _discover_worker(config)
    if socket_path:
        results = _search_worker(query, limit, socket_path)

    # Fallback to FTS on worker unavailable or error (Amendment 5)
    if results is None:
        source = "fts"
        results = _search_fts(query, limit, db_path, type_filter)

    if output_json:
        console.print(json.dumps({"results": results, "source": source}, indent=2))
        return

    if not results:
        console.print("[dim]No results found.[/dim]")
        return

    table = Table(title=f"Search Results ({source})", show_lines=True)
    table.add_column("Title", max_width=40)
    table.add_column("Summary", max_width=50)
    table.add_column("Tool")
    table.add_column("Time")

    for r in results:
        table.add_row(
            r.get("title", ""),
            r.get("summary", "")[:100],
            r.get("tool_name", ""),
            r.get("created_at", ""),
        )
    console.print(table)
