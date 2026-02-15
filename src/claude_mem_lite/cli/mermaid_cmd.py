"""CLI mermaid command: generate Mermaid diagrams from function map."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from claude_mem_lite.ast_tracker.mermaid import generate_mermaid
from claude_mem_lite.ast_tracker.types import CallInfo, FunctionInfo

console = Console()


def _row_to_function_info(row: sqlite3.Row, call_edges: list[sqlite3.Row]) -> FunctionInfo:
    """Hydrate FunctionInfo from SQL row + call edges.

    Args:
        row: A function_map row with Row factory.
        call_edges: Call graph edges from call_graph table.

    Returns:
        FunctionInfo dataclass instance.
    """
    qualified_name = row["qualified_name"]
    calls = [
        CallInfo(
            raw_name=edge["callee_function"],
            resolved_name=edge["callee_function"],
            resolution=edge["resolution"] if edge["resolution"] else "direct",
            line_number=0,
        )
        for edge in call_edges
        if edge["caller_function"] == qualified_name
    ]
    return FunctionInfo(
        qualified_name=qualified_name,
        kind=row["kind"] if row["kind"] else "function",
        parent_class=None,  # Not stored in function_map table
        signature=row["signature"],
        decorators=json.loads(row["decorators"]) if row["decorators"] else [],
        docstring=row["docstring"],
        line_start=0,  # Not stored in function_map table
        line_end=0,  # Not stored in function_map table
        body_hash=row["body_hash"],
        calls=calls,
    )


def _query_functions_and_edges(
    conn: sqlite3.Connection,
    *,
    file_path: str | None = None,
    session_id: str | None = None,
    show_all: bool = False,
) -> tuple[dict[str, list[FunctionInfo]], dict[str, dict[str, str]], list[sqlite3.Row]]:
    """Query function_map and call_graph, grouped by file.

    Returns:
        Tuple of (functions_by_file, change_types_by_file, call_edges).
    """
    # Build function query
    fn_query = "SELECT * FROM function_map WHERE 1=1"
    fn_params: list = []
    if file_path:
        fn_query += " AND file_path = ?"
        fn_params.append(file_path)
    if session_id:
        fn_query += " AND session_id = ?"
        fn_params.append(session_id)
    if not show_all:
        fn_query += " AND change_type != 'unchanged'"
    fn_query += " ORDER BY file_path, qualified_name"

    fn_rows = conn.execute(fn_query, fn_params).fetchall()

    # Build call graph query
    cg_query = "SELECT * FROM call_graph WHERE 1=1"
    cg_params: list = []
    if file_path:
        cg_query += " AND caller_file = ?"
        cg_params.append(file_path)
    if session_id:
        cg_query += " AND session_id = ?"
        cg_params.append(session_id)

    call_edges = conn.execute(cg_query, cg_params).fetchall()

    # Group by file
    functions_by_file: dict[str, list[FunctionInfo]] = {}
    change_types_by_file: dict[str, dict[str, str]] = {}

    for row in fn_rows:
        fp = row["file_path"]
        fi = _row_to_function_info(row, call_edges)
        functions_by_file.setdefault(fp, []).append(fi)
        change_types_by_file.setdefault(fp, {})[row["qualified_name"]] = row["change_type"]

    return functions_by_file, change_types_by_file, call_edges


def mermaid_cmd(
    file: Annotated[str | None, typer.Argument(help="File path to generate graph for.")] = None,
    session: Annotated[str | None, typer.Option(help="Session ID to scope to.")] = None,
    show_all: Annotated[bool, typer.Option("--all", help="Include unchanged functions.")] = False,
    output: Annotated[
        str | None, typer.Option("-o", "--output", help="Write output to file.")
    ] = None,
) -> None:
    """Generate Mermaid diagram from function map data."""
    from claude_mem_lite.config import Config

    config = Config()
    db_path = str(config.db_path)

    if not Path(db_path).exists():
        console.print("[red]Database not found.[/red] Run claude-mem first to create it.")
        raise typer.Exit(code=1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        functions_by_file, change_types_by_file, _ = _query_functions_and_edges(
            conn,
            file_path=file,
            session_id=session,
            show_all=show_all,
        )

        if not functions_by_file:
            console.print("[dim]No function changes found.[/dim]")
            return

        # Generate mermaid for each file
        diagrams: list[str] = []
        for fp, funcs in sorted(functions_by_file.items()):
            ct = change_types_by_file.get(fp, {})
            diagram = generate_mermaid(
                funcs,
                file_path=fp,
                show_all=show_all,
                change_types=ct,
            )
            diagrams.append(diagram)

        # Combine multi-file diagrams
        if len(diagrams) == 1:
            result = diagrams[0]
        else:
            # Merge into a single graph with subgraphs
            lines = ["graph TD"]
            for diagram in diagrams:
                # Skip the "graph TD" header from each sub-diagram
                for line in diagram.splitlines():
                    if line.strip() != "graph TD":
                        lines.append(line)
            result = "\n".join(lines)

        if output:
            Path(output).write_text(result)
            console.print(f"Diagram written to {output}")
        else:
            console.print(result)
    finally:
        conn.close()
