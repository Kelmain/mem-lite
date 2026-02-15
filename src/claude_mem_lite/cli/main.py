"""Root Typer app for claude-mem CLI."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="claude-mem",
    help="claude-mem-lite: Local memory for Claude Code sessions.",
    no_args_is_help=True,
)


def _register_commands() -> None:
    """Register all CLI commands and sub-apps."""
    from claude_mem_lite.cli.eval_cmd import eval_app
    from claude_mem_lite.cli.mermaid_cmd import mermaid_cmd
    from claude_mem_lite.cli.report import report_cmd
    from claude_mem_lite.cli.search_cmd import search_cmd
    from claude_mem_lite.cli.status_cmd import status_cmd

    app.command(name="report")(report_cmd)
    app.command(name="search")(search_cmd)
    app.command(name="mermaid")(mermaid_cmd)
    app.command(name="status")(status_cmd)
    app.add_typer(eval_app, name="eval")


_register_commands()
