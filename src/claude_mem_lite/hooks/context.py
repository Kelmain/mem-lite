"""SessionStart hook -- create session, inject context."""

from __future__ import annotations

import json
import os
import sys


def _build_minimal_context(store) -> str:
    """Build lightweight context from recent sessions."""
    sessions = store.list_sessions(limit=5)
    if not sessions:
        return ""

    lines = ["## Recent Sessions (claude-mem-lite)"]
    for s in sessions:
        summary = s.summary or "no summary"
        status = s.status
        lines.append(f"- [{s.started_at}] {summary} ({status})")

    lines.append("")
    lines.append("Use `curl` to search past work: see mem-search skill.")
    return "\n".join(lines)


def _process_event(event: dict, store) -> str:
    """Core context logic. Creates session, returns context text."""
    session_id = event.get("session_id", "")
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd", "")

    if session_id:
        store.create_session(project_dir, id=session_id)

    return _build_minimal_context(store)


def main() -> None:
    """CLI entry point: read event from stdin, output context to stdout."""
    try:
        try:
            event = json.load(sys.stdin)
        except (json.JSONDecodeError, EOFError):
            sys.exit(0)

        from claude_mem_lite.config import Config
        from claude_mem_lite.storage.sqlite_store import SQLiteStore

        config = Config()
        config.ensure_dirs()
        store = SQLiteStore(config.db_path)
        context_text = _process_event(event, store)
        store.close()

        if context_text:
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": context_text,
                }
            }
            json.dump(output, sys.stdout)

    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
