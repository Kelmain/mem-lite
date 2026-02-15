"""SessionStart hook -- create session, inject context.

Context injection modes:
1. Worker available: Full progressive disclosure (all layers)
2. Worker unavailable: Basic session list from SQLite (lightweight)
3. Empty DB: No context (brand new project, nothing to inject)
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import sys
from urllib.parse import quote


def _get_worker_context(socket_path: str, project_path: str) -> str:
    """Call worker's /api/context endpoint over UDS.

    Uses stdlib http.client to avoid importing httpx in the hook
    (which would add ~200ms cold import time).
    """
    if not hasattr(socket, "AF_UNIX"):
        return ""

    try:
        conn = _UDSHTTPConnection(socket_path)
        encoded_path = quote(project_path, safe="")
        conn.request("GET", f"/api/context?project_path={encoded_path}")
        conn.sock.settimeout(5)
        response = conn.getresponse()
        if response.status == 200:
            data = json.loads(response.read())
            result: str = data.get("context", "")
            return result
    except (ConnectionRefusedError, TimeoutError, OSError, json.JSONDecodeError):
        return ""
    return ""


class _UDSHTTPConnection(http.client.HTTPConnection):
    """HTTP connection over Unix Domain Socket using stdlib only."""

    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")
        self._socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._socket_path)
        self.sock.settimeout(5)


def _get_basic_context(store) -> str:
    """Build basic context directly from SQLite (no worker needed).

    Only includes session index — the cheapest, most useful layer.
    """
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

    return _get_basic_context(store)


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

        # Create/ensure session exists
        _process_event(event, store)

        # Try worker path, fall back to basic SQLite context
        context_text = ""
        socket_path = str(config.socket_path)
        project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd", "")

        if config.socket_path.exists():
            context_text = _get_worker_context(socket_path, project_dir)

        if not context_text:
            context_text = _get_basic_context(store)

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
