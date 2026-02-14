"""Stop hook -- mark session for summary generation."""

from __future__ import annotations

import json
import sys


def _should_skip(event: dict) -> bool:
    """Check if we should skip processing (e.g., stop_hook_active loop guard)."""
    return bool(event.get("stop_hook_active", False))


def _process_event(event: dict, store, logger) -> None:
    """Core stop logic. Logs stop event with pending count."""
    session_id = event.get("session_id", "unknown")
    obs_count = store.count_pending(session_id)
    logger.log(
        "hook.stop",
        {
            "session_id": session_id,
            "pending_observations": obs_count,
        },
        session_id=session_id,
    )


def main() -> None:
    """CLI entry point: read event from stdin, log stop event."""
    try:
        try:
            event = json.load(sys.stdin)
        except (json.JSONDecodeError, EOFError):
            sys.exit(0)

        if _should_skip(event):
            sys.exit(0)

        from claude_mem_lite.config import Config
        from claude_mem_lite.logging.logger import MemLogger
        from claude_mem_lite.storage.sqlite_store import SQLiteStore

        config = Config()
        config.ensure_dirs()
        store = SQLiteStore(config.db_path)
        logger = MemLogger(config.log_dir, store.conn)
        _process_event(event, store, logger)
        store.close()

        json.dump({"continue": True, "suppressOutput": True}, sys.stdout)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
