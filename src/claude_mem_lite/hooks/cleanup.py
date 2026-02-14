"""SessionEnd hook -- close session, checkpoint WAL."""

from __future__ import annotations

import json
import sys

from claude_mem_lite.storage.models import SessionStatus


def _process_event(event: dict, store, logger) -> None:
    """Core cleanup logic. Closes session, checkpoints, logs."""
    session_id = event.get("session_id", "unknown")
    reason = event.get("reason", "other")

    store.update_session(session_id, status=SessionStatus.CLOSED)
    store.checkpoint(mode="PASSIVE")

    logger.log(
        "hook.session_end",
        {
            "session_id": session_id,
            "reason": reason,
        },
        session_id=session_id,
    )


def main() -> None:
    """CLI entry point: read event from stdin, close session."""
    try:
        try:
            event = json.load(sys.stdin)
        except (json.JSONDecodeError, EOFError):
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

    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
