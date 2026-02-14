"""PostToolUse hook -- capture tool execution data to pending_queue."""

from __future__ import annotations

import json
import os
import sys

# Tools that represent mutations or actions -- always compress individually
HIGH_VALUE_TOOLS = {"Write", "Edit", "MultiEdit", "Bash"}

# Tools that represent information gathering -- batch-compress or skip
LOW_VALUE_TOOLS = {"Read", "Glob", "Grep", "TodoRead", "TodoWrite"}


def _extract_files(tool_name: str, tool_input: dict) -> list[str]:
    """Extract file paths from tool_input based on tool type."""
    files: list[str] = []
    if tool_name in ("Write", "Edit", "MultiEdit", "Read"):
        fp = tool_input.get("file_path")
        if fp:
            files.append(fp)
    return files


def _process_event(event: dict, store) -> None:
    """Core capture logic. Testable without stdin/stdout."""
    session_id = event.get("session_id", "unknown")
    tool_name = event.get("tool_name", "unknown")

    # Determine priority
    if tool_name in HIGH_VALUE_TOOLS:
        priority = "high"
    elif tool_name in LOW_VALUE_TOOLS:
        priority = "low"
    else:
        priority = "normal"

    tool_input = event.get("tool_input", {})
    files_touched = _extract_files(tool_name, tool_input)

    # Resolve project directory (CLAUDE_PROJECT_DIR preferred over cwd)
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd", "")

    raw_payload = {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_response": event.get("tool_response", {}),
        "project_dir": project_dir,
    }

    store.enqueue(
        session_id=session_id,
        tool_name=tool_name,
        raw_output=json.dumps(raw_payload, default=str),
        files_touched=files_touched,
        priority=priority,
    )


def main() -> None:
    """CLI entry point: read event from stdin, enqueue to pending_queue."""
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
        _process_event(event, store)
        store.close()

        json.dump({"continue": True, "suppressOutput": True}, sys.stdout)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
