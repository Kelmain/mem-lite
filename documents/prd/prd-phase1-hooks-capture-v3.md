# PRD: Phase 1 â€” Hook Scripts + Direct Capture (v3)

**Project**: claude-mem-lite (fork of claude-mem, Python rewrite)
**Phase**: 1 of 9
**Status**: Ready for implementation
**Dependencies**: Phase 0 (Storage Layer) â€” SQLiteStore, Config, MemLogger, Pydantic models
**Estimated effort**: 1-2 sessions (~6-10 hours)
**Python**: 3.14.3 (latest stable)

### Changelog (v2 â†’ v3)

| # | Fix | Severity | What changed |
|---|-----|----------|--------------|
| 1 | `python3` path trap â€” pipx/venv installs can't resolve module | Critical | `install.py` now writes `sys.executable` absolute path instead of bare `python3` |
| 2 | Latency KPIs were impossible (<10ms with Pydantic import) | High | Revised to `<200ms` capture, `<30ms` no-op, with fast-exit-before-import pattern |
| 3 | `cwd` causes project identity fragmentation on `cd` | High | All hooks use `CLAUDE_PROJECT_DIR` env var with `cwd` fallback |
| 4 | `TRUNCATE` checkpoint blocks on active readers | Medium | Changed to `PASSIVE` â€” non-blocking, future-proof for Phase 3 worker |
| 5 | Bash priority too coarse (`ls` = `rm -rf`) | Low | Added TODO for Phase 3 command inspection. No Phase 1 change. |
| 6 | `settings.json` may contain JSONC comments | Medium | `install.py` uses `try/except` with diagnostic error on parse failure |

---

## 1. Purpose & Context

### 1.1 What this phase delivers
End-to-end data flow between Claude Code and claude-mem-lite. When Claude uses a tool, an observation is captured. When a session starts, context can (eventually) be injected. When a session ends, it's closed cleanly.

Specifically:
- 4 hook scripts that fire on Claude Code lifecycle events
- A CLI command (`claude-mem install-hooks`) that registers hooks in `~/.claude/settings.json`
- Direct SQLite writes from hooks â€” no worker dependency, no HTTP
- Session lifecycle: create on start, capture on tool use, mark on stop, close on end

### 1.2 What this phase does NOT deliver
- **AI compression** â€” raw tool data goes to `pending_queue`, compressed in Phase 3
- **Context injection** â€” SessionStart hook returns empty context; wired in Phase 5
- **Session summarization** â€” Stop hook marks session for summary; actual AI summary in Phase 3
- **Worker communication** â€” all hooks use direct SQLite; HTTP calls to worker added when worker exists (Phase 3)
- **AST scanning** â€” file-level code intelligence added in Phase 2

### 1.3 Relationship to claude-mem (original)
claude-mem uses 6 hook scripts (TypeScript â†’ ESM, built to `plugin/scripts/*-hook.js`) distributed as a Claude Code plugin via `${CLAUDE_PLUGIN_ROOT}`. Its hooks depend on a running worker service (Express on port 37777) and have a documented race condition at startup (issue #775 â€” context-hook fails when worker isn't ready).

claude-mem-lite eliminates these problems:
- **No worker dependency for capture**: PostToolUse writes directly to SQLite
- **No plugin system dependency**: Hooks registered in user settings with absolute interpreter paths
- **No startup race**: SessionStart doesn't need the worker; it reads SQLite directly
- **Python scripts**: No build step (TypeScriptâ†’ESM), no Bun/Node runtime

---

## 2. Architecture Decision: Distribution Approach

The implementation plan references a `plugin/` directory structure. After researching the Claude Code plugin system, there are three viable approaches:

### Option A: Plugin Distribution
```
claude-mem-lite-plugin/
â”œâ”€â”€ .claude-plugin/plugin.json
â”œâ”€â”€ hooks/hooks.json
â”œâ”€â”€ scripts/
â””â”€â”€ skills/mem-search/SKILL.md
```
**Pros**: Clean structure, installable via `/plugin install`, auto-discovered
**Cons**: `${CLAUDE_PLUGIN_ROOT}` has expansion bugs (GitHub issues #9354, #18517 â€” path not updated on version changes, variable undefined in some contexts). Plugin system still maturing. Adds metadata/manifest overhead.

### Option B: User Settings with Absolute Paths
```
~/.claude-mem/scripts/*.py
~/.claude/settings.json  â† hooks added here
```
**Pros**: Dead simple, reliable, absolute paths never break, no plugin system dependency
**Cons**: Manual setup per machine (solvable with install command), not sharable via marketplace

### Option C: Hybrid â€” pip package + install command â† **RECOMMENDED**
```
pip install claude-mem-lite   â† installs package + scripts
claude-mem install-hooks      â† writes hooks into ~/.claude/settings.json
```
**Pros**: Reliable absolute paths, automated setup, pip-native distribution, no plugin bugs, scripts live alongside the Python package
**Cons**: Not in Claude Code marketplace (irrelevant for personal tool)

### Decision: **Option C**

Rationale: This is a personal development tool, not a marketplace product. Plugin system bugs (`CLAUDE_PLUGIN_ROOT` expansion failures) have caused real issues for claude-mem. A `claude-mem install-hooks` command is equally simple to set up but dramatically more reliable. If marketplace distribution becomes desirable later, adding a plugin wrapper is trivial.

The scripts are installed as part of the pip package and invoked via the absolute Python interpreter path resolved at install time.

---

## 3. Claude Code Hooks API â€” Verified Specification

Research conducted against official docs (code.claude.com/docs/en/hooks, v2.1.x) on 2026-02-08.

### 3.1 Hook Input (stdin JSON)

All hooks receive JSON on stdin with these common fields:
```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/transcript.jsonl",
  "cwd": "/current/working/dir",
  "permission_mode": "default",
  "hook_event_name": "PostToolUse"
}
```

Event-specific additions:

| Event | Extra Fields |
|-------|-------------|
| `SessionStart` | `source`: `"startup"` \| `"resume"` \| `"clear"` \| `"compact"` |
| `PostToolUse` | `tool_name`, `tool_input` (object), `tool_response` (object), `tool_use_id` |
| `Stop` | `stop_hook_active` (bool â€” true if already continuing from a stop hook) |
| `SessionEnd` | `reason`: `"clear"` \| `"logout"` \| `"prompt_input_exit"` \| `"other"` |

### 3.2 Hook Output (stdout + exit code)

| Exit Code | Behavior |
|-----------|----------|
| `0` | Success. stdout parsed as JSON for structured control. For `SessionStart`, stdout text is injected into Claude's context. |
| `2` | Blocking error. stderr shown to Claude. |
| Other | Non-blocking error. stderr shown in verbose mode. Execution continues. |

### 3.3 Context Injection (SessionStart)

SessionStart hooks can inject context into Claude via two mechanisms:
1. **Plain text stdout** â€” added directly to Claude's context
2. **JSON stdout** with `hookSpecificOutput.additionalContext`:
```json
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "## Recent Sessions\n- [2h ago] Implemented JWT auth..."
  }
}
```

### 3.4 PostToolUse Data Availability

The `tool_input` and `tool_response` schemas vary by tool:

| Tool | `tool_input` fields | `tool_response` fields |
|------|-------------------|----------------------|
| `Write` | `file_path`, `content` | `filePath`, `success` |
| `Edit` | `file_path`, `old_string`, `new_string` | `filePath`, `success`, `linesChanged` |
| `MultiEdit` | `file_path`, `edits[]` | `filePath`, `success` |
| `Bash` | `command`, `description` | `exit_code`, stdout/stderr content |
| `Read` | `file_path`, line range opts | file content |
| `Glob` | `pattern` | matched file list |
| `Grep` | `pattern`, `path` | matched lines |
| `WebFetch` | `url` | fetched content |
| `Task` | `description` | task result |

**Critical insight**: For Write/Edit, `tool_input` contains the actual content written, which is the high-value data for compression. `tool_response` is just metadata (success/fail). For Bash, `tool_response` contains the command output. Both `tool_input` and `tool_response` must be captured.

### 3.5 Environment Variables

| Variable | Availability | Description |
|----------|-------------|-------------|
| `CLAUDE_PROJECT_DIR` | All hooks | Project root directory (stable â€” does not change on `cd`) |
| `CLAUDE_SESSION_ID` | All hooks (v2.1.9+) | Current session ID |
| `CLAUDE_ENV_FILE` | SessionStart only | File path for persisting env vars |

**v3 note**: `CLAUDE_PROJECT_DIR` is the canonical project identifier. It reflects where Claude Code was started, not the current working directory. Verified working from `~/.claude/settings.json` hooks. Known bug (#9447) affects plugin hooks only â€” not applicable to our distribution approach.

### 3.6 Known Issues

| Issue | Impact | Mitigation |
|-------|--------|-----------|
| SessionStart not firing for new conversations (GitHub #10373) | Context injection missed on first session | Graceful: system works without context injection. User can `/clear` to trigger. Reported fixed in later versions. |
| Hooks run in parallel | Multiple hooks on same event execute concurrently | No issue â€” our hooks are independent. Each has its own SQLite connection. |
| Hook timeout: 10 minutes default | Could hang if script blocks | Set explicit `timeout: 10` (seconds) on all our hooks |
| `CLAUDE_PROJECT_DIR` empty in plugin hooks (#9447) | Would break project identification | Not applicable â€” we use user settings hooks, not plugin hooks |

---

## 4. Technical Specification

### 4.1 Project Structure (additions to Phase 0)

```
claude-mem-lite/
â”œâ”€â”€ src/claude_mem_lite/
â”‚   â”œâ”€â”€ hooks/                    # NEW â€” This phase
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ capture.py            # PostToolUse handler
â”‚   â”‚   â”œâ”€â”€ context.py            # SessionStart handler
â”‚   â”‚   â”œâ”€â”€ summary.py            # Stop handler
â”‚   â”‚   â””â”€â”€ cleanup.py            # SessionEnd handler
â”‚   â”œâ”€â”€ cli/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â””â”€â”€ install.py            # NEW â€” `claude-mem install-hooks`
â”‚   â”œâ”€â”€ storage/                  # Phase 0
â”‚   â”œâ”€â”€ logging/                  # Phase 0
â”‚   â””â”€â”€ config.py                 # Phase 0
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ test_storage.py           # Phase 0
â”‚   â”œâ”€â”€ test_hooks.py             # NEW â€” This phase
â”‚   â””â”€â”€ test_install.py           # NEW â€” This phase
â””â”€â”€ pyproject.toml                # Updated with entry points
```

### 4.2 Hook Scripts

All hooks follow the same pattern:
1. Read JSON from stdin
2. Parse with json.loads (stdlib) â€” **before any heavy imports**
3. Check for early-exit conditions (no-op paths exit in <30ms)
4. Import `claude_mem_lite` modules (pays Pydantic cost only when needed)
5. Resolve project directory from `CLAUDE_PROJECT_DIR` env var (fallback: `cwd`)
6. Perform SQLite operation (direct, synchronous)
7. Write JSON to stdout (exit 0) or stderr (exit 2)
8. Exit (<200ms target for capture/context, <30ms for no-op paths)

All hooks import from the installed `claude_mem_lite` package. No standalone scripts with duplicated logic.

**v3: Fast-exit-before-import pattern**: Every hook checks for no-op conditions (malformed JSON, `stop_hook_active`, etc.) *before* importing any `claude_mem_lite` modules. This avoids paying the ~50-100ms Pydantic import cost on paths that don't need it.

#### 4.2.1 `capture.py` â€” PostToolUse Handler

**When**: After every tool execution (matcher: `*`)
**Purpose**: Write raw tool data to `pending_queue` for later compression, with priority tagging

**Matcher strategy**: Capture ALL tool uses (`*`), but tag each with a priority level. Phase 3's compression pipeline processes high-priority items first and can batch-compress or skip low-priority ones. This avoids missing data while keeping the hot path for mutations clear.

```python
#!/usr/bin/env python3
"""PostToolUse hook â€” capture tool execution data to pending_queue."""
import json
import os
import sys
import uuid
from datetime import datetime, timezone

# Tools that represent mutations or actions â€” always compress individually
HIGH_VALUE_TOOLS = {"Write", "Edit", "MultiEdit", "Bash"}

# Tools that represent information gathering â€” batch-compress or skip
LOW_VALUE_TOOLS = {"Read", "Glob", "Grep", "TodoRead", "TodoWrite"}

# TODO(phase3): Refine Bash priority by inspecting command.
# Read-only commands (ls, cat, echo, pwd, which) â†’ low
# Mutation/install commands (rm, git commit, npm install, pip install) â†’ high
# Keep all Bash as "high" in Phase 1 to avoid missing important data.

# Everything else (WebFetch, Task, etc.) defaults to "normal"


def main() -> None:
    try:
        # --- Fast path: parse JSON before any heavy imports ---
        try:
            event = json.load(sys.stdin)
        except (json.JSONDecodeError, EOFError):
            sys.exit(0)  # Non-blocking: don't break Claude Code

        session_id = event.get("session_id", "unknown")
        tool_name = event.get("tool_name", "unknown")

        # Determine priority
        if tool_name in HIGH_VALUE_TOOLS:
            priority = "high"
        elif tool_name in LOW_VALUE_TOOLS:
            priority = "low"
        else:
            priority = "normal"

        # Extract files touched from tool_input
        tool_input = event.get("tool_input", {})
        files_touched = _extract_files(tool_name, tool_input)

        # Resolve project directory (v3: CLAUDE_PROJECT_DIR preferred)
        project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd", "")

        # Build raw payload â€” both input and response are valuable
        raw_payload = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": event.get("tool_response", {}),
            "project_dir": project_dir,
        }

        # --- Heavy imports only after fast-path checks pass ---
        from claude_mem_lite.config import Config
        from claude_mem_lite.storage.sqlite_store import SQLiteStore

        config = Config()
        store = SQLiteStore(config.db_path)

        store.enqueue(
            id=str(uuid.uuid4()),
            session_id=session_id,
            tool_name=tool_name,
            raw_output=json.dumps(raw_payload, default=str),
            files_touched=json.dumps(files_touched),
            priority=priority,
        )
        store.close()

        # Silent success â€” suppress output from transcript
        json.dump({"continue": True, "suppressOutput": True}, sys.stdout)

    except Exception:
        # NEVER crash Claude Code. Swallow everything.
        sys.exit(0)


def _extract_files(tool_name: str, tool_input: dict) -> list[str]:
    """Extract file paths from tool_input based on tool type."""
    files = []
    if tool_name in ("Write", "Edit", "MultiEdit", "Read"):
        fp = tool_input.get("file_path")
        if fp:
            files.append(fp)
    elif tool_name == "Bash":
        # Can't reliably extract files from bash commands
        pass
    elif tool_name == "Glob":
        # Results are in tool_response, not tool_input
        pass
    return files


if __name__ == "__main__":
    main()
```

**Design notes**:
1. **`{"continue": True, "suppressOutput": True}`**: Prevents hook output from appearing in Claude's transcript. claude-mem uses the same pattern.
2. **Swallow all exceptions**: A hook crash would print errors to verbose mode but not break Claude Code (non-zero exit codes other than 2 are non-blocking). We swallow anyway to keep logs clean.
3. **Import at call site**: `from claude_mem_lite...` is imported inside `main()` after fast-path checks to keep startup fast. Module-level imports add latency from loading pydantic, etc.
4. **`raw_output` stores both `tool_input` and `tool_response`**: For Write/Edit, the content is in `tool_input`. For Bash, output is in `tool_response`. Capturing both ensures no data loss.
5. **No session existence check**: The session may not exist yet if SessionStart hook hasn't fired (race in parallel hooks). The `pending_queue` has a `session_id` TEXT column without a FK constraint for this reason. We reconcile in Phase 3.
6. **Priority tagging**: `high` = mutations/commands (Write, Edit, Bash), `low` = reads/listings (Read, Glob, Grep), `normal` = everything else (WebFetch, Task). Phase 3 processor drains high first, can batch-compress or skip low-priority items.
7. **v3: `CLAUDE_PROJECT_DIR`**: Used as the canonical project directory. Falls back to `cwd` from the event JSON only if the env var is absent. Stored in `raw_payload` as `project_dir` instead of `cwd`.

#### 4.2.2 `context.py` â€” SessionStart Handler

**When**: Session start (matcher: `startup|resume|clear|compact`)
**Purpose**: Create session record, optionally inject context

```python
#!/usr/bin/env python3
"""SessionStart hook â€” create session, inject context."""
import json
import os
import sys
import uuid
from datetime import datetime, timezone


def main() -> None:
    try:
        # --- Fast path: parse JSON before any heavy imports ---
        try:
            event = json.load(sys.stdin)
        except (json.JSONDecodeError, EOFError):
            sys.exit(0)

        session_id = event.get("session_id", str(uuid.uuid4()))
        source = event.get("source", "startup")

        # v3: Resolve project directory â€” CLAUDE_PROJECT_DIR is stable across cd
        project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd", "")

        # --- Heavy imports ---
        from claude_mem_lite.config import Config
        from claude_mem_lite.storage.sqlite_store import SQLiteStore

        config = Config()
        store = SQLiteStore(config.db_path)

        # Create session record (idempotent â€” ignore if exists for resume/compact)
        store.create_session(
            id=session_id,
            project_dir=project_dir,
        )

        # Phase 1: Return minimal context (session list from SQLite)
        # Phase 5: Will call worker's GET /api/context for rich injection
        context_text = _build_minimal_context(store)
        store.close()

        if context_text:
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": context_text,
                }
            }
            json.dump(output, sys.stdout)
        # Exit 0 â€” success

    except Exception:
        sys.exit(0)


def _build_minimal_context(store) -> str:
    """Build lightweight context from recent sessions. ~100-200 tokens.

    Phase 5 replaces this with full progressive disclosure from the worker.
    """
    sessions = store.list_sessions(limit=5, status_filter=None)
    if not sessions:
        return ""

    lines = ["## Recent Sessions (claude-mem-lite)"]
    for s in sessions:
        summary = s.summary or "no summary"
        status = s.status
        lines.append(f"- [{s.started_at:%Y-%m-%d %H:%M}] {summary} ({status})")

    lines.append("")
    lines.append("Use `curl` to search past work: see mem-search skill.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
```

**Design notes**:
1. **Idempotent session creation**: `resume` and `compact` sources may fire for an existing session. `create_session` should use `INSERT OR IGNORE`.
2. **Minimal context now, rich context later**: Phase 1 returns a simple session list (~100 tokens). Phase 5 wires in the full progressive disclosure builder via the worker API.
3. **Context injection mechanism**: `hookSpecificOutput.additionalContext` is the official API per Claude Code docs. Plain text on stdout also works for SessionStart.
4. **v3: `CLAUDE_PROJECT_DIR`**: Used instead of `cwd` for session's `project_dir`. This ensures `cd src` during a session doesn't create a new project identity.

#### 4.2.3 `summary.py` â€” Stop Handler

**When**: Claude finishes responding (no matcher â€” fires on every stop)
**Purpose**: Mark session as needing summary

```python
#!/usr/bin/env python3
"""Stop hook â€” mark session for summary generation."""
import json
import sys


def main() -> None:
    try:
        # --- Fast path: parse JSON and check for no-op BEFORE importing ---
        try:
            event = json.load(sys.stdin)
        except (json.JSONDecodeError, EOFError):
            sys.exit(0)

        session_id = event.get("session_id", "unknown")
        stop_hook_active = event.get("stop_hook_active", False)

        # Don't re-process if we're already in a stop hook loop.
        # v3: This exits in <30ms â€” no heavy imports paid.
        if stop_hook_active:
            sys.exit(0)

        # --- Heavy imports only when we actually need to do work ---
        from claude_mem_lite.config import Config
        from claude_mem_lite.storage.sqlite_store import SQLiteStore
        from claude_mem_lite.logging.logger import MemLogger

        config = Config()
        store = SQLiteStore(config.db_path)
        logger = MemLogger(config.log_dir, store.conn)

        # Log the stop event with observation count
        obs_count = store.count_pending(session_id)
        logger.log("hook.stop", {
            "session_id": session_id,
            "pending_observations": obs_count,
        }, session_id=session_id)

        store.close()

        # Phase 1: Just log. Phase 3 wires in POST /api/summarize to worker.
        json.dump({"continue": True, "suppressOutput": True}, sys.stdout)

    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
```

**Design notes**:
1. **`stop_hook_active` guard**: Prevents infinite loops if a Stop hook causes Claude to continue and stop again. Official docs warn about this.
2. **v3: Fast exit before import**: The `stop_hook_active` check happens *before* importing `claude_mem_lite` modules. If Claude is in a stop loop, we exit in <30ms without paying the Pydantic import cost.
3. **No `decision: "block"`**: We don't want to prevent Claude from stopping. We just observe.
4. **Phase 3 adds summarization**: The Stop hook will call `POST /api/summarize` on the worker to generate an AI session summary.

#### 4.2.4 `cleanup.py` â€” SessionEnd Handler

**When**: Session ends (exit, clear, logout)
**Purpose**: Mark session as closed, checkpoint WAL

```python
#!/usr/bin/env python3
"""SessionEnd hook â€” close session, checkpoint WAL."""
import json
import sys
from datetime import datetime, timezone


def main() -> None:
    try:
        # --- Fast path: parse JSON before heavy imports ---
        try:
            event = json.load(sys.stdin)
        except (json.JSONDecodeError, EOFError):
            sys.exit(0)

        session_id = event.get("session_id", "unknown")
        reason = event.get("reason", "other")

        # --- Heavy imports ---
        from claude_mem_lite.config import Config
        from claude_mem_lite.storage.sqlite_store import SQLiteStore
        from claude_mem_lite.logging.logger import MemLogger

        config = Config()
        store = SQLiteStore(config.db_path)
        logger = MemLogger(config.log_dir, store.conn)

        # Close session
        store.update_session(session_id, status="closed", ended_at=datetime.now(timezone.utc))

        # v3: PASSIVE checkpoint â€” non-blocking if other readers exist.
        # TRUNCATE would throw SQLITE_BUSY with concurrent readers (Phase 3 worker).
        # PASSIVE checkpoints as much as possible without blocking.
        store.checkpoint(mode="PASSIVE")

        logger.log("hook.session_end", {
            "session_id": session_id,
            "reason": reason,
        }, session_id=session_id)

        store.close()

    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
```

**Design notes**:
1. **v3: `PASSIVE` checkpoint**: Replaced `TRUNCATE` with `PASSIVE`. `TRUNCATE` blocks if any readers are active and throws `SQLITE_BUSY` in multi-process scenarios. `PASSIVE` checkpoints as much as possible without blocking â€” correct default for a system that will gain a background worker in Phase 3. `RESTART` was considered but `PASSIVE` is sufficient since we don't need to guarantee WAL size reduction on every session end.
2. **v3: `store.checkpoint(mode=)` parameter**: The `SQLiteStore.checkpoint()` method should accept a `mode` parameter (default `"PASSIVE"`) so Phase 3 can use `RESTART` or `TRUNCATE` in controlled maintenance contexts if needed.

### 4.3 Hook Registration (`cli/install.py`)

**v3 critical fix**: The installer resolves the absolute Python interpreter path at install time using `sys.executable`. This ensures hooks work regardless of how the package was installed (pip, pipx, virtualenv, system Python).

```python
"""claude-mem install-hooks â€” register hooks in ~/.claude/settings.json"""
import json
import sys
from pathlib import Path

from claude_mem_lite.config import Config

# v3: Resolve the absolute Python interpreter path.
# If installed via pipx, sys.executable points to the pipx venv's Python,
# which is the only interpreter that can see claude_mem_lite.
# Bare "python3" would fail with ModuleNotFoundError in that case.
PYTHON_PATH = sys.executable


def _build_hooks_config() -> dict:
    """Build hooks config with the resolved interpreter path."""
    return {
        "PostToolUse": [
            {
                "matcher": "*",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{PYTHON_PATH} -m claude_mem_lite.hooks.capture",
                        "timeout": 10,
                    }
                ],
            }
        ],
        "SessionStart": [
            {
                "matcher": "startup|resume|clear|compact",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{PYTHON_PATH} -m claude_mem_lite.hooks.context",
                        "timeout": 10,
                    }
                ],
            }
        ],
        "Stop": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{PYTHON_PATH} -m claude_mem_lite.hooks.summary",
                        "timeout": 10,
                    }
                ],
            }
        ],
        "SessionEnd": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{PYTHON_PATH} -m claude_mem_lite.hooks.cleanup",
                        "timeout": 10,
                    }
                ],
            }
        ],
    }


def install_hooks() -> None:
    """Add claude-mem-lite hooks to ~/.claude/settings.json."""
    settings_path = Path.home() / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    if settings_path.exists():
        # v3: Robust JSON parsing with diagnostic error on failure.
        # Claude Code uses a JSONC parser internally, but users' settings.json
        # files are standard JSON in practice. If comments are present,
        # json.load will fail â€” we detect this and give a clear error.
        try:
            with open(settings_path) as f:
                settings = json.load(f)
        except json.JSONDecodeError as e:
            raw = settings_path.read_text()
            if "//" in raw or "/*" in raw:
                print(
                    f"âœ— Cannot parse {settings_path}: file contains JSONC comments.\n"
                    f"  claude-mem-lite requires standard JSON (no // or /* */ comments).\n"
                    f"  Please remove comments from your settings.json and retry.\n"
                    f"  Parse error: {e}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"âœ— Cannot parse {settings_path}: malformed JSON.\n"
                    f"  Please fix the file and retry.\n"
                    f"  Parse error: {e}",
                    file=sys.stderr,
                )
            sys.exit(1)
    else:
        settings = {}

    hooks_config = _build_hooks_config()
    hooks = settings.setdefault("hooks", {})

    # Merge â€” don't overwrite existing hooks from other tools
    for event, matchers in hooks_config.items():
        existing = hooks.get(event, [])
        # Check if we already have a claude-mem-lite hook
        already_installed = any(
            "claude_mem_lite" in str(h.get("hooks", []))
            for h in existing
        )
        if not already_installed:
            existing.extend(matchers)
        hooks[event] = existing

    settings["hooks"] = hooks

    # v3: Atomic write â€” write to temp file then rename to prevent corruption
    tmp_path = settings_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(settings, f, indent=2)
    tmp_path.rename(settings_path)

    print(f"âœ“ Hooks installed in {settings_path}")
    print(f"  Python interpreter: {PYTHON_PATH}")
    print("  Restart Claude Code for hooks to take effect.")


def uninstall_hooks() -> None:
    """Remove claude-mem-lite hooks from ~/.claude/settings.json."""
    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        print("No settings file found.")
        return

    try:
        with open(settings_path) as f:
            settings = json.load(f)
    except json.JSONDecodeError as e:
        print(
            f"âœ— Cannot parse {settings_path}: {e}\n"
            f"  Cannot safely remove hooks from a malformed settings file.",
            file=sys.stderr,
        )
        sys.exit(1)

    hooks = settings.get("hooks", {})
    for event in list(hooks.keys()):
        hooks[event] = [
            m for m in hooks[event]
            if "claude_mem_lite" not in str(m.get("hooks", []))
        ]
        if not hooks[event]:
            del hooks[event]

    tmp_path = settings_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(settings, f, indent=2)
    tmp_path.rename(settings_path)

    print(f"âœ“ Hooks removed from {settings_path}")


def main():
    """CLI dispatcher for claude-mem subcommands."""
    if len(sys.argv) < 2:
        print("Usage: claude-mem <command>")
        print("  install-hooks    Register hooks in ~/.claude/settings.json")
        print("  uninstall-hooks  Remove hooks from ~/.claude/settings.json")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "install-hooks":
        install_hooks()
    elif cmd == "uninstall-hooks":
        uninstall_hooks()
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)
```

**Design notes**:
1. **v3: `sys.executable` absolute path**: The critical fix. When `install.py` runs (via `claude-mem install-hooks`), `sys.executable` resolves to the interpreter running the install â€” e.g., `/home/user/.local/pipx/venvs/claude-mem-lite/bin/python`. This is the *only* interpreter guaranteed to find `claude_mem_lite`. Bare `python3` would fail for pipx, conda, and most virtualenv installs.
2. **v3: JSONC detection and error message**: If `json.load` fails, we check for `//` or `/*` in the raw file to give a specific diagnostic. We do NOT attempt regex comment stripping â€” that would risk corrupting the file.
3. **v3: Atomic write**: Settings are written to a `.tmp` file first, then renamed. This prevents corruption if the process is killed mid-write.
4. **Merge, don't overwrite**: Other tools may have hooks registered. We append our hooks and check for duplicates.
5. **10-second timeout**: Our hooks target <200ms, but 10s gives generous headroom for slow disks or cold imports. The default 10-minute timeout is absurdly long for us.
6. **Idempotent**: Running `install-hooks` twice doesn't duplicate hooks.

### 4.4 Schema Consideration: pending_queue FK Relaxation

Phase 0 PRD defines `pending_queue.session_id` with `REFERENCES sessions(id)`. However, Claude Code hook events fire in parallel. PostToolUse may fire before SessionStart completes, meaning the session record doesn't exist yet.

**Decision**: Remove the FK constraint on `pending_queue.session_id`. This table is a transient queue â€” data integrity comes from the processing pipeline (Phase 3), not from referential constraints on a race-prone write path.

This requires Phase 0 schema adjustments:
```sql
-- Before (Phase 0)
session_id TEXT NOT NULL REFERENCES sessions(id),

-- After (Phase 1 correction)
session_id TEXT NOT NULL,  -- No FK â€” PostToolUse may fire before SessionStart
```

Add `priority` column for Phase 3 queue processing:
```sql
-- Addition to pending_queue
priority TEXT NOT NULL DEFAULT 'normal',  -- high | normal | low
```

Update the `idx_queue_status` index to include priority for efficient drain ordering:
```sql
-- Before (Phase 0)
CREATE INDEX idx_queue_status ON pending_queue(status, created_at);

-- After (Phase 1 correction)
CREATE INDEX idx_queue_status ON pending_queue(status, priority, created_at);
```

Similarly, `create_session` must use `INSERT OR IGNORE` since SessionStart may fire multiple times (resume, compact) for the same session_id.

### 4.5 Phase 0 Storage Layer: `checkpoint()` Method Update

**v3**: The `SQLiteStore.checkpoint()` method must accept a `mode` parameter:

```python
def checkpoint(self, mode: str = "PASSIVE") -> None:
    """Checkpoint WAL file.

    Modes:
        PASSIVE â€” checkpoint without blocking. Safe with concurrent readers.
        RESTART â€” block new readers, wait for existing ones, then checkpoint.
        TRUNCATE â€” like RESTART, but also truncates the WAL file.

    Phase 1 uses PASSIVE (default). Phase 3+ may use RESTART for maintenance.
    """
    assert mode in ("PASSIVE", "RESTART", "TRUNCATE"), f"Invalid mode: {mode}"
    self.conn.execute(f"PRAGMA wal_checkpoint({mode})")
```

### 4.6 pyproject.toml Updates

```toml
[project.scripts]
claude-mem = "claude_mem_lite.cli.install:main"

# Or more granular:
# claude-mem-install-hooks = "claude_mem_lite.cli.install:install_hooks"
# claude-mem-uninstall-hooks = "claude_mem_lite.cli.install:uninstall_hooks"
```

The `main()` function dispatches subcommands: `claude-mem install-hooks`, `claude-mem uninstall-hooks`, etc.

---

## 5. Data Flow

### 5.1 PostToolUse (capture) â€” Most frequent path

```
Claude executes tool (Write, Edit, Bash, etc.)
    â”‚
    â–¼
Claude Code fires PostToolUse hook
    â”‚
    â–¼
capture.py reads JSON from stdin (stdlib json â€” fast)
    â”‚  {session_id, tool_name, tool_input, tool_response, ...}
    â”‚
    â–¼
Resolve project_dir from CLAUDE_PROJECT_DIR env var (fallback: cwd)
    â”‚
    â–¼
Import claude_mem_lite modules (pays ~50-100ms Pydantic cost)
    â”‚
    â–¼
capture.py opens SQLite connection (direct, sqlite3 stdlib)
    â”‚
    â–¼
INSERT INTO pending_queue (id, session_id, tool_name, raw_output, files_touched, priority, status='raw')
    â”‚  raw_output = JSON.dumps({tool_name, tool_input, tool_response, project_dir})
    â”‚
    â–¼
stdout: {"continue": true, "suppressOutput": true}
exit 0

Total latency: <200ms (Python startup + Pydantic import + SQLite INSERT)
  - Of which ~5ms is the actual SQLite INSERT on local SSD with WAL mode
  - Remainder is Python interpreter startup and module imports
```

### 5.2 SessionStart (context)

```
User starts Claude Code or runs /clear
    â”‚
    â–¼
Claude Code fires SessionStart hook
    â”‚
    â–¼
context.py reads JSON from stdin
    â”‚  {session_id, source, cwd, ...}
    â”‚
    â–¼
Resolve project_dir from CLAUDE_PROJECT_DIR env var
    â”‚
    â–¼
Import claude_mem_lite modules
    â”‚
    â–¼
INSERT OR IGNORE INTO sessions (id, project_dir, started_at, status='active')
    â”‚
    â–¼
SELECT recent sessions for minimal context
    â”‚
    â–¼
stdout: {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "..."}}
exit 0

Total latency: <200ms (Python startup + imports + SQLite reads + JSON formatting)
Context injected: ~100-200 tokens (session list)
```

### 5.3 Stop (summary marker)

```
Claude finishes responding
    â”‚
    â–¼
Claude Code fires Stop hook
    â”‚
    â–¼
summary.py reads JSON from stdin
    â”‚  Checks stop_hook_active â€” if true, exits immediately (<30ms, no imports)
    â”‚
    â–¼
Import claude_mem_lite modules (only if not a no-op)
    â”‚
    â–¼
Log event: hook.stop {session_id, pending_observations count}
    â”‚
    â–¼
stdout: {"continue": true, "suppressOutput": true}
exit 0

Total latency: <30ms (no-op path) / <200ms (normal path)
```

### 5.4 SessionEnd (cleanup)

```
User exits Claude Code / runs /clear
    â”‚
    â–¼
Claude Code fires SessionEnd hook
    â”‚
    â–¼
cleanup.py reads JSON from stdin
    â”‚
    â–¼
Import claude_mem_lite modules
    â”‚
    â–¼
UPDATE sessions SET status='closed', ended_at=now() WHERE id=?
PRAGMA wal_checkpoint(PASSIVE)
Log event: hook.session_end
    â”‚
    â–¼
exit 0

Total latency: <200ms
```

---

## 6. Corrections to Implementation Plan

| Item | Implementation Plan | Corrected (Phase 1 PRD) | Rationale |
|------|-------------------|------------------------|-----------|
| Distribution | `plugin/hooks.json` as Claude Code plugin | `~/.claude/settings.json` via `claude-mem install-hooks` | `${CLAUDE_PLUGIN_ROOT}` has expansion bugs (#9354, #18517). Absolute paths are reliable. |
| Hook invocation | `python3 -m claude_mem_lite.hooks.capture` | `{sys.executable} -m claude_mem_lite.hooks.capture` | v3: Bare `python3` fails for pipx/venv installs. Absolute interpreter path resolved at install time. |
| Project directory | `cwd` from hook JSON input | `CLAUDE_PROJECT_DIR` env var with `cwd` fallback | v3: `cwd` changes on `cd`. Env var is stable project root. |
| PostToolUse data | "raw tool output" | `tool_input` + `tool_response` JSON | Hook receives structured fields, not raw stdout. Both must be captured. |
| Hook names | `capture-hook.py`, `context-hook.py`, etc. | `capture.py`, `context.py`, etc. | Cleaner as Python modules (importable, testable) |
| Context injection | "returns empty string" | Returns minimal session list JSON via `hookSpecificOutput.additionalContext` | Even Phase 1 should inject *something* useful |
| pending_queue FK | `REFERENCES sessions(id)` | No FK constraint | Hooks fire in parallel â€” PostToolUse may fire before SessionStart |
| Stop hook behavior | "marks session needs summary" | Logs event + pending count. No session status change. | Session isn't done when Stop fires â€” Stop fires on every response, not just task completion |
| SessionStart matcher | Not specified | `startup\|resume\|clear\|compact` | Must handle all session start sources |
| Hook latency target | "<10ms for capture, <5ms for cleanup" | <200ms capture/cleanup, <30ms no-op | v3: Python cold start + Pydantic import = ~100-150ms minimum. 10ms was impossible. |
| WAL checkpoint | `PRAGMA wal_checkpoint(TRUNCATE)` | `PRAGMA wal_checkpoint(PASSIVE)` | v3: TRUNCATE blocks on active readers. PASSIVE is non-blocking. |
| PostToolUse filtering | "No AI, no worker yet" (capture all equally) | Capture all with priority tagging (`high`/`normal`/`low`) | Phase 3 drains high-priority first. Avoids missing data while managing noise from Reads. |
| pending_queue schema | `session_id REFERENCES sessions(id)` | No FK + add `priority TEXT` column + updated index | Parallel hooks + priority-ordered processing |

---

## 7. Dependencies

### 7.1 Phase 1 runtime dependencies (additions to Phase 0)

None. All hook scripts use:
- `json`, `sys`, `os`, `uuid`, `datetime` â€” stdlib
- `sqlite3` â€” stdlib (via SQLiteStore from Phase 0)
- `pydantic` â€” already installed (Phase 0)

### 7.2 Phase 1 dev dependencies (additions to Phase 0)

None beyond pytest and ruff (already in Phase 0).

---

## 8. Test Plan

### 8.1 Test categories

| Category | Tests | What it validates |
|----------|-------|-------------------|
| **capture.py** | 6 | Parses PostToolUse stdin, extracts files from Write/Edit/Bash, writes to pending_queue, handles malformed JSON, handles missing fields gracefully, uses `CLAUDE_PROJECT_DIR` over `cwd` |
| **context.py** | 5 | Creates session on startup, idempotent on resume, builds minimal context, handles empty DB, uses `CLAUDE_PROJECT_DIR` for project_dir |
| **summary.py** | 4 | Logs stop event, respects stop_hook_active guard (fast exit), handles missing session, exits <30ms on no-op |
| **cleanup.py** | 4 | Closes session, checkpoints WAL with PASSIVE, handles unknown session_id, no error on concurrent readers |
| **install.py** | 7 | Fresh install, idempotent re-install, merge with existing hooks, uninstall, handles missing settings file, rejects JSONC with diagnostic error, rejects malformed JSON |
| **Integration** | 3 | Full lifecycle (startâ†’captureâ†’stopâ†’end), parallel capture writes (10 threads), pending_queue data integrity |
| **Total** | **29** | |

### 8.2 Test infrastructure

```python
# conftest.py additions
import json
import os
from io import StringIO
from unittest.mock import patch

@pytest.fixture
def mock_posttooluse_event():
    """Simulated PostToolUse hook input."""
    return {
        "session_id": "test-session-001",
        "transcript_path": "/tmp/test-transcript.jsonl",
        "cwd": "/home/user/project",
        "permission_mode": "default",
        "hook_event_name": "PostToolUse",
        "tool_name": "Write",
        "tool_input": {
            "file_path": "/home/user/project/src/auth.py",
            "content": "def authenticate(email, password): ...",
        },
        "tool_response": {
            "filePath": "/home/user/project/src/auth.py",
            "success": True,
        },
        "tool_use_id": "toolu_01ABC123",
    }

@pytest.fixture
def mock_sessionstart_event():
    return {
        "session_id": "test-session-001",
        "transcript_path": "/tmp/test-transcript.jsonl",
        "cwd": "/home/user/project",
        "permission_mode": "default",
        "hook_event_name": "SessionStart",
        "source": "startup",
    }

@pytest.fixture
def mock_claude_project_dir():
    """Set CLAUDE_PROJECT_DIR env var for testing."""
    with patch.dict(os.environ, {"CLAUDE_PROJECT_DIR": "/home/user/project"}):
        yield "/home/user/project"
```

### 8.3 Testing hooks as functions

Hook scripts are invoked by Claude Code via stdin/stdout, but for testing we import and call the internal logic directly:

```python
def test_capture_writes_to_queue(store, mock_posttooluse_event, mock_claude_project_dir):
    """capture.py should INSERT into pending_queue."""
    from claude_mem_lite.hooks.capture import _process_event

    _process_event(mock_posttooluse_event, store)

    items = store.list_pending(status="raw")
    assert len(items) == 1
    assert items[0].tool_name == "Write"
    assert items[0].session_id == "test-session-001"

    payload = json.loads(items[0].raw_output)
    assert payload["tool_input"]["file_path"] == "/home/user/project/src/auth.py"
    assert payload["project_dir"] == "/home/user/project"


def test_capture_prefers_claude_project_dir(store, mock_posttooluse_event):
    """capture.py should use CLAUDE_PROJECT_DIR over cwd."""
    with patch.dict(os.environ, {"CLAUDE_PROJECT_DIR": "/canonical/project/root"}):
        from claude_mem_lite.hooks.capture import _process_event
        _process_event(mock_posttooluse_event, store)

    payload = json.loads(store.list_pending(status="raw")[0].raw_output)
    assert payload["project_dir"] == "/canonical/project/root"


def test_capture_falls_back_to_cwd(store, mock_posttooluse_event):
    """capture.py should fall back to cwd if CLAUDE_PROJECT_DIR is not set."""
    with patch.dict(os.environ, {}, clear=True):
        from claude_mem_lite.hooks.capture import _process_event
        _process_event(mock_posttooluse_event, store)

    payload = json.loads(store.list_pending(status="raw")[0].raw_output)
    assert payload["project_dir"] == "/home/user/project"


def test_install_rejects_jsonc(tmp_path):
    """install.py should fail with diagnostic error on JSONC files."""
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text('{\n  // this is a comment\n  "hooks": {}\n}')

    with patch("claude_mem_lite.cli.install.Path.home", return_value=tmp_path):
        with pytest.raises(SystemExit, match="1"):
            install_hooks()


def test_summary_fast_exit_on_stop_hook_active(store):
    """summary.py should exit before importing heavy modules when stop_hook_active=True."""
    import time

    event = {"session_id": "test", "stop_hook_active": True}
    start = time.monotonic()
    # Simulate the fast path â€” should not import claude_mem_lite
    from claude_mem_lite.hooks.summary import main as summary_main
    # ... (test the early exit path)
    elapsed = time.monotonic() - start
    # Fast path should be well under 100ms even with test overhead
    assert elapsed < 0.1


def test_cleanup_uses_passive_checkpoint(store):
    """cleanup.py should use PASSIVE WAL checkpoint."""
    # Verify the store.checkpoint method is called with mode="PASSIVE"
    with patch.object(store, "checkpoint") as mock_checkpoint:
        from claude_mem_lite.hooks.cleanup import _process_event
        _process_event({"session_id": "test", "reason": "other"}, store)
        mock_checkpoint.assert_called_once_with(mode="PASSIVE")
```

Each hook script exposes a `_process_event(event: dict, store: SQLiteStore)` function for testability, with the `main()` function handling stdin/stdout plumbing.

### 8.4 Performance targets

| Operation | Target | Rationale |
|-----------|--------|-----------|
| capture.py end-to-end | <200ms | Python startup (~15-30ms) + Pydantic import (~50-80ms) + module init + SQLite INSERT (~5ms) |
| context.py end-to-end | <200ms | Same startup cost + SQLite reads + JSON formatting |
| cleanup.py end-to-end | <200ms | Same startup cost + SQLite UPDATE + WAL PASSIVE checkpoint |
| summary.py no-op (stop_hook_active) | <30ms | Python startup + JSON parse + exit. No heavy imports. |
| summary.py normal | <200ms | Full startup + SQLite count + log write |
| 10 concurrent captures | 0 errors | WAL mode concurrent writes |
| install-hooks parse failure | Clear error message | Identifies JSONC vs malformed JSON |

**Note on cold vs warm**: First invocation of each hook pays Python startup + Pydantic import cost (~100-150ms). Subsequent invocations within the same session still pay this cost (each hook invocation is a new process), but OS filesystem cache makes module loading faster (~80-120ms). If cold start proves problematic, we can add a daemon mode in a later phase that keeps a process resident.

---

## 9. Acceptance Criteria

Phase 1 is complete when:

- [ ] All 29 tests pass (pytest, <15s total runtime)
- [ ] `ruff check` and `ruff format --check` pass with zero warnings
- [ ] `claude-mem install-hooks` writes correct hooks config with absolute Python path to `~/.claude/settings.json`
- [ ] `claude-mem install-hooks` fails gracefully on JSONC or malformed settings files
- [ ] `claude-mem uninstall-hooks` removes only claude-mem-lite hooks
- [ ] PostToolUse hook captures tool data to `pending_queue` (verified in real Claude Code session)
- [ ] All hooks use `CLAUDE_PROJECT_DIR` env var with `cwd` fallback
- [ ] SessionStart hook creates session record and returns context JSON
- [ ] Stop hook logs event without blocking Claude
- [ ] Stop hook exits in <30ms when `stop_hook_active` is true
- [ ] SessionEnd hook closes session and checkpoints WAL with PASSIVE mode
- [ ] Hooks survive malformed input (empty stdin, invalid JSON, missing fields)
- [ ] No new runtime dependencies beyond Phase 0

---

## 10. Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Python cold start latency (100-150ms) | High | Low | Acceptable for hooks (10s timeout, <200ms actual). Not in Claude's critical path. If problematic, add daemon mode in later phase. |
| SessionStart not firing (Claude Code bug #10373) | Low (reported fixed) | Medium | System works without context injection. First prompt still works. |
| Hooks fire in parallel â€” PostToolUse before SessionStart | Medium | Low | `pending_queue` has no FK on `session_id`. Reconcile in Phase 3 processor. |
| Interpreter path changes after package upgrade | Low | Medium | User re-runs `claude-mem install-hooks` after upgrading. Post-install hook can automate this. Document in README. |
| Large tool_input content (Write with big files) | High (by design) | Low | SQLite handles TEXT columns up to 1GB. `raw_output` size bounded by tool_input size. 30-day retention policy (Phase 9). |
| Settings.json has JSONC comments | Low | Medium | Installer detects and reports the issue with a specific error message. Does not attempt to strip comments. |
| Settings.json corruption on concurrent write | Very Low | High | Hook install does atomic write (write to temp + rename). Settings.json is rarely modified. |
| `CLAUDE_PROJECT_DIR` not set in edge cases | Very Low | Low | Falls back to `cwd` from event JSON. Logged for debugging. |

---

## 11. Resolved Design Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| **Hook invocation method** | `{sys.executable} -m claude_mem_lite.hooks.capture` | v3: Absolute interpreter path from `sys.executable`, resolved at install time. Bare `python3` fails for pipx, conda, and most virtualenv installs. `-m` invocation finds the correct package within the resolved interpreter's environment. |
| **Project directory source** | `CLAUDE_PROJECT_DIR` env var, `cwd` fallback | v3: `cwd` changes when Claude runs `cd`. `CLAUDE_PROJECT_DIR` is stable â€” set to where Claude Code was started. Verified working from `~/.claude/settings.json` hooks. |
| **WAL checkpoint mode** | `PASSIVE` | v3: `TRUNCATE` blocks on active readers (`SQLITE_BUSY`). `PASSIVE` checkpoints non-blockingly. Future-proof for Phase 3 worker that will be a persistent reader. |
| **Latency targets** | <200ms normal, <30ms no-op | v3: Python startup + Pydantic import = ~100-150ms minimum. Previous <10ms target was impossible with this architecture. Fast-exit-before-import pattern enables <30ms no-op paths. |
| **Settings.json parsing** | `json.load` + `try/except` + diagnostic error | v3: Claude Code internally uses JSONC parser, but users' files are standard JSON in practice. If comments present, we abort with a clear message. No regex comment stripping â€” too fragile. |
| **PostToolUse matcher** | `*` with priority tagging (hybrid) | Capture everything, never miss data. Tag `high` (Write, Edit, MultiEdit, Bash), `low` (Read, Glob, Grep, TodoRead, TodoWrite), `normal` (everything else). Phase 3 processor drains high-priority first, can batch-compress or skip low-priority items. |
| **Stop hook behavior** | Log only â€” no session mutations | Stop fires on every Claude response, not just task completion. `event_log` captures turn counts derivably (`SELECT COUNT(*) WHERE event_type='hook.stop'`). Adding counters is premature â€” if Phase 5/7 needs quick stats, compute at SessionEnd (accurate, no races, no hot-path cost). |
| **Bash priority refinement** | Deferred to Phase 3 | All Bash tagged `high` in Phase 1. Phase 3 worker can inspect the command string: read-only commands (ls, cat, echo) â†’ low, mutation commands (rm, git, npm install) â†’ high. Avoids complicating the hook with command parsing. |

### Phase 0 Schema Changes Required

This PRD requires three changes to the Phase 0 `pending_queue` table before implementation:

1. **Remove FK constraint**: `session_id TEXT NOT NULL` (no `REFERENCES sessions(id)`) â€” hooks fire in parallel, PostToolUse may precede SessionStart
2. **Add priority column**: `priority TEXT NOT NULL DEFAULT 'normal'` â€” values: `high`, `normal`, `low`
3. **Update index**: `CREATE INDEX idx_queue_status ON pending_queue(status, priority, created_at)` â€” enables efficient priority-ordered drain

### Phase 0 Storage Layer Change Required

4. **`checkpoint()` method**: Must accept `mode` parameter (default `"PASSIVE"`) â€” see Section 4.5
