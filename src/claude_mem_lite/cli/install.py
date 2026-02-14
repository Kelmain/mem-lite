"""claude-mem install-hooks -- register hooks in ~/.claude/settings.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Resolve absolute Python interpreter path at import time.
# For pipx/venv installs, sys.executable is the only interpreter that can see claude_mem_lite.
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
        try:
            with settings_path.open() as f:
                settings = json.load(f)
        except json.JSONDecodeError as e:
            raw = settings_path.read_text()
            if "//" in raw or "/*" in raw:
                msg = (
                    f"Cannot parse {settings_path}: file contains JSONC comments.\n"
                    f"  claude-mem-lite requires standard JSON (no // or /* */ comments).\n"
                    f"  Please remove comments from your settings.json and retry.\n"
                    f"  Parse error: {e}"
                )
            else:
                msg = (
                    f"Cannot parse {settings_path}: malformed JSON.\n"
                    f"  Please fix the file and retry.\n"
                    f"  Parse error: {e}"
                )
            print(msg, file=sys.stderr)
            sys.exit(1)
    else:
        settings = {}

    hooks_config = _build_hooks_config()
    hooks = settings.setdefault("hooks", {})

    # Merge -- don't overwrite existing hooks from other tools
    for event, matchers in hooks_config.items():
        existing = hooks.get(event, [])
        # Check if we already have a claude-mem-lite hook
        already_installed = any("claude_mem_lite" in str(h.get("hooks", [])) for h in existing)
        if not already_installed:
            existing.extend(matchers)
        hooks[event] = existing

    settings["hooks"] = hooks

    # Atomic write -- write to temp file then rename
    tmp_path = settings_path.with_suffix(".tmp")
    with tmp_path.open("w") as f:
        json.dump(settings, f, indent=2)
    tmp_path.rename(settings_path)

    print(f"Hooks installed in {settings_path}")
    print(f"  Python interpreter: {PYTHON_PATH}")
    print("  Restart Claude Code for hooks to take effect.")


def uninstall_hooks() -> None:
    """Remove claude-mem-lite hooks from ~/.claude/settings.json."""
    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        print("No settings file found.")
        return

    try:
        with settings_path.open() as f:
            settings = json.load(f)
    except json.JSONDecodeError as e:
        msg = (
            f"Cannot parse {settings_path}: {e}\n"
            f"  Cannot safely remove hooks from a malformed settings file."
        )
        print(msg, file=sys.stderr)
        sys.exit(1)

    hooks = settings.get("hooks", {})
    for event in list(hooks.keys()):
        hooks[event] = [m for m in hooks[event] if "claude_mem_lite" not in str(m.get("hooks", []))]
        if not hooks[event]:
            del hooks[event]

    tmp_path = settings_path.with_suffix(".tmp")
    with tmp_path.open("w") as f:
        json.dump(settings, f, indent=2)
    tmp_path.rename(settings_path)

    print(f"Hooks removed from {settings_path}")


def main() -> None:
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
