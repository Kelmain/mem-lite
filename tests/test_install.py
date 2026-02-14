"""Tests for Phase 1 hook installer -- 7 tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_mem_lite.cli.install import install_hooks, uninstall_hooks


# ---------------------------------------------------------------------------
# Install / Uninstall hooks -- 7 tests
# ---------------------------------------------------------------------------
class TestInstallHooks:
    def test_fresh_install_creates_settings(self, tmp_path):
        """No existing settings.json -- install_hooks creates one with all 4 hook events."""
        with patch.object(Path, "home", return_value=tmp_path):
            install_hooks()

        settings_path = tmp_path / ".claude" / "settings.json"
        assert settings_path.exists()

        settings = json.loads(settings_path.read_text())
        hooks = settings["hooks"]

        # All 4 event types are present
        assert "PostToolUse" in hooks
        assert "SessionStart" in hooks
        assert "Stop" in hooks
        assert "SessionEnd" in hooks

        # PostToolUse has matcher "*" and correct command pattern
        post_tool = hooks["PostToolUse"]
        assert len(post_tool) == 1
        assert post_tool[0]["matcher"] == "*"
        cmd = post_tool[0]["hooks"][0]["command"]
        assert sys.executable in cmd
        assert "claude_mem_lite.hooks.capture" in cmd
        assert post_tool[0]["hooks"][0]["timeout"] == 10

        # SessionStart has the correct matcher
        session_start = hooks["SessionStart"]
        assert len(session_start) == 1
        assert session_start[0]["matcher"] == "startup|resume|clear|compact"
        cmd = session_start[0]["hooks"][0]["command"]
        assert sys.executable in cmd
        assert "claude_mem_lite.hooks.context" in cmd
        assert session_start[0]["hooks"][0]["timeout"] == 10

        # Stop hook
        stop = hooks["Stop"]
        assert len(stop) == 1
        cmd = stop[0]["hooks"][0]["command"]
        assert sys.executable in cmd
        assert "claude_mem_lite.hooks.summary" in cmd
        assert stop[0]["hooks"][0]["timeout"] == 10

        # SessionEnd hook
        session_end = hooks["SessionEnd"]
        assert len(session_end) == 1
        cmd = session_end[0]["hooks"][0]["command"]
        assert sys.executable in cmd
        assert "claude_mem_lite.hooks.cleanup" in cmd
        assert session_end[0]["hooks"][0]["timeout"] == 10

    def test_idempotent_install(self, tmp_path):
        """Running install_hooks twice does not duplicate hook entries."""
        with patch.object(Path, "home", return_value=tmp_path):
            install_hooks()
            install_hooks()

        settings_path = tmp_path / ".claude" / "settings.json"
        settings = json.loads(settings_path.read_text())
        hooks = settings["hooks"]

        # Each event type should have exactly 1 matcher entry for claude-mem-lite
        for event in ("PostToolUse", "SessionStart", "Stop", "SessionEnd"):
            matchers = hooks[event]
            mem_lite_entries = [m for m in matchers if "claude_mem_lite" in str(m.get("hooks", []))]
            assert len(mem_lite_entries) == 1, (
                f"{event} has {len(mem_lite_entries)} claude-mem-lite entries after double install"
            )

    def test_merge_with_existing_hooks(self, tmp_path):
        """install_hooks preserves existing hooks from other tools."""
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)

        existing_settings = {
            "hooks": {
                "PostToolUse": [
                    {
                        "matcher": "Write",
                        "hooks": [{"type": "command", "command": "echo other-tool", "timeout": 5}],
                    }
                ]
            }
        }
        settings_path.write_text(json.dumps(existing_settings, indent=2))

        with patch.object(Path, "home", return_value=tmp_path):
            install_hooks()

        settings = json.loads(settings_path.read_text())
        post_tool = settings["hooks"]["PostToolUse"]

        # The existing "other-tool" hook is still present
        other_hooks = [m for m in post_tool if "other-tool" in str(m.get("hooks", []))]
        assert len(other_hooks) == 1

        # The claude-mem-lite hook was added alongside it
        mem_hooks = [m for m in post_tool if "claude_mem_lite" in str(m.get("hooks", []))]
        assert len(mem_hooks) == 1

        # Total PostToolUse matchers: 2 (other-tool + claude-mem-lite)
        assert len(post_tool) == 2

    def test_uninstall_removes_only_ours(self, tmp_path):
        """uninstall_hooks removes claude-mem-lite hooks but keeps others."""
        # First, install our hooks
        with patch.object(Path, "home", return_value=tmp_path):
            install_hooks()

        # Manually add a hook from another tool to PostToolUse
        settings_path = tmp_path / ".claude" / "settings.json"
        settings = json.loads(settings_path.read_text())
        settings["hooks"]["PostToolUse"].append(
            {
                "matcher": "Write",
                "hooks": [{"type": "command", "command": "echo other-tool", "timeout": 5}],
            }
        )
        settings_path.write_text(json.dumps(settings, indent=2))

        # Uninstall
        with patch.object(Path, "home", return_value=tmp_path):
            uninstall_hooks()

        settings = json.loads(settings_path.read_text())
        hooks = settings.get("hooks", {})

        # claude-mem-lite hooks are gone from all event types
        for event in ("SessionStart", "Stop", "SessionEnd"):
            assert event not in hooks, f"{event} should be removed (was only claude-mem-lite)"

        # PostToolUse still has the other-tool entry
        assert "PostToolUse" in hooks
        post_tool = hooks["PostToolUse"]
        assert len(post_tool) == 1
        assert "other-tool" in str(post_tool[0]["hooks"])

        # No claude-mem-lite hooks remain
        mem_hooks = [m for m in post_tool if "claude_mem_lite" in str(m.get("hooks", []))]
        assert len(mem_hooks) == 0

    def test_rejects_jsonc_with_diagnostic(self, tmp_path, capsys):
        """install_hooks fails with SystemExit(1) and diagnostic on JSONC comments."""
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text('{\n  // this is a comment\n  "hooks": {}\n}')

        with (
            patch.object(Path, "home", return_value=tmp_path),
            pytest.raises(SystemExit) as exc_info,
        ):
            install_hooks()

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "JSONC" in captured.err or "comments" in captured.err

    def test_rejects_malformed_json(self, tmp_path, capsys):
        """install_hooks fails with SystemExit(1) and error on malformed JSON."""
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("{broken")

        with (
            patch.object(Path, "home", return_value=tmp_path),
            pytest.raises(SystemExit) as exc_info,
        ):
            install_hooks()

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "malformed JSON" in captured.err or "malformed" in captured.err

    def test_uninstall_no_settings_file(self, tmp_path, capsys):
        """uninstall_hooks prints message and does not raise when no settings.json exists."""
        with patch.object(Path, "home", return_value=tmp_path):
            # Should not raise
            uninstall_hooks()

        captured = capsys.readouterr()
        assert "No settings file found" in captured.out
