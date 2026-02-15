"""AST Tracker — deterministic Python code intelligence.

Public API:
    scan_file(file_path, source=None) -> FileSnapshot
    scan_files(file_paths) -> list[FileSnapshot]
    diff_file(current, previous) -> list[FunctionDiff]
"""

from __future__ import annotations

from pathlib import Path

from claude_mem_lite.ast_tracker.diff import compare_snapshots
from claude_mem_lite.ast_tracker.extractor import FunctionExtractor
from claude_mem_lite.ast_tracker.types import (
    CallInfo,
    FileSnapshot,
    FunctionDiff,
    FunctionInfo,
)


def scan_file(file_path: str, source: str | None = None) -> FileSnapshot:
    """Scan a single Python file. Reads from disk if source not provided."""
    if source is None:
        with Path(file_path).open(encoding="utf-8", errors="replace") as f:
            source = f.read()
    return FunctionExtractor(source, file_path).extract()


def scan_files(file_paths: list[str]) -> list[FileSnapshot]:
    """Scan multiple Python files. Skips non-.py files silently."""
    return [scan_file(fp) for fp in file_paths if fp.endswith(".py")]


def diff_file(
    current: FileSnapshot,
    previous: list[FunctionInfo],
) -> list[FunctionDiff]:
    """Compare current snapshot against previous. Wrapper around compare_snapshots."""
    return compare_snapshots(current, previous)


__all__ = [
    "CallInfo",
    "FileSnapshot",
    "FunctionDiff",
    "FunctionInfo",
    "diff_file",
    "scan_file",
    "scan_files",
]
