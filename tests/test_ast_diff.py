"""Tests for Phase 2: AST Tracker diff module -- 7 tests."""

from __future__ import annotations

from claude_mem_lite.ast_tracker.types import FileSnapshot, FunctionDiff, FunctionInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_func(
    qualified_name: str = "my_func",
    kind: str = "function",
    body_hash: str = "abc123",
    *,
    parent_class: str | None = None,
    signature: str = "my_func() -> None",
    line_start: int = 1,
    line_end: int = 5,
) -> FunctionInfo:
    """Create a FunctionInfo with sensible defaults for testing."""
    return FunctionInfo(
        qualified_name=qualified_name,
        kind=kind,
        parent_class=parent_class,
        signature=signature,
        decorators=[],
        docstring=None,
        line_start=line_start,
        line_end=line_end,
        body_hash=body_hash,
    )


def _make_snapshot(
    functions: list[FunctionInfo] | None = None,
    file_path: str = "/test/example.py",
) -> FileSnapshot:
    """Create a FileSnapshot with sensible defaults for testing."""
    return FileSnapshot(
        file_path=file_path,
        functions=functions or [],
        import_map={},
    )


def _diff_by_name(diffs: list[FunctionDiff]) -> dict[str, FunctionDiff]:
    """Index a list of FunctionDiff by qualified_name for easy assertion."""
    return {d.qualified_name: d for d in diffs}


# ---------------------------------------------------------------------------
# Diff: change types (5 tests)
# ---------------------------------------------------------------------------


class TestDiffChangeTypes:
    """Test that compare_snapshots correctly classifies changes."""

    def test_new_function(self):
        """Function in current but not in previous is classified as 'new'."""
        from claude_mem_lite.ast_tracker.diff import compare_snapshots

        func = _make_func(qualified_name="new_func", body_hash="hash1")
        current = _make_snapshot(functions=[func])
        previous: list[FunctionInfo] = []

        result = compare_snapshots(current, previous)

        assert len(result) == 1
        diff = result[0]
        assert diff.qualified_name == "new_func"
        assert diff.change_type == "new"
        assert diff.current is func
        assert diff.previous_hash is None

    def test_modified_function(self):
        """Function in both with different body_hash is classified as 'modified'."""
        from claude_mem_lite.ast_tracker.diff import compare_snapshots

        current_func = _make_func(qualified_name="calc", body_hash="new_hash")
        prev_func = _make_func(qualified_name="calc", body_hash="old_hash")
        current = _make_snapshot(functions=[current_func])

        result = compare_snapshots(current, [prev_func])

        assert len(result) == 1
        diff = result[0]
        assert diff.qualified_name == "calc"
        assert diff.change_type == "modified"
        assert diff.current is current_func
        assert diff.previous_hash == "old_hash"

    def test_deleted_function(self):
        """Function in previous but not in current is classified as 'deleted'."""
        from claude_mem_lite.ast_tracker.diff import compare_snapshots

        prev_func = _make_func(qualified_name="removed_func", body_hash="old_hash")
        current = _make_snapshot(functions=[])

        result = compare_snapshots(current, [prev_func])

        assert len(result) == 1
        diff = result[0]
        assert diff.qualified_name == "removed_func"
        assert diff.change_type == "deleted"
        assert diff.current is None
        assert diff.previous_hash == "old_hash"

    def test_unchanged_function(self):
        """Function in both with same body_hash is classified as 'unchanged'."""
        from claude_mem_lite.ast_tracker.diff import compare_snapshots

        current_func = _make_func(qualified_name="stable", body_hash="same_hash")
        prev_func = _make_func(qualified_name="stable", body_hash="same_hash")
        current = _make_snapshot(functions=[current_func])

        result = compare_snapshots(current, [prev_func])

        assert len(result) == 1
        diff = result[0]
        assert diff.qualified_name == "stable"
        assert diff.change_type == "unchanged"
        assert diff.current is current_func
        assert diff.previous_hash == "same_hash"

    def test_mixed_changes(self):
        """Multiple functions with different change types in a single diff."""
        from claude_mem_lite.ast_tracker.diff import compare_snapshots

        # Current: new_func (new), modified_func (modified), unchanged_func (unchanged)
        new_func = _make_func(qualified_name="new_func", body_hash="n1")
        modified_func = _make_func(qualified_name="modified_func", body_hash="m2")
        unchanged_func = _make_func(qualified_name="unchanged_func", body_hash="u1")
        current = _make_snapshot(functions=[new_func, modified_func, unchanged_func])

        # Previous: deleted_func (deleted), modified_func (old hash), unchanged_func (same hash)
        deleted_func = _make_func(qualified_name="deleted_func", body_hash="d1")
        prev_modified = _make_func(qualified_name="modified_func", body_hash="m1")
        prev_unchanged = _make_func(qualified_name="unchanged_func", body_hash="u1")
        previous = [deleted_func, prev_modified, prev_unchanged]

        result = compare_snapshots(current, previous)
        by_name = _diff_by_name(result)

        assert len(result) == 4
        assert by_name["new_func"].change_type == "new"
        assert by_name["new_func"].previous_hash is None
        assert by_name["modified_func"].change_type == "modified"
        assert by_name["modified_func"].previous_hash == "m1"
        assert by_name["unchanged_func"].change_type == "unchanged"
        assert by_name["unchanged_func"].previous_hash == "u1"
        assert by_name["deleted_func"].change_type == "deleted"
        assert by_name["deleted_func"].current is None
        assert by_name["deleted_func"].previous_hash == "d1"


# ---------------------------------------------------------------------------
# Diff: edge cases (2 tests)
# ---------------------------------------------------------------------------


class TestDiffEdgeCases:
    """Test edge cases in compare_snapshots."""

    def test_empty_previous_all_new(self):
        """When previous is empty, all current functions are classified as 'new'."""
        from claude_mem_lite.ast_tracker.diff import compare_snapshots

        funcs = [
            _make_func(qualified_name="func_a", body_hash="ha"),
            _make_func(qualified_name="func_b", body_hash="hb"),
            _make_func(qualified_name="func_c", body_hash="hc"),
        ]
        current = _make_snapshot(functions=funcs)

        result = compare_snapshots(current, [])

        assert len(result) == 3
        assert all(d.change_type == "new" for d in result)
        assert all(d.previous_hash is None for d in result)
        by_name = _diff_by_name(result)
        assert set(by_name.keys()) == {"func_a", "func_b", "func_c"}

    def test_empty_current_all_deleted(self):
        """When current snapshot has no functions, all previous are 'deleted'."""
        from claude_mem_lite.ast_tracker.diff import compare_snapshots

        previous = [
            _make_func(qualified_name="old_a", body_hash="ha"),
            _make_func(qualified_name="old_b", body_hash="hb"),
        ]
        current = _make_snapshot(functions=[])

        result = compare_snapshots(current, previous)

        assert len(result) == 2
        assert all(d.change_type == "deleted" for d in result)
        assert all(d.current is None for d in result)
        by_name = _diff_by_name(result)
        assert by_name["old_a"].previous_hash == "ha"
        assert by_name["old_b"].previous_hash == "hb"
