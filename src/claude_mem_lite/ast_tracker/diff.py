"""Change detection -- compare current file state against previous snapshot."""

from __future__ import annotations

from claude_mem_lite.ast_tracker.types import FileSnapshot, FunctionDiff, FunctionInfo


def compare_snapshots(
    current: FileSnapshot,
    previous: list[FunctionInfo],
) -> list[FunctionDiff]:
    """Compare current extraction against previous snapshot.

    Classifies each function as new, deleted, modified, or unchanged
    by matching on ``qualified_name`` and comparing ``body_hash``.

    Duplicate qualified_names are resolved by last-one-wins (dict keying),
    matching Python's shadowing semantics.

    Args:
        current: The current file snapshot with extracted functions.
        previous: List of FunctionInfo from the previous snapshot (e.g. from DB).

    Returns:
        List of FunctionDiff, one per unique qualified_name across both snapshots.
    """
    current_map: dict[str, FunctionInfo] = {f.qualified_name: f for f in current.functions}
    previous_map: dict[str, FunctionInfo] = {f.qualified_name: f for f in previous}

    diffs: list[FunctionDiff] = []

    # Walk current functions: classify as new, modified, or unchanged.
    for name, func in current_map.items():
        prev = previous_map.get(name)
        if prev is None:
            diffs.append(
                FunctionDiff(
                    qualified_name=name,
                    change_type="new",
                    current=func,
                    previous_hash=None,
                )
            )
        elif func.body_hash != prev.body_hash:
            diffs.append(
                FunctionDiff(
                    qualified_name=name,
                    change_type="modified",
                    current=func,
                    previous_hash=prev.body_hash,
                )
            )
        else:
            diffs.append(
                FunctionDiff(
                    qualified_name=name,
                    change_type="unchanged",
                    current=func,
                    previous_hash=prev.body_hash,
                )
            )

    # Walk previous functions not in current: classify as deleted.
    for name, func in previous_map.items():
        if name not in current_map:
            diffs.append(
                FunctionDiff(
                    qualified_name=name,
                    change_type="deleted",
                    current=None,
                    previous_hash=func.body_hash,
                )
            )

    return diffs
