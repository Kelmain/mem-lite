"""Mermaid diagram generation from function call data."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_mem_lite.ast_tracker.types import FunctionInfo

# Style mapping: change_type -> fill color
_CHANGE_STYLES: dict[str, str] = {
    "new": "#dfd",
    "modified": "#ffd",
    "deleted": "#fdd",
}


def _sanitize_for_mermaid(label: str) -> str:
    """Escape Mermaid syntax chars in signatures."""
    return (
        label.replace("[", "\u27e8")  # U+27E8
        .replace("]", "\u27e9")  # U+27E9
        .replace("{", "(")
        .replace("}", ")")
        .replace('"', "'")
        .replace("|", "\u2223")  # U+2223
        .replace("<", "\u2039")  # U+2039
        .replace(">", "\u203a")  # U+203A
    )


def _node_id(index: int) -> str:
    """Generate sequential node ID (A, B, C, ..., Z, AA, AB, ...)."""
    result = ""
    i = index
    while True:
        result = chr(ord("A") + i % 26) + result
        i = i // 26 - 1
        if i < 0:
            break
    return result


def _resolve_included_names(
    func_by_name: dict[str, FunctionInfo],
    show_all: bool,
    change_types: dict[str, str],
) -> set[str]:
    """Determine which function names to include in the graph."""
    if show_all:
        return set(func_by_name.keys())

    # Changed set: functions with new/modified/deleted status
    changed_set = {
        name
        for name, ct in change_types.items()
        if ct in ("new", "modified", "deleted") and name in func_by_name
    }
    included = set(changed_set)

    # Dependencies: call targets of changed functions that exist in file
    for name in changed_set:
        for call in func_by_name[name].calls:
            target = call.resolved_name
            if target and target in func_by_name:
                included.add(target)

    return included


def _build_edges(
    sorted_names: list[str],
    func_by_name: dict[str, FunctionInfo],
    name_to_id: dict[str, str],
) -> list[str]:
    """Build edge lines for call relationships."""
    edge_lines: list[str] = []
    for name in sorted_names:
        nid = name_to_id[name]
        for call in func_by_name[name].calls:
            target = call.resolved_name
            raw_target = call.raw_name.rstrip("()")
            target_name = target if target and target in name_to_id else None
            if not target_name and raw_target in name_to_id:
                target_name = raw_target

            if target_name:
                target_id = name_to_id[target_name]
                arrow = "-.->" if call.resolution == "unresolved" else "-->"
                edge_lines.append(f"    {nid} {arrow} {target_id}")
    return edge_lines


def generate_mermaid(
    functions: list[FunctionInfo],
    file_path: str | None = None,
    show_all: bool = False,
    change_types: dict[str, str] | None = None,
) -> str:
    """Generate Mermaid graph from function call data.

    Default scoping: includes changed functions (new/modified/deleted) plus
    any unchanged functions that are direct call targets of changed functions.
    This gives the relevant subgraph without dumping the entire file.

    With show_all=True: includes every function in the file.

    Args:
        functions: List of FunctionInfo from AST extraction.
        file_path: Optional file path for subgraph label.
        show_all: If True, include all functions regardless of change status.
        change_types: Mapping of qualified_name to change_type string.

    Returns:
        Mermaid graph definition string.
    """
    change_types = change_types or {}
    func_by_name: dict[str, FunctionInfo] = {f.qualified_name: f for f in functions}

    included_names = _resolve_included_names(func_by_name, show_all, change_types)
    sorted_names = sorted(included_names)
    name_to_id: dict[str, str] = {name: _node_id(i) for i, name in enumerate(sorted_names)}

    lines: list[str] = ["graph TD"]
    if not sorted_names:
        return "\n".join(lines)

    # Node definitions
    node_lines = [
        f'    {name_to_id[n]}["{_sanitize_for_mermaid(f"{n}: {func_by_name[n].signature}")}"]'
        for n in sorted_names
    ]

    # Edges
    edge_lines = _build_edges(sorted_names, func_by_name, name_to_id)

    # Style lines for changed functions
    style_lines = [
        f"    style {name_to_id[n]} fill:{_CHANGE_STYLES[ct]}"
        for n in sorted_names
        if (ct := change_types.get(n)) and ct in _CHANGE_STYLES
    ]

    # Assemble with optional subgraph wrapper
    if file_path:
        lines.append(f"    subgraph {file_path}")
        lines.extend(f"    {ln}" for ln in node_lines)
        lines.extend(f"    {ln}" for ln in edge_lines)
        lines.append("    end")
    else:
        lines.extend(node_lines)
        lines.extend(edge_lines)

    lines.extend(style_lines)
    return "\n".join(lines)
