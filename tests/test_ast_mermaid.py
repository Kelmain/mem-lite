"""Tests for AST Tracker Mermaid diagram generation -- 5 tests."""

from __future__ import annotations

from claude_mem_lite.ast_tracker.mermaid import generate_mermaid
from claude_mem_lite.ast_tracker.types import CallInfo, FunctionInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_func(
    qualified_name: str,
    signature: str,
    *,
    calls: list[CallInfo] | None = None,
    kind: str = "function",
    parent_class: str | None = None,
    line_start: int = 1,
    line_end: int = 10,
) -> FunctionInfo:
    """Build a FunctionInfo with sensible defaults for testing."""
    return FunctionInfo(
        qualified_name=qualified_name,
        kind=kind,
        parent_class=parent_class,
        signature=signature,
        decorators=[],
        docstring=None,
        line_start=line_start,
        line_end=line_end,
        body_hash="a" * 32,
        calls=calls or [],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleFileGraph:
    """Basic graph with a few functions and calls, using show_all=True."""

    def test_graph_td_header(self):
        """Output must start with 'graph TD'."""
        funcs = [
            _make_func("foo", "foo()"),
            _make_func("bar", "bar(x: int) -> str"),
        ]
        result = generate_mermaid(funcs, file_path="src/app.py", show_all=True)
        assert result.strip().startswith("graph TD")

    def test_subgraph_wraps_file(self):
        """When file_path given, nodes wrapped in subgraph."""
        funcs = [_make_func("foo", "foo()")]
        result = generate_mermaid(funcs, file_path="src/app.py", show_all=True)
        assert "subgraph src/app.py" in result
        assert "end" in result

    def test_no_subgraph_without_file_path(self):
        """When file_path is None, no subgraph wrapper."""
        funcs = [_make_func("foo", "foo()")]
        result = generate_mermaid(funcs, show_all=True)
        assert "subgraph" not in result

    def test_nodes_contain_qualified_name_and_signature(self):
        """Each node label has 'qualified_name: signature' format (sanitized)."""
        funcs = [
            _make_func(
                "AuthService.authenticate",
                "authenticate(email, password) -> Token",
            ),
        ]
        result = generate_mermaid(funcs, show_all=True)
        # The -> arrow gets sanitized (> becomes U+203A), but the structure is preserved
        assert "AuthService.authenticate: authenticate(email, password)" in result
        assert "Token" in result

    def test_edges_for_resolved_calls(self):
        """Resolved calls produce solid arrows between nodes."""
        call = CallInfo(
            raw_name="bar()",
            resolved_name="bar",
            resolution="direct",
            line_number=5,
        )
        funcs = [
            _make_func("foo", "foo()", calls=[call]),
            _make_func("bar", "bar()"),
        ]
        result = generate_mermaid(funcs, show_all=True)
        # Should have a solid arrow between the two node IDs
        lines = result.strip().splitlines()
        edge_lines = [ln.strip() for ln in lines if "-->" in ln and "-.->" not in ln]
        assert len(edge_lines) >= 1


class TestChangeTypeStyling:
    """Functions with new/modified/deleted change types get correct style fills."""

    def test_new_function_green_style(self):
        funcs = [_make_func("create_user", "create_user(data) -> User")]
        result = generate_mermaid(
            funcs,
            show_all=True,
            change_types={"create_user": "new"},
        )
        # Node ID for create_user should have green fill
        assert "fill:#dfd" in result

    def test_modified_function_yellow_style(self):
        funcs = [_make_func("update_user", "update_user(id, data) -> User")]
        result = generate_mermaid(
            funcs,
            show_all=True,
            change_types={"update_user": "modified"},
        )
        assert "fill:#ffd" in result

    def test_deleted_function_red_style(self):
        funcs = [_make_func("delete_user", "delete_user(id) -> None")]
        result = generate_mermaid(
            funcs,
            show_all=True,
            change_types={"delete_user": "deleted"},
        )
        assert "fill:#fdd" in result

    def test_unchanged_function_no_style(self):
        """Unchanged functions (not in change_types) get no style line."""
        funcs = [_make_func("helper", "helper() -> None")]
        result = generate_mermaid(
            funcs,
            show_all=True,
            change_types={},
        )
        assert "style" not in result


class TestUnresolvedEdgeDashing:
    """Edges from unresolved calls use '-.->'' not '-->'."""

    def test_unresolved_call_dashed_edge(self):
        """An unresolved call produces a dashed arrow."""
        call = CallInfo(
            raw_name="external_api()",
            resolved_name=None,
            resolution="unresolved",
            line_number=7,
        )
        funcs = [
            _make_func("process", "process(data)", calls=[call]),
            _make_func("external_api", "external_api()"),
        ]
        result = generate_mermaid(funcs, show_all=True)
        lines = result.strip().splitlines()
        dashed = [ln.strip() for ln in lines if "-.->" in ln]
        assert len(dashed) >= 1

    def test_resolved_call_solid_edge(self):
        """A resolved call produces a solid arrow (not dashed)."""
        call = CallInfo(
            raw_name="helper()",
            resolved_name="helper",
            resolution="direct",
            line_number=3,
        )
        funcs = [
            _make_func("main", "main()", calls=[call]),
            _make_func("helper", "helper()"),
        ]
        result = generate_mermaid(funcs, show_all=True)
        lines = result.strip().splitlines()
        solid = [ln.strip() for ln in lines if "-->" in ln and "-.->" not in ln]
        assert len(solid) >= 1


class TestDefaultScoping:
    """Only changed functions + their direct call targets shown by default."""

    def test_unchanged_without_connection_excluded(self):
        """Unchanged function not called by any changed function is excluded."""
        funcs = [
            _make_func("changed_fn", "changed_fn()"),
            _make_func("unrelated", "unrelated()"),
        ]
        result = generate_mermaid(
            funcs,
            change_types={"changed_fn": "modified"},
        )
        assert "changed_fn" in result
        assert "unrelated" not in result

    def test_dependency_of_changed_included(self):
        """Unchanged function that is a call target of changed function IS included."""
        call = CallInfo(
            raw_name="dep()",
            resolved_name="dep",
            resolution="direct",
            line_number=5,
        )
        funcs = [
            _make_func("changed_fn", "changed_fn()", calls=[call]),
            _make_func("dep", "dep()"),
        ]
        result = generate_mermaid(
            funcs,
            change_types={"changed_fn": "modified"},
        )
        assert "changed_fn" in result
        assert "dep" in result

    def test_no_changes_yields_empty(self):
        """When change_types is None (default), show_all=False gives empty graph."""
        funcs = [
            _make_func("foo", "foo()"),
            _make_func("bar", "bar()"),
        ]
        result = generate_mermaid(funcs)
        # Only the header, no nodes
        lines = [ln.strip() for ln in result.strip().splitlines() if ln.strip()]
        assert lines == ["graph TD"]

    def test_dependency_not_in_file_excluded(self):
        """Call target not present in functions list is not rendered as a node."""
        call = CallInfo(
            raw_name="external()",
            resolved_name="other_module.external",
            resolution="import",
            line_number=5,
        )
        funcs = [
            _make_func("changed_fn", "changed_fn()", calls=[call]),
        ]
        result = generate_mermaid(
            funcs,
            change_types={"changed_fn": "new"},
        )
        assert "changed_fn" in result
        assert "other_module.external" not in result


class TestTypeHintSanitization:
    """Signatures with brackets and special chars don't break Mermaid syntax."""

    def test_list_brackets_escaped(self):
        """list[str] brackets replaced with angle brackets."""
        funcs = [
            _make_func(
                "get_names",
                "get_names() -> list[str]",
            ),
        ]
        result = generate_mermaid(funcs, show_all=True)
        # Original brackets should not appear in node labels
        assert "list[str]" not in result
        assert "list\u27e8str\u27e9" in result

    def test_dict_brackets_escaped(self):
        """dict[str, int] brackets replaced with angle brackets."""
        funcs = [
            _make_func(
                "get_counts",
                "get_counts() -> dict[str, int]",
            ),
        ]
        result = generate_mermaid(funcs, show_all=True)
        assert "dict[str, int]" not in result
        assert "dict\u27e8str, int\u27e9" in result

    def test_quotes_in_signature_escaped(self):
        """Double quotes in signatures replaced with single quotes."""
        funcs = [
            _make_func(
                "greet",
                'greet(name: str = "world") -> str',
            ),
        ]
        result = generate_mermaid(funcs, show_all=True)
        # The node label wrapping uses quotes, inner quotes must be escaped
        assert "'world'" in result

    def test_pipe_in_signature_escaped(self):
        """Pipe char | in union types replaced with math divider."""
        funcs = [
            _make_func(
                "maybe",
                "maybe() -> str | None",
            ),
        ]
        result = generate_mermaid(funcs, show_all=True)
        assert "str | None" not in result
        assert "str \u2223 None" in result
