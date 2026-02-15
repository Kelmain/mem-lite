"""Integration tests for AST Tracker public API (scan_file from disk and from source)."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

from claude_mem_lite.ast_tracker import scan_file

if TYPE_CHECKING:
    from pathlib import Path


class TestScanFileFromDisk:
    """Test scan_file reading a real .py file from disk."""

    def test_scan_file_from_disk(self, tmp_path: Path):
        """Write a .py file to tmp_path, scan it, verify FileSnapshot correctness."""
        source = dedent("""\
            import os

            class Greeter:
                \"\"\"A simple greeter.\"\"\"

                def greet(self, name: str) -> str:
                    return f"Hello, {name}"

            def add(a: int, b: int) -> int:
                \"\"\"Add two numbers.\"\"\"
                return a + b
        """)
        file = tmp_path / "example.py"
        file.write_text(source, encoding="utf-8")

        snapshot = scan_file(str(file))

        assert snapshot.file_path == str(file)
        assert snapshot.parse_error is None

        names = {f.qualified_name for f in snapshot.functions}
        assert "Greeter" in names
        assert "Greeter.greet" in names
        assert "add" in names
        assert len(snapshot.functions) == 3

        by_name = {f.qualified_name: f for f in snapshot.functions}

        greeter_cls = by_name["Greeter"]
        assert greeter_cls.kind == "class"
        assert greeter_cls.docstring == "A simple greeter."

        greet_method = by_name["Greeter.greet"]
        assert greet_method.kind == "method"
        assert greet_method.signature == "greet(name: str) -> str"
        assert greet_method.parent_class == "Greeter"

        add_func = by_name["add"]
        assert add_func.kind == "function"
        assert add_func.docstring == "Add two numbers."
        assert add_func.signature == "add(a: int, b: int) -> int"

        # Import map should contain 'os'
        assert "os" in snapshot.import_map


class TestScanFileFromSource:
    """Test scan_file with source provided directly (no disk read)."""

    def test_scan_file_from_source(self):
        """Call scan_file with source kwarg, verify extraction without disk access."""
        snapshot = scan_file("virtual.py", source="def hello(): pass")

        assert snapshot.file_path == "virtual.py"
        assert snapshot.parse_error is None
        assert len(snapshot.functions) == 1

        func = snapshot.functions[0]
        assert func.qualified_name == "hello"
        assert func.kind == "function"
        assert func.signature == "hello()"
        assert func.docstring is None
        assert func.line_start == 1
        assert func.line_end == 1
        assert len(func.body_hash) == 32  # MD5 hex digest
