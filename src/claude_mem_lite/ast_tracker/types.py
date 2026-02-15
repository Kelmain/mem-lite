"""Internal data types for AST tracker.

Frozen dataclasses for extracted function metadata, call sites,
file snapshots, and change diffs. These map to Phase 0 Pydantic
models (FunctionMapEntry, CallGraphEdge) when persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CallInfo:
    """A function call site within a function body."""

    raw_name: str  # As written: "self.validate()", "db.query()"
    resolved_name: str | None  # Best guess: "AuthService.validate" or None
    resolution: str  # direct | self_method | import | unresolved
    line_number: int


@dataclass(frozen=True, slots=True)
class FunctionInfo:
    """Extracted function/method/class metadata."""

    qualified_name: str  # "AuthService.authenticate"
    kind: str  # function | method | async_function | async_method | class
    parent_class: str | None
    signature: str  # "authenticate(email: str, password: str) -> Token"
    decorators: list[str]  # ["@router.post('/login')", "@require_auth"]
    docstring: str | None  # First line only
    line_start: int
    line_end: int
    body_hash: str  # MD5 of ast.dump(node), full 32 hex chars
    calls: list[CallInfo] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """Complete AST extraction result for a single file."""

    file_path: str
    functions: list[FunctionInfo]
    import_map: dict[str, str]  # local_name -> qualified_name
    parse_error: str | None = None


@dataclass(frozen=True, slots=True)
class FunctionDiff:
    """Change classification for a single function."""

    qualified_name: str
    change_type: str  # new | modified | deleted | unchanged
    current: FunctionInfo | None  # None if deleted
    previous_hash: str | None  # None if new
