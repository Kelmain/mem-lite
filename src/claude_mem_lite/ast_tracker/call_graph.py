"""CallExtractor — extract and resolve function calls within function bodies."""

from __future__ import annotations

import ast

from claude_mem_lite.ast_tracker.types import CallInfo

# ---------------------------------------------------------------------------
# Noise filters — callables, objects, and attribute methods to skip
# ---------------------------------------------------------------------------

NOISE_CALLABLES: frozenset[str] = frozenset(
    {
        # Builtins
        "print",
        "len",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "isinstance",
        "issubclass",
        "hasattr",
        "getattr",
        "setattr",
        "delattr",
        "super",
        "type",
        "id",
        "hash",
        "repr",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "frozenset",
        "bytes",
        "bytearray",
        "sorted",
        "reversed",
        "min",
        "max",
        "sum",
        "any",
        "all",
        "abs",
        "round",
        "divmod",
        "pow",
        "chr",
        "ord",
        "hex",
        "oct",
        "bin",
        "format",
        "vars",
        "dir",
        "callable",
        "iter",
        "next",
        "open",
        "input",
        "staticmethod",
        "classmethod",
        "property",
        # Common stdlib that's noise
        "logging.getLogger",
        "logging.debug",
        "logging.info",
        "logging.warning",
        "logging.error",
        "logging.critical",
    }
)

NOISE_OBJECTS: frozenset[str] = frozenset(
    {
        "logger",
        "log",
        "logging",
        "os",
        "sys",
        "re",
        "json",
        "math",
        "copy",
        "functools",
        "itertools",
        "collections",
        "pathlib",
        "Path",
    }
)

NOISE_ATTRIBUTE_CALLS: frozenset[str] = frozenset(
    {
        # dict/list/set methods
        "items",
        "keys",
        "values",
        "get",
        "pop",
        "update",
        "append",
        "extend",
        "insert",
        "remove",
        "clear",
        "copy",
        "sort",
        "add",
        "discard",
        # string methods
        "strip",
        "lstrip",
        "rstrip",
        "split",
        "rsplit",
        "join",
        "replace",
        "format",
        "encode",
        "decode",
        "lower",
        "upper",
        "startswith",
        "endswith",
        "find",
        "rfind",
        "count",
        # general object methods
        "__init__",
        "__str__",
        "__repr__",
        "__enter__",
        "__exit__",
        "__aenter__",
        "__aexit__",
    }
)


def _get_attr_chain(node: ast.expr) -> list[str] | None:
    """Unpack a chain of attribute accesses into name parts.

    Examples:
        self.validate -> ["self", "validate"]
        a.b.c.method -> ["a", "b", "c", "method"]
        foo -> ["foo"]

    Returns None if the expression contains non-Name/Attribute nodes.
    """
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        parts.reverse()
        return parts
    return None


class CallExtractor:
    """Extract and resolve function calls within a function body.

    Resolution types:
    - direct: Simple name call, resolved via local scope. foo() -> foo
    - self_method: self.method() -> ClassName.method
    - import: Resolved via import map. db.query() -> sqlalchemy.orm.Session.query
    - unresolved: Can't determine target. a.b.c.method() -> None
    """

    def __init__(self, import_map: dict[str, str], current_class: str | None = None) -> None:
        self.import_map = import_map
        self.current_class = current_class

    def extract_calls(self, func_node: ast.AST) -> list[CallInfo]:
        """Extract all non-noise function calls from a function body."""
        calls: list[CallInfo] = []
        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue
            call_info = self._resolve_call(node)
            if call_info is not None:
                calls.append(call_info)
        return calls

    def _resolve_call(self, node: ast.Call) -> CallInfo | None:
        """Resolve a single ast.Call node to a CallInfo, or None if noise."""
        func = node.func
        line = node.lineno

        # Case 1: Simple name call — foo()
        if isinstance(func, ast.Name):
            return self._resolve_name_call(func.id, line)

        # Case 2+: Attribute call — obj.method()
        if isinstance(func, ast.Attribute):
            return self._resolve_attr_call(func, line)

        # Unrecognized call form (e.g. subscript call) — skip
        return None

    def _resolve_name_call(self, name: str, line: int) -> CallInfo | None:
        """Resolve a simple name call like foo()."""
        if name in NOISE_CALLABLES:
            return None

        resolved = self.import_map.get(name, name)
        return CallInfo(
            raw_name=name,
            resolved_name=resolved,
            resolution="direct",
            line_number=line,
        )

    def _resolve_attr_call(self, func: ast.Attribute, line: int) -> CallInfo | None:
        """Resolve an attribute call like obj.method() or a.b.c.method()."""
        parts = _get_attr_chain(func)
        if parts is None:
            return None

        # Build raw_name from parts (no parens)
        raw_name = ".".join(parts)
        method = parts[-1]

        # Self method: self.method()
        if len(parts) == 2 and parts[0] == "self" and self.current_class:
            return CallInfo(
                raw_name=raw_name,
                resolved_name=f"{self.current_class}.{method}",
                resolution="self_method",
                line_number=line,
            )

        # Object is the first part of the chain
        obj = parts[0]

        # Noise filter: object (logger, os, sys) or qualified callable (logging.getLogger)
        if obj in NOISE_OBJECTS or raw_name in NOISE_CALLABLES:
            return None

        # Import map resolution: object name is in import_map
        if obj in self.import_map:
            qualified = self.import_map[obj]
            # Append remaining parts after the object name
            suffix = ".".join(parts[1:])
            return CallInfo(
                raw_name=raw_name,
                resolved_name=f"{qualified}.{suffix}",
                resolution="import",
                line_number=line,
            )

        # Chained attribute or unknown object -> unresolved
        # Filter noise attribute calls on unresolved edges
        if method in NOISE_ATTRIBUTE_CALLS:
            return None

        return CallInfo(
            raw_name=raw_name,
            resolved_name=None,
            resolution="unresolved",
            line_number=line,
        )
