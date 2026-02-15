"""FunctionExtractor — extract function/method/class definitions from Python source."""

from __future__ import annotations

import ast
import hashlib

from claude_mem_lite.ast_tracker.call_graph import CallExtractor
from claude_mem_lite.ast_tracker.types import FileSnapshot, FunctionInfo


class FunctionExtractor(ast.NodeVisitor):
    """Extract function/method/class definitions from Python source.

    Usage:
        extractor = FunctionExtractor(source, file_path)
        snapshot = extractor.extract()
    """

    def __init__(self, source: str, file_path: str) -> None:
        self.source = source
        self.file_path = file_path
        self.functions: list[FunctionInfo] = []
        self.import_map: dict[str, str] = {}
        self._context_stack: list[str] = []
        self._current_class: str | None = None

    def extract(self) -> FileSnapshot:
        """Parse source and extract all functions. Returns FileSnapshot."""
        try:
            tree = ast.parse(self.source)
        except SyntaxError as exc:
            return FileSnapshot(
                file_path=self.file_path,
                functions=[],
                import_map={},
                parse_error=str(exc),
            )

        self._build_import_map(tree)
        self.visit(tree)

        return FileSnapshot(
            file_path=self.file_path,
            functions=self.functions,
            import_map=self.import_map,
        )

    def _build_import_map(self, tree: ast.Module) -> None:
        """Walk tree once for Import/ImportFrom, build local->qualified map."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # import foo -> foo -> foo
                    # import foo as bar -> bar -> foo
                    local = alias.asname if alias.asname else alias.name
                    self.import_map[local] = alias.name

            elif isinstance(node, ast.ImportFrom):
                level = node.level or 0
                module = node.module or ""

                if level > 0:
                    # Relative import: from . import utils, from .service import Y
                    prefix = f"__rel{level}__"
                    base = f"{prefix}.{module}" if module else prefix
                else:
                    base = module

                for alias in node.names:
                    local = alias.asname if alias.asname else alias.name
                    self.import_map[local] = f"{base}.{alias.name}"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class as FunctionInfo, then visit children for methods."""
        # Build qualified name from context stack
        qualified_name = self._qualified_name(node.name)

        # Extract decorators
        decorators = [self._decorator_string(d) for d in node.decorator_list]

        # Extract docstring
        docstring = self._extract_docstring(node)

        # Build signature (class name with base classes if any)
        signature = self._build_class_signature(node)

        # Compute body hash
        body_hash = self._compute_body_hash(node)

        func_info = FunctionInfo(
            qualified_name=qualified_name,
            kind="class",
            parent_class=self._current_class,
            signature=signature,
            decorators=decorators,
            docstring=docstring,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            body_hash=body_hash,
        )
        self.functions.append(func_info)

        # Push class context and visit children
        prev_class = self._current_class
        self._current_class = qualified_name
        self._context_stack.append(node.name)

        self.generic_visit(node)

        self._context_stack.pop()
        self._current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract synchronous function definition."""
        self._process_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Extract asynchronous function definition."""
        self._process_function(node, is_async=True)

    def _process_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, is_async: bool
    ) -> None:
        """Extract all metadata from a function node, build FunctionInfo."""
        qualified_name = self._qualified_name(node.name)

        # Determine kind
        if self._current_class is not None:
            kind = "async_method" if is_async else "method"
        else:
            kind = "async_function" if is_async else "function"

        # Extract calls using CallExtractor
        call_extractor = CallExtractor(
            import_map=self.import_map,
            current_class=self._current_class,
        )
        calls = call_extractor.extract_calls(node)

        # Build signature
        signature = self._build_signature(node)

        # Extract decorators
        decorators = [self._decorator_string(d) for d in node.decorator_list]

        # Extract docstring
        docstring = self._extract_docstring(node)

        # Compute body hash
        body_hash = self._compute_body_hash(node)

        func_info = FunctionInfo(
            qualified_name=qualified_name,
            kind=kind,
            parent_class=self._current_class,
            signature=signature,
            decorators=decorators,
            docstring=docstring,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            body_hash=body_hash,
            calls=calls,
        )
        self.functions.append(func_info)

        # Push context for nested functions, visit children
        # Clear _current_class so nested functions inside methods are "function" not "method"
        prev_class = self._current_class
        self._current_class = None
        self._context_stack.append(node.name)
        self.generic_visit(node)
        self._context_stack.pop()
        self._current_class = prev_class

    def _qualified_name(self, name: str) -> str:
        """Build qualified name from context stack + name."""
        if self._context_stack:
            return ".".join([*self._context_stack, name])
        return name

    def _build_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Build readable signature string from function node."""
        args = node.args
        parts: list[str] = []

        # Positional args (posonlyargs + regular args)
        all_positional = [*args.posonlyargs, *args.args]
        # Defaults are right-aligned: last N positional args get defaults
        num_defaults = len(args.defaults)
        num_positional = len(all_positional)

        for i, arg in enumerate(all_positional):
            # Skip self/cls
            if i == 0 and arg.arg in ("self", "cls"):
                continue

            default_index = i - (num_positional - num_defaults)
            if default_index >= 0:
                default = args.defaults[default_index]
                parts.append(self._format_arg(arg, default=default))
            else:
                parts.append(self._format_arg(arg))

        # *args or bare * for keyword-only
        if args.vararg:
            parts.append(self._format_arg(args.vararg, prefix="*"))
        elif args.kwonlyargs:
            parts.append("*")

        # Keyword-only args
        for i, arg in enumerate(args.kwonlyargs):
            kw_default = args.kw_defaults[i]
            if kw_default is not None:
                parts.append(self._format_arg(arg, default=kw_default))
            else:
                parts.append(self._format_arg(arg))

        # **kwargs
        if args.kwarg:
            parts.append(self._format_arg(args.kwarg, prefix="**"))

        sig = f"{node.name}({', '.join(parts)})"

        # Return type
        if node.returns:
            sig += f" -> {ast.unparse(node.returns)}"

        return sig

    def _format_arg(
        self, arg: ast.arg, *, default: ast.expr | None = None, prefix: str = ""
    ) -> str:
        """Format a single argument with optional annotation and default."""
        result = f"{prefix}{arg.arg}"
        if arg.annotation:
            result += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            result += f"={ast.unparse(default)}"
        return result

    def _build_class_signature(self, node: ast.ClassDef) -> str:
        """Build class signature with base classes."""
        if node.bases:
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            return f"{node.name}({bases})"
        return node.name

    def _compute_body_hash(self, node: ast.AST) -> str:
        """Compute MD5 hash of ast.dump(node) — full 32 hex chars."""
        return hashlib.md5(ast.dump(node).encode()).hexdigest()

    def _extract_docstring(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> str | None:
        """Extract first line of docstring from function/class body."""
        if not node.body:
            return None
        first_stmt = node.body[0]
        if (
            isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)
        ):
            docstring = first_stmt.value.value.strip()
            # Return first line only
            return docstring.split("\n")[0].strip()
        return None

    def _decorator_string(self, node: ast.expr) -> str:
        """Convert a decorator AST node to its string representation."""
        return f"@{ast.unparse(node)}"
