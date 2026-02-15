"""Tests for FunctionExtractor — 27 tests (basics, signatures, classes, nesting, decorators, docstrings, body_hash)."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

from claude_mem_lite.ast_tracker.extractor import FunctionExtractor

if TYPE_CHECKING:
    from claude_mem_lite.ast_tracker.types import FileSnapshot, FunctionInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract(source: str, file_path: str = "/test/example.py") -> FileSnapshot:
    """Extract all functions from source and return the FileSnapshot."""
    extractor = FunctionExtractor(source, file_path)
    return extractor.extract()


def _by_name(snapshot: FileSnapshot) -> dict[str, FunctionInfo]:
    """Index functions by qualified_name for easy assertions."""
    return {f.qualified_name: f for f in snapshot.functions}


# ---------------------------------------------------------------------------
# Extractor: basics — 6 tests
# ---------------------------------------------------------------------------
class TestExtractorBasics:
    """Test basic extraction functionality."""

    def test_simple_function(self):
        """Extract a basic function with correct qualified_name, kind, signature, lines."""
        source = dedent("""\
            def greet(name: str) -> str:
                return f"Hello, {name}"
        """)
        snapshot = _extract(source)

        assert len(snapshot.functions) == 1
        func = snapshot.functions[0]
        assert func.qualified_name == "greet"
        assert func.kind == "function"
        assert func.signature == "greet(name: str) -> str"
        assert func.line_start == 1
        assert func.line_end == 2

    def test_multiple_functions(self):
        """Extract multiple functions from one source."""
        source = dedent("""\
            def first():
                pass

            def second():
                pass

            def third():
                pass
        """)
        snapshot = _extract(source)

        assert len(snapshot.functions) == 3
        names = [f.qualified_name for f in snapshot.functions]
        assert names == ["first", "second", "third"]

    def test_empty_file(self):
        """Empty string returns FileSnapshot with no functions, no parse_error."""
        snapshot = _extract("")

        assert snapshot.functions == []
        assert snapshot.parse_error is None
        assert snapshot.file_path == "/test/example.py"

    def test_syntax_error(self):
        """Invalid Python returns FileSnapshot with parse_error set, functions=[]."""
        source = "def broken(:\n    pass"
        snapshot = _extract(source)

        assert snapshot.functions == []
        assert snapshot.parse_error is not None
        assert "parse_error" not in repr(snapshot) or snapshot.parse_error  # parse_error is set

    def test_non_python_content(self):
        """Binary/garbage content returns snapshot with parse_error."""
        source = "\x00\x01\x02\x03 not valid python at all {"
        snapshot = _extract(source)

        assert snapshot.functions == []
        assert snapshot.parse_error is not None

    def test_duplicate_function_names(self):
        """Redefined function: extractor captures both instances."""
        source = dedent("""\
            def foo():
                return 1

            def foo():
                return 2
        """)
        snapshot = _extract(source)

        assert len(snapshot.functions) == 2
        assert snapshot.functions[0].qualified_name == "foo"
        assert snapshot.functions[1].qualified_name == "foo"


# ---------------------------------------------------------------------------
# Extractor: signatures — 6 tests
# ---------------------------------------------------------------------------
class TestExtractorSignatures:
    """Test signature building."""

    def test_positional_args(self):
        """def foo(a, b, c) -> 'foo(a, b, c)'."""
        source = dedent("""\
            def foo(a, b, c):
                pass
        """)
        snapshot = _extract(source)
        assert snapshot.functions[0].signature == "foo(a, b, c)"

    def test_default_values(self):
        """def foo(a, b=10, c='hi') -> "foo(a, b=10, c='hi')"."""
        source = dedent("""\
            def foo(a, b=10, c='hi'):
                pass
        """)
        snapshot = _extract(source)
        assert snapshot.functions[0].signature == "foo(a, b=10, c='hi')"

    def test_args_kwargs(self):
        """def foo(*args, **kwargs) -> 'foo(*args, **kwargs)'."""
        source = dedent("""\
            def foo(*args, **kwargs):
                pass
        """)
        snapshot = _extract(source)
        assert snapshot.functions[0].signature == "foo(*args, **kwargs)"

    def test_keyword_only_args(self):
        """def foo(a, *, key=None) -> 'foo(a, *, key=None)'."""
        source = dedent("""\
            def foo(a, *, key=None):
                pass
        """)
        snapshot = _extract(source)
        assert snapshot.functions[0].signature == "foo(a, *, key=None)"

    def test_return_type(self):
        """def foo() -> int -> 'foo() -> int'."""
        source = dedent("""\
            def foo() -> int:
                pass
        """)
        snapshot = _extract(source)
        assert snapshot.functions[0].signature == "foo() -> int"

    def test_self_stripping(self):
        """Method def bar(self, x: int) -> 'bar(x: int)' (self stripped)."""
        source = dedent("""\
            class MyClass:
                def bar(self, x: int):
                    pass
        """)
        snapshot = _extract(source)
        funcs = _by_name(snapshot)
        assert funcs["MyClass.bar"].signature == "bar(x: int)"


# ---------------------------------------------------------------------------
# Extractor: classes — 4 tests
# ---------------------------------------------------------------------------
class TestExtractorClasses:
    """Test class extraction."""

    def test_class_methods(self):
        """Extract methods from class, kind='method', parent_class set."""
        source = dedent("""\
            class UserService:
                def create(self, data):
                    pass

                def delete(self, user_id):
                    pass
        """)
        snapshot = _extract(source)
        funcs = _by_name(snapshot)

        # Class itself
        assert "UserService" in funcs
        assert funcs["UserService"].kind == "class"

        # Methods
        assert funcs["UserService.create"].kind == "method"
        assert funcs["UserService.create"].parent_class == "UserService"
        assert funcs["UserService.delete"].kind == "method"
        assert funcs["UserService.delete"].parent_class == "UserService"

    def test_async_methods(self):
        """Async method kind='async_method'."""
        source = dedent("""\
            class Service:
                async def fetch(self):
                    pass
        """)
        snapshot = _extract(source)
        funcs = _by_name(snapshot)

        assert funcs["Service.fetch"].kind == "async_method"

    def test_class_decorators(self):
        """Class has decorators captured."""
        source = dedent("""\
            @dataclass
            @frozen
            class Config:
                pass
        """)
        snapshot = _extract(source)
        funcs = _by_name(snapshot)

        assert funcs["Config"].decorators == ["@dataclass", "@frozen"]

    def test_nested_classes(self):
        """class Outer: class Inner: qualified_name='Outer.Inner'."""
        source = dedent("""\
            class Outer:
                class Inner:
                    def method(self):
                        pass
        """)
        snapshot = _extract(source)
        funcs = _by_name(snapshot)

        assert "Outer" in funcs
        assert "Outer.Inner" in funcs
        assert funcs["Outer.Inner"].kind == "class"
        assert "Outer.Inner.method" in funcs


# ---------------------------------------------------------------------------
# Extractor: nesting — 3 tests
# ---------------------------------------------------------------------------
class TestExtractorNesting:
    """Test nested function extraction."""

    def test_nested_functions(self):
        """def outer(): def inner(): -> qualified_name='outer.inner'."""
        source = dedent("""\
            def outer():
                def inner():
                    return 42
                return inner()
        """)
        snapshot = _extract(source)
        funcs = _by_name(snapshot)

        assert "outer" in funcs
        assert "outer.inner" in funcs
        assert funcs["outer"].kind == "function"
        assert funcs["outer.inner"].kind == "function"

    def test_closure(self):
        """Nested function accessing outer vars still extracts fine."""
        source = dedent("""\
            def make_adder(n):
                def adder(x):
                    return x + n
                return adder
        """)
        snapshot = _extract(source)
        funcs = _by_name(snapshot)

        assert "make_adder" in funcs
        assert "make_adder.adder" in funcs

    def test_function_inside_method(self):
        """Method containing nested function."""
        source = dedent("""\
            class Processor:
                def run(self):
                    def helper():
                        pass
                    helper()
        """)
        snapshot = _extract(source)
        funcs = _by_name(snapshot)

        assert "Processor.run" in funcs
        assert "Processor.run.helper" in funcs
        assert funcs["Processor.run.helper"].kind == "function"


# ---------------------------------------------------------------------------
# Extractor: decorators — 2 tests
# ---------------------------------------------------------------------------
class TestExtractorDecorators:
    """Test decorator extraction."""

    def test_simple_decorators(self):
        """@staticmethod, @property captured in decorators list."""
        source = dedent("""\
            class MyClass:
                @staticmethod
                def static_method():
                    pass

                @property
                def name(self):
                    return self._name
        """)
        snapshot = _extract(source)
        funcs = _by_name(snapshot)

        assert funcs["MyClass.static_method"].decorators == ["@staticmethod"]
        assert funcs["MyClass.name"].decorators == ["@property"]

    def test_decorator_with_arguments(self):
        """@router.post('/login') captured as full string."""
        source = dedent("""\
            @router.post("/login")
            def login():
                pass
        """)
        snapshot = _extract(source)
        func = snapshot.functions[0]

        assert len(func.decorators) == 1
        assert func.decorators[0] == "@router.post('/login')"


# ---------------------------------------------------------------------------
# Extractor: docstrings — 2 tests
# ---------------------------------------------------------------------------
class TestExtractorDocstrings:
    """Test docstring extraction."""

    def test_single_line_docstring(self):
        """Single-line docstring captured."""
        source = dedent('''\
            def greet():
                """Say hello."""
                return "hello"
        ''')
        snapshot = _extract(source)

        assert snapshot.functions[0].docstring == "Say hello."

    def test_multiline_docstring_first_line(self):
        """Only first line of multi-line docstring captured."""
        source = dedent('''\
            def process(data):
                """Process the input data.

                This function takes data and does complex
                processing on it.

                Args:
                    data: The input data.
                """
                return data
        ''')
        snapshot = _extract(source)

        assert snapshot.functions[0].docstring == "Process the input data."


# ---------------------------------------------------------------------------
# Extractor: body_hash — 4 tests
# ---------------------------------------------------------------------------
class TestExtractorBodyHash:
    """Test body hash computation."""

    def test_hash_same_logic_same_hash(self):
        """Identical logic produces same hash."""
        source1 = dedent("""\
            def foo():
                return 42
        """)
        source2 = dedent("""\
            def foo():
                return 42
        """)
        snap1 = _extract(source1)
        snap2 = _extract(source2)

        assert snap1.functions[0].body_hash == snap2.functions[0].body_hash
        assert len(snap1.functions[0].body_hash) == 32  # MD5 hex digest

    def test_hash_different_logic_different_hash(self):
        """Different return values produce different hash."""
        source1 = dedent("""\
            def foo():
                return 42
        """)
        source2 = dedent("""\
            def foo():
                return 99
        """)
        snap1 = _extract(source1)
        snap2 = _extract(source2)

        assert snap1.functions[0].body_hash != snap2.functions[0].body_hash

    def test_hash_ignores_whitespace(self):
        """Extra blank lines produce same hash (ast.dump ignores whitespace)."""
        source1 = dedent("""\
            def foo():
                return 42
        """)
        source2 = dedent("""\
            def foo():

                return 42

        """)
        snap1 = _extract(source1)
        snap2 = _extract(source2)

        assert snap1.functions[0].body_hash == snap2.functions[0].body_hash

    def test_hash_ignores_comments(self):
        """Added comments produce same hash (comments not in AST)."""
        source1 = dedent("""\
            def foo():
                return 42
        """)
        source2 = dedent("""\
            def foo():
                # This is a comment
                return 42  # inline comment
        """)
        snap1 = _extract(source1)
        snap2 = _extract(source2)

        assert snap1.functions[0].body_hash == snap2.functions[0].body_hash
