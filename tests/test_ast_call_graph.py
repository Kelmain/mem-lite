"""Tests for CallExtractor — 15 tests (resolution, import map, noise filtering)."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from claude_mem_lite.ast_tracker.call_graph import CallExtractor

if TYPE_CHECKING:
    from claude_mem_lite.ast_tracker.types import CallInfo


def _parse_function(source: str, func_name: str = "") -> ast.AST:
    """Parse source and return the first function or specified function node."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and (
            not func_name or node.name == func_name
        ):
            return node
    msg = f"No function {func_name!r} found in source"
    raise ValueError(msg)


def _names(calls: list[CallInfo]) -> list[str]:
    """Extract raw_name list from CallInfo list for easy assertions."""
    return [c.raw_name for c in calls]


def _resolved(calls: list[CallInfo]) -> dict[str, str | None]:
    """Map raw_name -> resolved_name for easy assertions."""
    return {c.raw_name: c.resolved_name for c in calls}


def _resolutions(calls: list[CallInfo]) -> dict[str, str]:
    """Map raw_name -> resolution type for easy assertions."""
    return {c.raw_name: c.resolution for c in calls}


# ---------------------------------------------------------------------------
# CallExtractor: resolution — 7 tests
# ---------------------------------------------------------------------------
class TestCallResolution:
    """Test call target resolution logic."""

    def test_direct_call_resolution(self):
        """Direct call foo() resolves via import_map or keeps local name."""
        source = """\
def example():
    result = foo()
    bar()
"""
        func_node = _parse_function(source)
        extractor = CallExtractor(import_map={}, current_class=None)
        calls = extractor.extract_calls(func_node)

        resolutions = _resolutions(calls)
        assert resolutions["foo"] == "direct"
        assert resolutions["bar"] == "direct"

        resolved = _resolved(calls)
        assert resolved["foo"] == "foo"
        assert resolved["bar"] == "bar"

    def test_self_method_resolution(self):
        """self.baz() inside class Foo resolves to 'Foo.baz'."""
        source = """\
def process(self):
    self.baz()
    self.validate()
"""
        func_node = _parse_function(source)
        extractor = CallExtractor(import_map={}, current_class="Foo")
        calls = extractor.extract_calls(func_node)

        resolved = _resolved(calls)
        assert resolved["self.baz"] == "Foo.baz"
        assert resolved["self.validate"] == "Foo.validate"

        resolutions = _resolutions(calls)
        assert resolutions["self.baz"] == "self_method"
        assert resolutions["self.validate"] == "self_method"

    def test_import_resolution(self):
        """AuthService.authenticate() resolves via import map."""
        source = """\
def login():
    token = AuthService.authenticate(email, password)
"""
        func_node = _parse_function(source)
        extractor = CallExtractor(
            import_map={"AuthService": "myapp.services.auth.AuthService"},
            current_class=None,
        )
        calls = extractor.extract_calls(func_node)

        assert len(calls) == 1
        call = calls[0]
        assert call.raw_name == "AuthService.authenticate"
        assert call.resolved_name == "myapp.services.auth.AuthService.authenticate"
        assert call.resolution == "import"

    def test_chained_attribute_unresolved(self):
        """a.b.c.method() with deep chaining is unresolved."""
        source = """\
def example():
    a.b.c.method()
"""
        func_node = _parse_function(source)
        extractor = CallExtractor(import_map={}, current_class=None)
        calls = extractor.extract_calls(func_node)

        assert len(calls) == 1
        call = calls[0]
        assert call.resolution == "unresolved"
        assert call.resolved_name is None

    def test_unresolved_call(self):
        """Unknown object.method() that is not in import_map is unresolved."""
        source = """\
def example():
    unknown_service.do_thing()
"""
        func_node = _parse_function(source)
        extractor = CallExtractor(import_map={}, current_class=None)
        calls = extractor.extract_calls(func_node)

        assert len(calls) == 1
        call = calls[0]
        assert call.raw_name == "unknown_service.do_thing"
        assert call.resolution == "unresolved"
        assert call.resolved_name is None

    def test_noise_callable_filtered(self):
        """Noise callables like print(), len(), isinstance() are skipped."""
        source = """\
def example():
    print("hello")
    x = len(items)
    if isinstance(x, int):
        real_work()
"""
        func_node = _parse_function(source)
        extractor = CallExtractor(import_map={}, current_class=None)
        calls = extractor.extract_calls(func_node)

        names = _names(calls)
        assert "print" not in names
        assert "len" not in names
        assert "isinstance" not in names
        assert "real_work" in names

    def test_await_call_resolution(self):
        """Calls inside await expressions are found and resolved."""
        source = """\
async def example():
    result = await fetch_data()
    await db.execute(query)
"""
        func_node = _parse_function(source)
        extractor = CallExtractor(
            import_map={"db": "sqlalchemy.ext.asyncio.AsyncSession"},
            current_class=None,
        )
        calls = extractor.extract_calls(func_node)

        names = _names(calls)
        assert "fetch_data" in names
        assert "db.execute" in names

        resolved = _resolved(calls)
        assert resolved["fetch_data"] == "fetch_data"
        assert resolved["db.execute"] == "sqlalchemy.ext.asyncio.AsyncSession.execute"


# ---------------------------------------------------------------------------
# CallExtractor: import map — 5 tests
# ---------------------------------------------------------------------------
class TestImportMap:
    """Test import map handling (maps are pre-built, passed to CallExtractor)."""

    def test_import_x(self):
        """import requests maps requests -> requests; calls resolve via import."""
        source = """\
def example():
    requests.get("http://example.com")
"""
        func_node = _parse_function(source)
        extractor = CallExtractor(
            import_map={"requests": "requests"},
            current_class=None,
        )
        calls = extractor.extract_calls(func_node)

        assert len(calls) == 1
        assert calls[0].resolved_name == "requests.get"
        assert calls[0].resolution == "import"

    def test_from_x_import_y(self):
        """from datetime import datetime maps datetime -> datetime.datetime."""
        source = """\
def example():
    now = datetime.now()
"""
        func_node = _parse_function(source)
        extractor = CallExtractor(
            import_map={"datetime": "datetime.datetime"},
            current_class=None,
        )
        calls = extractor.extract_calls(func_node)

        assert len(calls) == 1
        assert calls[0].raw_name == "datetime.now"
        assert calls[0].resolved_name == "datetime.datetime.now"
        assert calls[0].resolution == "import"

    def test_from_x_import_y_as_z(self):
        """from collections import OrderedDict as OD maps OD -> collections.OrderedDict."""
        source = """\
def example():
    d = OD()
"""
        func_node = _parse_function(source)
        extractor = CallExtractor(
            import_map={"OD": "collections.OrderedDict"},
            current_class=None,
        )
        calls = extractor.extract_calls(func_node)

        assert len(calls) == 1
        assert calls[0].raw_name == "OD"
        assert calls[0].resolved_name == "collections.OrderedDict"
        assert calls[0].resolution == "direct"

    def test_relative_import_level_1(self):
        """from . import utils maps utils -> __rel1__.utils."""
        source = """\
def example():
    utils.parse(data)
"""
        func_node = _parse_function(source)
        extractor = CallExtractor(
            import_map={"utils": "__rel1__.utils"},
            current_class=None,
        )
        calls = extractor.extract_calls(func_node)

        assert len(calls) == 1
        assert calls[0].resolved_name == "__rel1__.utils.parse"
        assert calls[0].resolution == "import"

    def test_relative_import_from_dotmodule(self):
        """from .service import AuthService maps AuthService -> __rel1__.service.AuthService."""
        source = """\
def example():
    AuthService.login(user)
"""
        func_node = _parse_function(source)
        extractor = CallExtractor(
            import_map={"AuthService": "__rel1__.service.AuthService"},
            current_class=None,
        )
        calls = extractor.extract_calls(func_node)

        assert len(calls) == 1
        assert calls[0].resolved_name == "__rel1__.service.AuthService.login"
        assert calls[0].resolution == "import"


# ---------------------------------------------------------------------------
# CallExtractor: noise filter — 3 tests
# ---------------------------------------------------------------------------
class TestNoiseFilter:
    """Test noise filtering for builtins, logging, and attribute calls."""

    def test_builtins_skipped(self):
        """Builtin calls (print, len, range, sorted, etc.) are not extracted."""
        builtins_to_test = ["print", "len", "range", "sorted", "isinstance", "type", "str", "int"]
        calls_src = "\n    ".join(f"{b}(x)" for b in builtins_to_test)
        source = f"""\
def example():
    {calls_src}
    actual_function()
"""
        func_node = _parse_function(source)
        extractor = CallExtractor(import_map={}, current_class=None)
        calls = extractor.extract_calls(func_node)

        names = _names(calls)
        for b in builtins_to_test:
            assert b not in names, f"{b} should be filtered as noise"
        assert "actual_function" in names

    def test_stdlib_logging_skipped(self):
        """logger.info(), logger.warning() skipped because logger is in NOISE_OBJECTS."""
        source = """\
def example():
    logger.info("starting")
    logger.warning("problem")
    logger.error("failure")
    real_service.process()
"""
        func_node = _parse_function(source)
        extractor = CallExtractor(import_map={}, current_class=None)
        calls = extractor.extract_calls(func_node)

        names = _names(calls)
        assert "logger.info" not in names
        assert "logger.warning" not in names
        assert "logger.error" not in names
        assert "real_service.process" in names

    def test_noise_attribute_calls_filtered(self):
        """.items(), .strip() are filtered from unresolved attribute calls."""
        source = """\
def example():
    data.items()
    name.strip()
    result.append(x)
    service.execute()
"""
        func_node = _parse_function(source)
        # No import_map entries for data, name, result, service => all unresolved
        extractor = CallExtractor(import_map={}, current_class=None)
        calls = extractor.extract_calls(func_node)

        names = _names(calls)
        # items, strip, append are NOISE_ATTRIBUTE_CALLS => filtered on unresolved
        assert "data.items" not in names
        assert "name.strip" not in names
        assert "result.append" not in names
        # service.execute is not in noise lists => kept
        assert "service.execute" in names
