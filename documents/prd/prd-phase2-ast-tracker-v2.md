# PRD: Phase 2 â€” AST Tracker (v2 â€” Review-Corrected)

**Project**: claude-mem-lite (fork of claude-mem, Python rewrite)
**Phase**: 2 of 9
**Status**: Ready for implementation
**Dependencies**: Phase 0 (Storage Layer) â€” SQLiteStore, `function_map` + `call_graph` tables, Pydantic models
**Parallel with**: Phase 1 (Hooks + Capture) â€” no dependency, but integrates when both complete
**Estimated effort**: 2 sessions (~8-12 hours)
**Python**: 3.14.3 (latest stable)

---

## 1. Purpose & Context

### 1.1 What this phase delivers
Deterministic, testable code intelligence for Python files. No AI, no network calls, no worker dependency. Pure `ast` stdlib.

Specifically:
- **Function extraction**: signatures, decorators, docstrings, line ranges, body hashes for every function/method/class in a Python file
- **Call graph extraction**: best-effort mapping of which functions call which, with confidence-typed edges
- **Change detection**: compare current file state against previous snapshot to classify functions as new/modified/deleted/unchanged
- **Mermaid diagram generation**: per-file and cross-file call graphs for CLI reports (Phase 8)
- **A library module** with zero hook/worker coupling â€” can be called from anywhere

### 1.2 What this phase does NOT deliver
- **Hook integration** â€” The AST tracker is a pure library. Wiring it into Phase 1's `capture.py` hook happens as a follow-up integration task once both phases are complete.
- **Cross-file import resolution** â€” We resolve imports within a single file. Full cross-file resolution (following `from foo.bar import Baz` to the actual `Baz` class definition) is deferred â€” it requires a project-wide index that doesn't exist yet.
- **Non-Python languages** â€” `ast` stdlib only parses Python. tree-sitter support for JS/TS/etc. is a Phase 9 consideration.
- **Self-healing call graph** â€” Confidence evolution via observation confirmation is Phase 6.

### 1.3 Why it matters for context injection
The function map is Layer 2 of the context injection budget (~500 tokens). It gives Claude a lightweight overview of what exists in a file without reading the full source. Example output for context injection:

```
auth/service.py (modified 2h ago):
  class AuthService:
    authenticate(email, password) -> Token  [L12-L45] @router.post MODIFIED
    refresh_token(token) -> Token           [L47-L62] @require_auth NEW
    _validate_password(plain, hashed) -> bool [L64-L71] UNCHANGED
```

~150 tokens for an entire file. Claude can decide whether to read the actual code.

### 1.4 Relationship to claude-mem (original)
claude-mem has no AST tracking. This is entirely new capability. The closest analog is how Claude Code's built-in indexing works, but we persist function maps across sessions and track changes over time.

### 1.5 Python 3.14 `ast` module changes

| Feature | Impact on Phase 2 |
|---------|-------------------|
| `ast.compare(node1, node2, compare_attributes=False)` | **New in 3.14**. Structural AST comparison ignoring line numbers. Useful for in-memory tests but doesn't replace body_hash (we need persistence). |
| `copy.replace()` for AST nodes | Not needed â€” we don't transform ASTs. |
| Removed: `ast.Num`, `ast.Str`, `ast.Bytes`, `ast.NameConstant`, `ast.Ellipsis` | Our code doesn't reference these. Use `ast.Constant` only. Custom `visit_Num` etc. on NodeVisitor subclasses no longer called â€” use `visit_Constant`. |
| `ast` import time improved | Minor startup benefit for hooks. |
| Deferred annotations (PEP 649) | Runtime-only change. `ast.parse()` still sees full annotation nodes in the AST â€” no impact on signature extraction. |

---

## 2. Technical Specification

### 2.1 Module Structure

```
src/claude_mem_lite/ast_tracker/
â”œâ”€â”€ __init__.py          # Public API: scan_file(), scan_files()
â”œâ”€â”€ extractor.py         # FunctionExtractor (ast.NodeVisitor)
â”œâ”€â”€ call_graph.py        # CallExtractor (call resolution + noise filtering)
â”œâ”€â”€ diff.py              # compare_snapshots() â€” change detection
â””â”€â”€ mermaid.py           # generate_mermaid() â€” diagram output
```

**Design decision: Extractor and CallExtractor are separate classes.**

The architecture spec combines extraction and call resolution in one `FunctionExtractor` class (~200 lines). Splitting them improves testability:
- `FunctionExtractor` focuses on function/class extraction, signatures, hashing
- `CallExtractor` focuses on call resolution, import maps, noise filtering
- Each can be tested independently with different fixtures
- `FunctionExtractor` calls `CallExtractor.extract_calls(func_node)` for each function

### 2.2 Data Types

These are internal dataclasses used within the AST tracker. They map to Phase 0's Pydantic models (`FunctionMapEntry`, `CallGraphEdge`) when persisted to SQLite.

```python
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class FunctionInfo:
    """Extracted function/method/class metadata."""
    qualified_name: str          # "AuthService.authenticate"
    kind: str                    # function | method | async_function | async_method | class
    parent_class: str | None
    signature: str               # "authenticate(self, email: str, password: str) -> Token"
    decorators: list[str]        # ["@router.post('/login')", "@require_auth"]
    docstring: str | None        # First line only
    line_start: int
    line_end: int
    body_hash: str               # MD5 of ast.dump(node), full 32 hex chars
    calls: list["CallInfo"] = field(default_factory=list)

@dataclass(frozen=True, slots=True)
class CallInfo:
    """A function call site within a function body."""
    raw_name: str                # As written: "self.validate()", "db.query()"
    resolved_name: str | None    # Best guess: "AuthService.validate" or None
    resolution: str              # direct | self_method | import | unresolved
    line_number: int

@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """Complete AST extraction result for a single file."""
    file_path: str
    functions: list[FunctionInfo]
    import_map: dict[str, str]   # local_name -> qualified_name
    parse_error: str | None = None  # SyntaxError message if file couldn't parse

@dataclass(frozen=True, slots=True)
class FunctionDiff:
    """Change classification for a single function."""
    qualified_name: str
    change_type: str             # new | modified | deleted | unchanged
    current: FunctionInfo | None  # None if deleted
    previous_hash: str | None    # None if new
```

**Design notes:**
1. **`frozen=True, slots=True`**: These are value objects â€” immutable after creation, memory-efficient. Hashable for set operations in diff.
2. **`body_hash` is full 32 hex chars**: Phase 0 PRD specifies this. The architecture spec truncated to 12 chars â€” corrected here.
3. **`docstring` first line only**: Full docstrings waste context tokens. First line is usually the most informative.

### 2.3 `extractor.py` â€” FunctionExtractor

```python
class FunctionExtractor(ast.NodeVisitor):
    """Extract function/method/class definitions from Python source.
    
    Usage:
        extractor = FunctionExtractor(source, file_path)
        snapshot = extractor.extract()
    """
    
    def __init__(self, source: str, file_path: str):
        self.source = source
        self.file_path = file_path
        self.functions: list[FunctionInfo] = []
        self.import_map: dict[str, str] = {}
        self._context_stack: list[str] = []     # Nesting: ["ClassName", "method"]
        self._current_class: str | None = None
    
    def extract(self) -> FileSnapshot:
        """Parse source and extract all functions. Returns FileSnapshot."""
        try:
            tree = ast.parse(self.source)
        except SyntaxError as e:
            return FileSnapshot(
                file_path=self.file_path,
                functions=[],
                import_map={},
                parse_error=str(e),
            )
        self._build_import_map(tree)
        self.visit(tree)
        return FileSnapshot(
            file_path=self.file_path,
            functions=self.functions,
            import_map=self.import_map,
        )
```

**Key methods:**

| Method | Purpose |
|--------|---------|
| `extract()` | Entry point. Returns `FileSnapshot` (or snapshot with `parse_error` on SyntaxError) |
| `_build_import_map(tree)` | Walk tree once for `Import`/`ImportFrom`, build `local_name â†’ qualified_name` map |
| `visit_ClassDef(node)` | Push class onto context stack, visit children, pop |
| `visit_FunctionDef(node)` | Delegate to `_process_function(node, is_async=False)` |
| `visit_AsyncFunctionDef(node)` | Delegate to `_process_function(node, is_async=True)` |
| `_process_function(node, is_async)` | Extract all metadata, build `FunctionInfo`, visit nested children |
| `_build_signature(node)` | Build readable signature string with type annotations and defaults |
| `_compute_body_hash(node)` | `hashlib.md5(ast.dump(node).encode()).hexdigest()` |

**`_build_signature` handles:**
- Positional args with type annotations and defaults
- `*args` and `**kwargs` with annotations
- Keyword-only args (between `*` and `**kwargs`)
- Return type annotation
- Strips `self` and `cls` from display â€” implicit from context (indented under class, `kind` field distinguishes regular/classmethod). Saves ~6 tokens per method.

**`_compute_body_hash` details:**

```python
def _compute_body_hash(self, node: ast.AST) -> str:
    """Hash the AST structure, ignoring line numbers and formatting.
    
    ast.dump() with include_attributes=False (the default) excludes
    line numbers and column offsets. This means:
    - Reformatting (black, ruff format) â†’ same hash âœ“
    - Moving function up/down in file â†’ same hash âœ“
    - Changing function body logic â†’ different hash âœ“
    - Adding/removing comments â†’ same hash âœ“ (comments aren't in AST)
    - Changing string content â†’ different hash âœ“
    """
    return hashlib.md5(ast.dump(node).encode()).hexdigest()
```

**Correction to architecture spec**: The architecture used `ast.dump(node).encode()` without noting that `include_attributes` defaults to `False`. This is actually correct behavior â€” line numbers are excluded by default. But worth being explicit about.

### 2.4 `call_graph.py` â€” CallExtractor

```python
class CallExtractor:
    """Extract and resolve function calls within a function body.
    
    Resolution types:
    - direct: Simple name call, resolved via local scope. foo() â†’ foo
    - self_method: self.method() â†’ ClassName.method  
    - import: Resolved via import map. db.query() â†’ sqlalchemy.orm.Session.query
    - unresolved: Can't determine target. a.b.c.method() â†’ None
    """
    
    def __init__(self, import_map: dict[str, str], current_class: str | None = None):
        self.import_map = import_map
        self.current_class = current_class
    
    def extract_calls(self, func_node: ast.AST) -> list[CallInfo]:
        """Extract all non-noise function calls from a function body."""
```

**Noise filtering â€” builtins and stdlib to skip:**

The architecture spec hardcodes a small list. This should be more comprehensive:

```python
# Functions that are never interesting for call graph purposes
NOISE_CALLABLES: frozenset[str] = frozenset({
    # Builtins
    "print", "len", "range", "enumerate", "zip", "map", "filter",
    "isinstance", "issubclass", "hasattr", "getattr", "setattr", "delattr",
    "super", "type", "id", "hash", "repr", "str", "int", "float", "bool",
    "list", "dict", "set", "tuple", "frozenset", "bytes", "bytearray",
    "sorted", "reversed", "min", "max", "sum", "any", "all", "abs",
    "round", "divmod", "pow", "chr", "ord", "hex", "oct", "bin",
    "format", "vars", "dir", "callable", "iter", "next", "open",
    "input", "staticmethod", "classmethod", "property",
    # Common stdlib that's noise
    "logging.getLogger", "logging.debug", "logging.info", 
    "logging.warning", "logging.error", "logging.critical",
})

# Attribute calls on these objects are always noise
NOISE_OBJECTS: frozenset[str] = frozenset({
    "logger", "log", "logging", "os", "sys", "re", "json",
    "math", "copy", "functools", "itertools", "collections",
    "pathlib", "Path",
})

# Method names on unknown objects that carry no architectural info.
# These are filtered from unresolved edges only â€” if the call resolves
# to a known target, it's kept regardless.
NOISE_ATTRIBUTE_CALLS: frozenset[str] = frozenset({
    # dict/list/set methods
    "items", "keys", "values", "get", "pop", "update", "append",
    "extend", "insert", "remove", "clear", "copy", "sort",
    "add", "discard",
    # string methods
    "strip", "lstrip", "rstrip", "split", "rsplit", "join",
    "replace", "format", "encode", "decode", "lower", "upper",
    "startswith", "endswith", "find", "rfind", "count",
    # general object methods
    "items", "keys", "values", "__init__", "__str__", "__repr__",
    "__enter__", "__exit__", "__aenter__", "__aexit__",
})
```

**Unresolved edge persistence policy**: Unresolved call edges are persisted to the `call_graph` table with `confidence=0.5`, **except** when the method name is in `NOISE_ATTRIBUTE_CALLS`. This keeps architecturally interesting unresolved calls (`db.execute`, `cache.get`, `client.post`) while filtering builtin-type method noise (`.items()`, `.strip()`, `.append()`). Phase 6's self-healing can later confirm the kept unresolved edges by matching them against compressed observations.

**Resolution accuracy (from architecture spec â€” validated):**

| Pattern | Accuracy | Example |
|---------|----------|---------|
| Function/class definitions | ~100% | `def foo():` â†’ always found |
| Direct calls: `foo()` | ~95% | Resolved via imports or local scope |
| `self.method()` | ~90% | Resolved to `ClassName.method` |
| Imported calls: `module.func()` | ~80% | Resolved via import map |
| Variable calls: `x = foo; x()` | ~0% | Not attempted â€” would need type inference |
| Dynamic: `getattr()`, `**dispatch` | ~0% | Not attempted â€” not worth it |

### 2.5 `diff.py` â€” Change Detection

```python
def compare_snapshots(
    current: FileSnapshot,
    previous: list[FunctionInfo],  # Previous snapshot from DB
) -> list[FunctionDiff]:
    """Compare current extraction against previous snapshot.
    
    Classification:
    - new: In current, not in previous (by qualified_name)
    - deleted: In previous, not in current
    - modified: In both, body_hash differs
    - unchanged: In both, body_hash matches
    """
```

**Algorithm:**
1. Build `dict[qualified_name, FunctionInfo]` for both current and previous
2. For each name in current: if not in previous â†’ `new`, elif hash differs â†’ `modified`, else â†’ `unchanged`
3. For each name in previous but not in current â†’ `deleted`

**Edge case: function rename.** If `foo()` is renamed to `bar()`, we see `foo` deleted and `bar` new. This is correct â€” we don't attempt rename detection. Git handles rename tracking. We track structural identity by qualified name.

**Edge case: class rename.** If `AuthService` â†’ `AuthHandler`, all methods change qualified names. We see the entire old class deleted and new class created. Again, correct behavior â€” no rename inference.

### 2.6 `mermaid.py` â€” Diagram Generation

```python
def generate_mermaid(
    functions: list[FunctionInfo],
    file_path: str | None = None,
    show_all: bool = False,
    change_types: dict[str, str] | None = None,  # qualified_name â†’ change_type
) -> str:
    """Generate Mermaid graph from function call data.
    
    Default scoping: includes changed functions (new/modified/deleted) plus
    any unchanged functions that are direct call targets of changed functions.
    This gives the relevant subgraph without dumping the entire file.
    
    With show_all=True: includes every function in the file.
    
    Returns a string like:
    ```mermaid
    graph TD
        subgraph auth/service.py
            A[AuthService.authenticate] --> B[AuthService._validate_password]
            A --> C[Token.create]
        end
        style A fill:#ffd
    ```
    """
```

**Default scoping algorithm:**
1. Collect all functions with `change_type` in (`new`, `modified`, `deleted`) â†’ "changed set"
2. For each function in the changed set, collect all call targets â†’ "dependency set"
3. Render: changed set âˆª (dependency set âˆ© all functions in file)
4. Changed functions get colored styles, dependency-only functions render unstyled (implicitly unchanged)

**Example**: File has 30 functions, 3 changed. Those 3 call 7 unchanged functions. Diagram shows 10 nodes, not 30. The `--all` CLI flag sets `show_all=True`.

**Style mapping:**
- `new` â†’ green fill (`#dfd`)
- `modified` â†’ yellow fill (`#ffd`)
- `deleted` â†’ red fill (`#fdd`)
- `unchanged` (included as dependency) â†’ no special style
- `unresolved` edges â†’ dashed lines (`-.->`)
- `resolved` edges â†’ solid lines (`-->`)

### 2.7 `__init__.py` â€” Public API

```python
def scan_file(
    file_path: str,
    source: str | None = None,
) -> FileSnapshot:
    """Scan a single Python file. Reads from disk if source not provided.
    
    Returns FileSnapshot with functions, import_map, and optional parse_error.
    """

def scan_files(file_paths: list[str]) -> list[FileSnapshot]:
    """Scan multiple Python files. Skips non-.py files silently."""

def diff_file(
    current: FileSnapshot,
    previous: list[FunctionInfo],
) -> list[FunctionDiff]:
    """Compare current snapshot against previous. Thin wrapper around diff.compare_snapshots."""
```

---

## 3. Integration Plan

### 3.1 Hook Integration (post-Phase 1+2 completion)

Once both Phase 1 and Phase 2 are complete, the `capture.py` hook gets an AST scanning step:

```python
# In capture.py, after enqueue() â€” only for file-modifying tools
if tool_name in ("Write", "Edit", "MultiEdit") and files_touched:
    from claude_mem_lite.ast_tracker import scan_file
    
    for fp in files_touched:
        if fp.endswith(".py"):
            snapshot = scan_file(fp)
            if snapshot.parse_error:
                # File has syntax error (mid-edit) â€” skip, log warning
                continue
            # Load previous, diff, store new snapshot
            _persist_snapshot(store, session_id, snapshot)
```

**Why only Write/Edit/MultiEdit**: Read doesn't change files. Bash *could* write files, but we can't reliably extract which files from bash commands. Glob/Grep don't modify anything.

**Why in the hook, not deferred**: `ast.parse()` is <10ms per file. The hook already takes ~5ms for the pending_queue INSERT. Adding AST scanning for 1-3 Python files adds 10-30ms â€” well within the 10-second timeout. Deferring to the worker (Phase 3) adds complexity with no benefit.

**Latency budget:**

| Step | Time | Notes |
|------|------|-------|
| Read file from disk | <1ms | Typical Python files are <100KB |
| `ast.parse()` | 2-8ms | Scales with file size, <10ms for files under 50KB |
| Extract functions + calls | 1-3ms | Single-pass visitor |
| Load previous snapshot | 1-3ms | SQLite indexed query |
| Diff + store new snapshot | 1-3ms | INSERT/UPDATE operations |
| **Total per file** | **5-18ms** | |
| **Per hook (1-3 files)** | **5-54ms** | Worst case: 3 large files |

### 3.2 Database Usage (Phase 0 Tables)

**`function_map` table â€” one row per function per snapshot:**

| Column | Source |
|--------|--------|
| `id` | Generated UUID |
| `session_id` | From hook event |
| `file_path` | From `FileSnapshot.file_path` |
| `qualified_name` | From `FunctionInfo.qualified_name` |
| `kind` | From `FunctionInfo.kind` |
| `signature` | From `FunctionInfo.signature` |
| `docstring` | From `FunctionInfo.docstring` |
| `body_hash` | From `FunctionInfo.body_hash` |
| `decorators` | JSON array from `FunctionInfo.decorators` |
| `change_type` | From `FunctionDiff.change_type` |
| `updated_at` | Current timestamp |

**`call_graph` table â€” one row per call edge:**

| Column | Source |
|--------|--------|
| `id` | Generated UUID |
| `caller_file` | File containing the calling function |
| `caller_function` | `FunctionInfo.qualified_name` |
| `callee_file` | Same file (cross-file resolution not in Phase 2) |
| `callee_function` | `CallInfo.resolved_name` or `CallInfo.raw_name` |
| `resolution` | `CallInfo.resolution` |
| `confidence` | 1.0 for resolved, 0.5 for unresolved |
| `times_confirmed` | 0 (Phase 6 increments this) |
| `source` | `"ast"` |
| `session_id` | From hook event |
| `created_at` | Current timestamp |

### 3.3 Persistence Strategy

**Snapshot storage model**: We store the *latest* snapshot per file, not every historical version. The `function_map` table gets UPSERTed:

```python
def _persist_snapshot(store, session_id: str, snapshot: FileSnapshot, diffs: list[FunctionDiff]):
    """Store function map and call graph for a file."""
    
    # 1. Delete previous function_map entries for this file
    store.execute(
        "DELETE FROM function_map WHERE file_path = ?",
        (snapshot.file_path,)
    )
    
    # 2. Insert current functions with change_type from diff
    change_map = {d.qualified_name: d.change_type for d in diffs}
    for func in snapshot.functions:
        store.create_function_entry(
            file_path=snapshot.file_path,
            session_id=session_id,
            qualified_name=func.qualified_name,
            kind=func.kind,
            signature=func.signature,
            docstring=func.docstring,
            body_hash=func.body_hash,
            decorators=func.decorators,
            change_type=change_map.get(func.qualified_name, "new"),
        )
    
    # 3. Replace call graph edges for functions in this file
    store.execute(
        "DELETE FROM call_graph WHERE caller_file = ?",
        (snapshot.file_path,)
    )
    for func in snapshot.functions:
        for call in func.calls:
            # Skip unresolved edges for builtin-type methods (.items, .strip, etc.)
            if call.resolution == "unresolved" and _is_noise_attribute(call.raw_name):
                continue
            store.add_call_edge(
                caller_file=snapshot.file_path,
                caller_function=func.qualified_name,
                callee_file=snapshot.file_path,  # Same file â€” cross-file deferred
                callee_function=call.resolved_name or call.raw_name,
                resolution=call.resolution,
                confidence=1.0 if call.resolution != "unresolved" else 0.5,
                session_id=session_id,
            )
```

**Why DELETE + INSERT instead of UPSERT**: function_map rows don't have a stable unique key across snapshots. A function might be renamed, causing a new qualified_name. DELETE all rows for the file, then INSERT the current state, is simpler and correct.

**Historical change tracking**: The `change_type` column records what changed *in this snapshot*. Combined with `session_id`, you can reconstruct the evolution timeline: "In session X, `authenticate` was modified. In session Y, `refresh_token` was added."

---

## 4. Corrections to Architecture Spec

| Item | Architecture Spec | Corrected (Phase 2 PRD) | Rationale |
|------|-------------------|--------------------------|-----------|
| `body_hash` length | `hexdigest()[:12]` (12 chars) | `hexdigest()` (full 32 chars) | Phase 0 PRD specifies full hash. 32 chars is negligible in a TEXT column. |
| Class design | Single `FunctionExtractor` (~200 lines) | Split: `FunctionExtractor` + `CallExtractor` | Better testability, single responsibility |
| Noise filter list | 16 builtins hardcoded | Comprehensive `NOISE_CALLABLES` + `NOISE_OBJECTS` frozensets | Architecture list misses common stdlib: logging, sorted, min/max, etc. |
| `_build_signature` | Includes `self`/`cls` in output | Strips `self`/`cls` | They're noise in context injection â€” Claude knows methods have self |
| `callee_name` format | `"self.validate()"` with parens | `"self.validate"` without parens | Parens are display formatting, not part of the name. Consistent with `qualified_name`. |
| Cross-file resolution | Implied as Phase 2 scope | Explicitly deferred | Requires project-wide import index â€” too complex for initial implementation |
| Persistence model | "Store new snapshot in function_map" | DELETE + INSERT per file | No stable UPSERT key when functions are renamed/moved |
| `ast.Num`/`ast.Str` usage | Not explicitly addressed | Confirmed removed in 3.14 | Our code uses `ast.Constant` â€” no impact, but documented |
| `ast.compare()` | Not mentioned (didn't exist in original spec) | Available in 3.14, useful for tests | Can replace hash comparison in unit tests for cleaner assertions |
| Unresolved edge noise | Not addressed â€” all unresolved edges implicitly kept | `NOISE_ATTRIBUTE_CALLS` filter for builtin-type methods | Keeps architecturally interesting unresolved calls, filters ~30 common dict/list/string methods |
| Mermaid scoping | "Per-file and cross-file call graphs" (no scoping detail) | Changed functions + direct call targets only (default) | Avoids dumping 30-node full file graphs. `--all` flag for completeness. |

---

## 5. Edge Cases & Error Handling

### 5.1 SyntaxError (file mid-edit)
Claude Code's Edit tool may produce temporarily invalid Python (e.g., partial edit applied). `ast.parse()` raises `SyntaxError`.

**Handling**: Return `FileSnapshot(parse_error=str(e))`. The hook logs a warning and skips AST scanning for this file. The previous snapshot in the database remains unchanged. The file will be re-scanned on the next successful edit.

### 5.2 Very large files
Files with thousands of functions (generated code, protobuf stubs) could produce expensive snapshots.

**Handling**: Configurable limit (default: 500 functions per file). If exceeded, log a warning and skip the file. This is a safety valve, not expected in normal use.

### 5.3 Encoding issues
Non-UTF-8 Python files may fail `ast.parse()`.

**Handling**: Read with `encoding="utf-8", errors="replace"`. If the file has a PEP 263 encoding declaration, respect it.

### 5.4 Binary/non-Python files
Hook may receive `.py` files that aren't actually Python (e.g., empty, binary-ish).

**Handling**: `ast.parse()` will raise `SyntaxError`. Same handling as 5.1.

### 5.5 Nested functions and closures
```python
def outer():
    def inner():
        def deeply_nested():
            pass
```

**Handling**: All three are extracted. Qualified names: `outer`, `outer.inner`, `outer.inner.deeply_nested`. The context stack tracks nesting depth correctly.

### 5.6 Decorated classes with methods
```python
@dataclass
class Config:
    name: str
    
    def validate(self) -> bool:
        ...
```

**Handling**: `Config` extracted as kind=`class`. `Config.validate` extracted as kind=`method`. Decorator `@dataclass` captured in decorators list.

### 5.7 `__init__.py` files with re-exports
```python
from .service import AuthService
from .routes import router
```

**Handling**: Import map captures `AuthService â†’ .service.AuthService`. No functions to extract (typically). This is correct â€” `__init__.py` is a re-export file.

### 5.8 Star imports
```python
from os.path import *
```

**Handling**: Can't resolve what `*` brings into scope without actually importing the module. These names go to `unresolved` resolution. This is an accepted limitation.

---

## 6. Performance Targets

| Operation | Target | Rationale |
|-----------|--------|-----------|
| `ast.parse()` for typical file (<1000 LOC) | <5ms | stdlib C implementation |
| `ast.parse()` for large file (5000+ LOC) | <15ms | Linear with file size |
| Full extraction (parse + extract + calls) | <10ms typical | Single-pass visitor |
| Diff (compare two snapshots, <100 functions) | <1ms | Dict lookups |
| Mermaid generation (<50 functions) | <1ms | String concatenation |
| `NOISE_CALLABLES` lookup | O(1) | frozenset |

---

## 7. Test Plan

### 7.1 Test fixtures approach

Rather than testing against "real Python files" (brittle, hard to maintain), tests use inline source strings:

```python
SOURCE_BASIC = '''
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}"
'''

SOURCE_CLASS = '''
from datetime import datetime

class UserService:
    def __init__(self, db):
        self.db = db
    
    async def get_user(self, user_id: int) -> dict:
        return await self.db.fetch_one(user_id)
    
    def _validate(self, data: dict) -> bool:
        return isinstance(data, dict)
'''
```

### 7.2 Test categories

| Category | Tests | What it validates |
|----------|-------|-------------------|
| **Extractor: basics** | 5 | Simple function, multiple functions, empty file, syntax error, non-Python content |
| **Extractor: signatures** | 6 | Positional args, defaults, `*args`/`**kwargs`, keyword-only, return type, self-stripping |
| **Extractor: classes** | 4 | Methods, async methods, class decorators, nested classes |
| **Extractor: nesting** | 3 | Nested functions, closure, function inside method |
| **Extractor: decorators** | 2 | Simple decorators, decorator with arguments |
| **Extractor: docstrings** | 2 | Single-line, multi-line (first line extracted) |
| **Extractor: body_hash** | 4 | Same logic = same hash, different logic = different hash, whitespace-only change = same hash, comment-only change = same hash |
| **CallExtractor: resolution** | 6 | Direct call, self.method, imported call, chained attribute, unresolved, noise filtered |
| **CallExtractor: import map** | 3 | `import x`, `from x import y`, `from x import y as z` |
| **CallExtractor: noise filter** | 3 | Builtins skipped, stdlib logging skipped, NOISE_ATTRIBUTE_CALLS filtered from unresolved edges |
| **Diff: change types** | 5 | New function, modified function, deleted function, unchanged function, mixed changes |
| **Diff: edge cases** | 2 | Empty previous (all new), empty current (all deleted) |
| **Mermaid: output** | 4 | Single file graph, change type styling, unresolved edge dashing, default scoping (only changed + direct dependencies shown) |
| **Integration: scan_file** | 2 | File from disk, source provided directly |
| **Total** | **51** |

### 7.3 Test infrastructure

```python
# conftest.py additions
@pytest.fixture
def extract(self):
    """Helper to extract functions from inline source."""
    def _extract(source: str, file_path: str = "test.py") -> FileSnapshot:
        return FunctionExtractor(source, file_path).extract()
    return _extract

@pytest.fixture
def extract_calls(self):
    """Helper to extract calls from a function within inline source."""
    def _extract_calls(source: str) -> list[CallInfo]:
        snapshot = FunctionExtractor(source, "test.py").extract()
        assert len(snapshot.functions) >= 1
        return snapshot.functions[0].calls
    return _extract_calls
```

### 7.4 Key test cases (detailed)

**body_hash stability â€” whitespace and comments:**
```python
def test_hash_ignores_whitespace():
    source_a = "def f():\n    return 1"
    source_b = "def f():\n\n    return 1\n\n"
    # Should produce same hash â€” ast.dump ignores whitespace
    
def test_hash_ignores_comments():
    source_a = "def f():\n    return 1"
    source_b = "def f():\n    # Important comment\n    return 1"
    # Should produce same hash â€” comments aren't in AST
    
def test_hash_changes_on_logic():
    source_a = "def f():\n    return 1"
    source_b = "def f():\n    return 2"
    # Must produce different hash
    
def test_hash_ignores_line_position():
    source_a = "def f():\n    return 1"
    source_b = "\n\n\ndef f():\n    return 1"
    # Should produce same hash â€” ast.dump default excludes line numbers
```

**Call resolution:**
```python
def test_self_method_resolution():
    source = '''
class Foo:
    def bar(self):
        self.baz()
'''
    # baz() should resolve to "Foo.baz" with resolution="self_method"

def test_import_resolution():
    source = '''
from auth.service import AuthService

def create():
    AuthService.authenticate()
'''
    # authenticate() should resolve to "auth.service.AuthService.authenticate"
    # with resolution="import"
```

---

## 8. Dependencies

### 8.1 Phase 2 runtime dependencies (additions to Phase 0)

**None.** Everything uses stdlib:
- `ast` â€” parsing and visiting
- `hashlib` â€” MD5 body hash
- `dataclasses` â€” internal data types

### 8.2 Phase 2 dev dependencies (additions to Phase 0)

**None** beyond pytest and ruff (already in Phase 0).

---

## 9. Acceptance Criteria

Phase 2 is complete when:

- [ ] All 51 tests pass (pytest, <5s total runtime)
- [ ] `ruff check` and `ruff format --check` pass with zero warnings
- [ ] `FunctionExtractor` correctly extracts functions, methods, async variants, classes, nested functions
- [ ] Signatures include type annotations, defaults, `*args`/`**kwargs`, and strip `self`/`cls`
- [ ] `body_hash` is stable across whitespace/comment changes, changes on logic changes
- [ ] `CallExtractor` resolves direct, self_method, and import calls correctly
- [ ] Noise filtering excludes builtins, common stdlib calls, and `NOISE_ATTRIBUTE_CALLS` on unresolved edges
- [ ] Unresolved edges persisted with `confidence=0.5` except when method name is in `NOISE_ATTRIBUTE_CALLS`
- [ ] `compare_snapshots` correctly classifies new/modified/deleted/unchanged
- [ ] `generate_mermaid` default scoping shows changed functions + their direct call targets (not full file)
- [ ] `generate_mermaid` produces valid Mermaid syntax (verifiable at mermaid.live)
- [ ] SyntaxError handling returns `FileSnapshot` with `parse_error`, doesn't crash
- [ ] No runtime dependencies beyond Phase 0 (`pydantic` only â€” but Phase 2 doesn't even use it internally)

---

## 10. Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `ast.dump()` output format changes across Python versions | Very Low | Medium | We target 3.14 only. Hash comparisons are session-local â€” cross-version migration not needed. |
| Noise filter too aggressive (filters real project calls) | Medium | Low | `NOISE_OBJECTS` is conservative. Users would see missing edges, not wrong edges. Easy to add exceptions. |
| Nested class/function depth causes context stack bugs | Low | Medium | Tested explicitly with 3-deep nesting. Stack is push/pop symmetric. |
| Large generated files (protobuf, etc.) cause slow hooks | Low | Low | 500-function limit per file. Configurable. |
| `self`/`cls` stripping in signatures may confuse some readers | Low | Low | Decision made: strip both. `kind` field disambiguates method types. Context injection token savings (~90/file) outweigh readability concern. |

---

## 11. Resolved Design Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| **Strip `self`/`cls` from signatures?** | **Strip both** | Implicit from context â€” methods are indented under class, `kind` field distinguishes regular/classmethod. Saves ~6 tokens per method, ~90 tokens for a 15-method file. Compounding savings across multiple files in Layer 2 context budget (~500 tokens). |
| **Persist unresolved call graph edges?** | **Yes, with `NOISE_ATTRIBUTE_CALLS` filter** | Unresolved edges persisted at `confidence=0.5` to enable Phase 6 self-healing (confirmation via observation matching). But builtin-type method noise (`.items()`, `.strip()`, `.append()`) filtered via `NOISE_ATTRIBUTE_CALLS` frozenset. Keeps architecturally interesting calls (`db.execute`, `cache.get`, `client.post`) while eliminating ~30 common dict/list/string methods. |
| **Mermaid: include unchanged functions?** | **Include changed + direct call targets only** | Changed functions plus any unchanged function that is a direct call target of a changed function. Gives the relevant subgraph (~10 nodes) without dumping full file (~30 nodes). Unstyled nodes are implicitly unchanged. `--all` flag for complete graph. |
