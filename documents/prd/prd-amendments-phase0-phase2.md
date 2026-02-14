# PRD Amendments: Phase 0 + Phase 2 (Post-Review)

**Date**: 2026-02-08
**Triggered by**: External review of Phase 2 PRD (5 comments)
**Affects**: Phase 0 v2 (schema), Phase 2 v2 (AST tracker)

---

## Amendment 1: Drop FK Constraints on `function_map` and `call_graph`

**Severity**: Critical
**Affects**: Phase 0 schema (migration v1 DDL)
**Precedent**: Phase 1 already identified and resolved the identical issue for `pending_queue`

### Problem

`function_map.session_id` and `call_graph.session_id` both define `REFERENCES sessions(id)`. The integration code in `capture.py` (Section 3.1 of Phase 2) writes to these tables during `PostToolUse` hooks. Claude Code hook events fire in parallel â€” `PostToolUse` can fire before `SessionStart` completes, meaning the `sessions` row doesn't exist yet.

Result: `IntegrityError` on INSERT. The AST data is silently lost or the hook crashes.

Phase 1 already solved this for `pending_queue` (Phase 1 PRD, Section 4.4). The same race condition applies to all tables written from hook context.

### Phase 0 Schema Change

In migration v1, update the `function_map` and `call_graph` DDL:

```sql
-- function_map: BEFORE
CREATE TABLE function_map (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    ...
);

-- function_map: AFTER
CREATE TABLE function_map (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,  -- No FK â€” hooks fire in parallel, PostToolUse may precede SessionStart
    ...
);

-- call_graph: BEFORE
CREATE TABLE call_graph (
    id TEXT PRIMARY KEY,
    ...
    session_id TEXT NOT NULL REFERENCES sessions(id),
    ...
);

-- call_graph: AFTER
CREATE TABLE call_graph (
    id TEXT PRIMARY KEY,
    ...
    session_id TEXT NOT NULL,  -- No FK â€” hooks fire in parallel, PostToolUse may precede SessionStart
    ...
);
```

### Phase 0 Changelog Addition

Add row to the Phase 0 v2 â†’ v3 changelog:

| Item | v2 (incorrect) | v3 (corrected) |
|------|----------------|----------------|
| `function_map` / `call_graph` FK | `session_id REFERENCES sessions(id)` | No FK â€” same parallel-hook race as `pending_queue` (Phase 1 precedent) |

### Phase 2 Integration Plan Addition

Add note to Section 3.2:

> **Note**: `function_map.session_id` and `call_graph.session_id` have no FK constraint (Phase 0 amendment, same rationale as `pending_queue`). Session reconciliation happens in Phase 3's processing pipeline. Orphan `session_id` values are expected during normal operation and resolved when Phase 3 processes the queue.

### Design Rationale

- Removing the FK is cleaner than `INSERT OR IGNORE` placeholder sessions
- Consistent with the `pending_queue` precedent from Phase 1
- Data integrity is enforced by Phase 3's processing pipeline, not by schema constraints on a race-prone write path
- All three hook-written tables (`pending_queue`, `function_map`, `call_graph`) now follow the same pattern

---

## Amendment 2: Mermaid Syntax Sanitization

**Severity**: High
**Affects**: Phase 2, Section 2.6 (`mermaid.py`)

### Problem

Python type hints contain characters that are Mermaid syntax tokens:

- `[]` â†’ rectangle node shape
- `()` â†’ rounded node shape
- `>` â†’ asymmetric node / edge arrow
- `{}` â†’ rhombus node shape

Example breakage:
```
A[process(items: list[str] = []) -> dict[str, int]]
```

Mermaid interprets the inner `[str]` as a nested node definition. The diagram renders broken text or fails entirely.

### Specification Change

Add `_sanitize_for_mermaid()` to Section 2.6 specification:

```python
def _sanitize_for_mermaid(label: str) -> str:
    """Escape characters that conflict with Mermaid syntax.
    
    Mermaid reserves: [] () {} | > < " for node shapes, edges, and labels.
    Python signatures use all of these in type hints and defaults.
    """
    return (
        label
        .replace("[", "âŸ¨")    # U+27E8 mathematical left angle bracket
        .replace("]", "âŸ©")    # U+27E9 mathematical right angle bracket
        .replace("{", "(")
        .replace("}", ")")
        .replace('"', "'")
        .replace("|", "âˆ£")    # U+2223 divides
        .replace("<", "â€¹")    # U+2039 single left-pointing angle quotation
        .replace(">", "â€º")    # U+203A single right-pointing angle quotation
    )
```

Usage in node rendering:

```python
def _render_node(node_id: str, func: FunctionInfo) -> str:
    safe_label = _sanitize_for_mermaid(func.signature)
    return f'{node_id}["{func.qualified_name}: {safe_label}"]'
```

**Alternative (simpler)**: If Unicode bracket substitution causes rendering issues in some terminals, use the plain ASCII approach:

```python
def _sanitize_for_mermaid(label: str) -> str:
    return (
        label
        .replace("[", "(").replace("]", ")")
        .replace("{", "(").replace("}", ")")
        .replace('"', "'")
        .replace("|", "/")
        .replace("<", "â€¹").replace(">", "â€º")
    )
```

Both approaches are acceptable. The implementor should pick based on CLI rendering tests.

### Test Addition

Add to Section 7.2, **Mermaid: output** category (increase from 4 to 5 tests):

```python
def test_mermaid_type_hint_sanitization():
    """Type hints with brackets must not break Mermaid syntax."""
    source = '''
def process(items: list[str] = []) -> dict[str, int]:
    pass
'''
    snapshot = FunctionExtractor(source, "test.py").extract()
    mermaid = generate_mermaid(snapshot.functions, show_all=True)
    # Must not contain raw [] inside node labels
    # The label should use sanitized brackets
    assert "list[str]" not in mermaid
    assert "list" in mermaid  # Content preserved, just brackets changed
```

### Acceptance Criteria Update

Update Section 9, line for Mermaid:

> - [ ] `generate_mermaid` produces valid Mermaid syntax **including for signatures with type hints containing `[]`, `{}`, `>`, `|`** (verifiable at mermaid.live)

### Test Count Update

Total tests: 51 â†’ **52**

---

## Amendment 3: Async Await â€” Test Coverage Only

**Severity**: Low (not a bug, but missing test coverage)
**Affects**: Phase 2, Section 7.2 (test plan)

### Analysis

`await foo()` in the Python AST produces `Await(value=Call(...))`. The `CallExtractor` uses `ast.NodeVisitor`, whose `generic_visit()` recursively walks all child nodes. Since `CallExtractor` implements `visit_Call` but **not** `visit_Await`, the default traversal on `Await` nodes descends into the `value` child (the `Call` node) and dispatches to `visit_Call` correctly.

Verified empirically: a `visit_Call` implementation finds calls inside `await` expressions without any special handling.

**No code change needed.** However, this behavior is non-obvious and should be documented via a test to prevent future regressions (e.g., if someone adds a `visit_Await` that doesn't call `generic_visit`).

### Test Addition

Add to Section 7.2, **CallExtractor: resolution** category (increase from 6 to 7 tests):

```python
def test_await_call_resolution():
    """Calls inside await expressions must be resolved."""
    source = '''
class Service:
    async def process(self):
        result = await self.db.query()
        data = await fetch_remote()
'''
    snapshot = FunctionExtractor(source, "test.py").extract()
    process_func = snapshot.functions[0]  # Service.process
    call_names = [c.raw_name for c in process_func.calls]
    assert "self.db.query" in call_names or "db.query" in call_names
    assert "fetch_remote" in call_names
```

### Test Count Update

Total tests: 52 â†’ **53** (cumulative with Amendment 2)

---

## Amendment 4: Relative Import `None` Guard

**Severity**: Medium
**Affects**: Phase 2, Section 2.3 (`_build_import_map`)

### Problem

Relative imports produce `ImportFrom` nodes where `module` is `None`:

```python
from . import utils       # ImportFrom(module=None, level=1, names=['utils'])
from .. import base       # ImportFrom(module=None, level=2, names=['base'])
from .service import Auth # ImportFrom(module='service', level=1, names=['Auth'])
```

If `_build_import_map` accesses `node.module` without guarding for `None`, it crashes on `from . import utils`.

### Specification Change

Add explicit handling to `_build_import_map` description in Section 2.3:

```python
def _build_import_map(self, tree: ast.Module) -> None:
    """Build local_name â†’ qualified_name mapping from imports.
    
    Handles:
    - import foo              â†’ foo â†’ foo
    - import foo as bar       â†’ bar â†’ foo
    - from foo import bar     â†’ bar â†’ foo.bar
    - from foo import bar as baz â†’ baz â†’ foo.bar
    - from . import utils     â†’ utils â†’ __rel1__.utils  (relative, level 1)
    - from ..base import X    â†’ X â†’ __rel2__.base.X     (relative, level 2)
    - from .service import Y  â†’ Y â†’ __rel1__.service.Y  (relative, level 1)
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                self.import_map[local] = alias.name
        elif isinstance(node, ast.ImportFrom):
            # Guard: relative imports have level > 0 and possibly module=None
            if node.level > 0:
                prefix = f"__rel{node.level}__"
                module_part = f"{prefix}.{node.module}" if node.module else prefix
            else:
                module_part = node.module or ""
            
            for alias in node.names:
                local = alias.asname or alias.name
                qualified = f"{module_part}.{alias.name}" if module_part else alias.name
                self.import_map[local] = qualified
```

**Design note**: The `__rel1__` / `__rel2__` placeholder is intentional for Phase 2. Full relative import resolution requires knowing the package structure (where `__init__.py` files are, what `file_path` maps to in the package hierarchy). This is deferred â€” cross-file resolution is explicitly out of scope (Section 1.2). The placeholder preserves the import for debugging and avoids crashes on `None`.

### Edge Case Addition

Add new Section 5.9:

> ### 5.9 Relative imports
> ```python
> from . import utils
> from ..base import Config
> ```
>
> **Handling**: Relative imports (level > 0) are mapped with a `__rel{N}__` prefix placeholder. `from . import utils` maps to `utils â†’ __rel1__.utils`. Full resolution requires package structure awareness, which is deferred to cross-file resolution (not Phase 2). The import map entry is usable for call graph edges â€” `utils.do_thing()` resolves to `__rel1__.utils.do_thing` with `resolution="import"`, which is correct enough for same-file call graphs and Phase 6 observation matching.

### Test Addition

Add to Section 7.2, **CallExtractor: import map** category (increase from 3 to 5 tests):

```python
def test_relative_import_level_1():
    """from . import utils must not crash on module=None."""
    source = '''
from . import utils

def setup():
    utils.configure()
'''
    snapshot = FunctionExtractor(source, "test.py").extract()
    assert "utils" in snapshot.import_map
    assert snapshot.import_map["utils"].startswith("__rel1__")

def test_relative_import_from_dotmodule():
    """from .service import Auth must resolve correctly."""
    source = '''
from .service import AuthService

def create():
    AuthService.authenticate()
'''
    snapshot = FunctionExtractor(source, "test.py").extract()
    assert "AuthService" in snapshot.import_map
    assert "__rel1__.service.AuthService" == snapshot.import_map["AuthService"]
```

### Test Count Update

Total tests: 53 â†’ **55** (cumulative with Amendments 2 and 3)

---

## Amendment 5: Duplicate Qualified Names in Same File â€” Edge Case Test

**Severity**: Low
**Affects**: Phase 2, Section 5 (edge cases) and Section 7.2 (test plan)

### Analysis

The original review raised `__init__.py` collision concerns. After analysis:

- **Re-exports in `__init__.py`**: Already handled (Section 5.7). Import map captures the name, no function extraction occurs.
- **Same function name in different files**: Not a collision â€” `function_map` rows have different `file_path` values. The diff algorithm operates per-file.
- **Same qualified_name in same file**: This is the real edge case.

Python allows function redefinition in the same scope:

```python
def process():
    return "v1"

def process():  # shadows first definition
    return "v2"
```

Both produce `qualified_name="process"`. The `compare_snapshots` diff algorithm builds `dict[qualified_name, FunctionInfo]` â€” the second definition silently overwrites the first. This is actually correct behavior (matches Python's runtime semantics), but should be documented.

### Edge Case Addition

Add new Section 5.10:

> ### 5.10 Duplicate function names in same scope
> ```python
> def process():
>     return "v1"
> 
> def process():
>     return "v2"
> ```
>
> **Handling**: The extractor produces two `FunctionInfo` objects with `qualified_name="process"`. The diff algorithm's dict keying on `qualified_name` retains only the last definition â€” matching Python's runtime shadowing semantics. This is correct behavior. Linters (ruff F811) catch redefinition at dev time. No special handling needed.

### Test Addition

Add to Section 7.2, **Extractor: basics** (increase from 5 to 6 tests):

```python
def test_duplicate_function_names():
    """Redefined function: last definition wins (matches Python semantics)."""
    source = '''
def process():
    return "v1"

def process():
    return "v2"
'''
    snapshot = FunctionExtractor(source, "test.py").extract()
    # Extractor sees both, but diff/persistence retains last
    process_funcs = [f for f in snapshot.functions if f.qualified_name == "process"]
    assert len(process_funcs) == 2  # Extractor captures both
    # When persisted via compare_snapshots, dict keying retains last
```

### Test Count Update

Total tests: 55 â†’ **56** (cumulative, final)

---

## Summary of All Changes

### Phase 0 Schema Changes

| Table | Column | Before | After |
|-------|--------|--------|-------|
| `function_map` | `session_id` | `TEXT NOT NULL REFERENCES sessions(id)` | `TEXT NOT NULL` (no FK) |
| `call_graph` | `session_id` | `TEXT NOT NULL REFERENCES sessions(id)` | `TEXT NOT NULL` (no FK) |

### Phase 2 Code Changes

| Module | Change | Type |
|--------|--------|------|
| `extractor.py` | `_build_import_map`: guard `node.module is None` for relative imports | Bug fix |
| `mermaid.py` | Add `_sanitize_for_mermaid()` for bracket/angle escaping | Bug fix |
| `mermaid.py` | Use sanitized labels in node rendering | Bug fix |

### Phase 2 Documentation Changes

| Section | Change |
|---------|--------|
| 2.3 | Add relative import handling spec to `_build_import_map` |
| 2.6 | Add `_sanitize_for_mermaid()` spec |
| 3.2 | Add note about FK removal and session reconciliation |
| 5.9 (new) | Relative import edge case |
| 5.10 (new) | Duplicate function name edge case |
| 7.2 | 5 new tests across 4 categories |
| 9 | Updated Mermaid acceptance criterion to include type hint signatures |

### Test Count

| Before | After | Delta |
|--------|-------|-------|
| 51 | 56 | +5 |

| New Test | Category |
|----------|----------|
| `test_mermaid_type_hint_sanitization` | Mermaid: output |
| `test_await_call_resolution` | CallExtractor: resolution |
| `test_relative_import_level_1` | CallExtractor: import map |
| `test_relative_import_from_dotmodule` | CallExtractor: import map |
| `test_duplicate_function_names` | Extractor: basics |
