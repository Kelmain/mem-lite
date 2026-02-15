"""Pydantic models and enums for the storage layer."""

from enum import StrEnum

from pydantic import BaseModel


class SessionStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"
    ERROR = "error"


class QueueStatus(StrEnum):
    RAW = "raw"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class ChangeType(StrEnum):
    NEW = "new"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


class FunctionKind(StrEnum):
    FUNCTION = "function"
    METHOD = "method"
    ASYNC_FUNCTION = "async_function"
    ASYNC_METHOD = "async_method"
    CLASS = "class"


class CallResolution(StrEnum):
    DIRECT = "direct"
    SELF_METHOD = "self_method"
    IMPORT = "import"
    UNRESOLVED = "unresolved"


class LearningCategory(StrEnum):
    ARCHITECTURE = "architecture"
    CONVENTION = "convention"
    GOTCHA = "gotcha"
    DEPENDENCY = "dependency"
    PATTERN = "pattern"


class Session(BaseModel):
    id: str
    project_dir: str
    started_at: str
    ended_at: str | None = None
    status: SessionStatus = SessionStatus.ACTIVE
    summary: str | None = None
    observation_count: int = 0


class Observation(BaseModel):
    id: str
    session_id: str
    tool_name: str
    title: str
    summary: str
    detail: str | None = None
    files_touched: str = "[]"
    functions_changed: str = "[]"
    tokens_raw: int = 0
    tokens_compressed: int = 0
    created_at: str


class FunctionMapEntry(BaseModel):
    id: str
    session_id: str
    file_path: str
    qualified_name: str
    kind: FunctionKind
    signature: str
    docstring: str | None = None
    body_hash: str
    decorators: str = "[]"
    change_type: ChangeType = ChangeType.NEW
    updated_at: str


class CallGraphEdge(BaseModel):
    id: str
    caller_file: str
    caller_function: str
    callee_file: str
    callee_function: str
    resolution: CallResolution
    confidence: float = 1.0
    times_confirmed: int = 0
    source: str = "ast"
    session_id: str
    created_at: str


class Learning(BaseModel):
    id: str
    category: LearningCategory
    content: str
    confidence: float = 0.5
    times_seen: int = 1
    source_sessions: str = "[]"  # JSON array of session IDs
    is_manual: bool = False
    is_active: bool = True
    created_at: str
    updated_at: str


class PendingQueueItem(BaseModel):
    id: str
    session_id: str
    tool_name: str
    raw_output: str
    files_touched: str = "[]"
    priority: str = "normal"
    status: QueueStatus = QueueStatus.RAW
    attempts: int = 0
    created_at: str


class EventLogEntry(BaseModel):
    id: str
    session_id: str | None = None
    event_type: str
    data: str = "{}"
    duration_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    created_at: str


# -----------------------------------------------------------------------
# Phase 3: Worker + Compression models
# -----------------------------------------------------------------------


class FunctionChangeRecord(BaseModel):
    """A single function change reported by the compressor."""

    file: str
    name: str
    action: str  # "new" | "modified" | "deleted"


class CompressedObservation(BaseModel):
    """Result of AI compression — becomes an Observation row."""

    title: str
    summary: str
    detail: str | None = None
    files_touched: list[str] = []
    functions_changed: list[FunctionChangeRecord] = []
    tokens_in: int = 0
    tokens_out: int = 0


class SessionSummary(BaseModel):
    """Result of session summarization."""

    summary: str
    key_files: list[str] = []
    key_decisions: list[str] = []


class QueueStats(BaseModel):
    """Queue status counts for debugging."""

    raw: int = 0
    processing: int = 0
    done: int = 0
    error: int = 0


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str = "ok"
    uptime_s: int = 0
    queue_depth: int = 0
    observations_today: int = 0


# -----------------------------------------------------------------------
# Phase 4: Embeddings + Search models
# -----------------------------------------------------------------------


class EmbeddingStatus(StrEnum):
    PENDING = "pending"
    EMBEDDED = "embedded"
    FAILED = "failed"


class SearchResult(BaseModel):
    """Single search result from hybrid search."""

    observation_id: str
    session_id: str
    title: str
    summary: str
    files_touched: str = ""
    score: float | None = None
    created_at: str = ""


class SearchResponse(BaseModel):
    """Response from the search endpoint."""

    results: list[SearchResult]
    query: str
    count: int
    search_type: str  # "hybrid" or "fts"
