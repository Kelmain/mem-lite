"""LanceDB table management for vector + FTS search."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from claude_mem_lite.config import Config
    from claude_mem_lite.search.embedder import Embedder

logger = logging.getLogger(__name__)


def _observation_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("observation_id", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("files_touched", pa.string()),
            pa.field("functions_changed", pa.string()),
            pa.field("created_at", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def _summary_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("session_id", pa.string()),
            pa.field("summary_text", pa.string()),
            pa.field("project_path", pa.string()),
            pa.field("created_at", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def _learning_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("learning_id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("category", pa.string()),
            pa.field("confidence", pa.float32()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def _rrf_merge(
    vector_results: list[dict],
    fts_results: list[dict],
    limit: int,
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion: merges two ranked lists by observation_id."""
    scores: dict[str, float] = {}
    for rank, r in enumerate(vector_results):
        rid = r["observation_id"]
        scores[rid] = scores.get(rid, 0) + 1.0 / (k + rank + 1)
    for rank, r in enumerate(fts_results):
        rid = r["observation_id"]
        scores[rid] = scores.get(rid, 0) + 1.0 / (k + rank + 1)

    all_results = {r["observation_id"]: r for r in [*vector_results, *fts_results]}
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    return [{**all_results[rid], "_relevance_score": scores[rid]} for rid in sorted_ids[:limit]]


class LanceStore:
    """Manages LanceDB tables for vector + FTS search."""

    def __init__(self, config: Config, embedder: Embedder) -> None:
        self.config = config
        self.embedder = embedder
        self._db: Any = None
        self._tables: dict[str, Any] = {}
        self._fts_created: set[str] = set()

    def connect(self) -> None:
        """Connect to LanceDB and ensure tables exist."""
        import lancedb

        self._db = lancedb.connect(str(self.config.lance_path))
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        existing = set(self._db.table_names())
        dim = self.config.embedding_dim
        schemas = [
            ("observations_vec", _observation_schema(dim)),
            ("summaries_vec", _summary_schema(dim)),
            ("learnings_vec", _learning_schema(dim)),
        ]
        for name, schema in schemas:
            if name not in existing:
                self._db.create_table(name, schema=schema)
            self._tables[name] = self._db.open_table(name)

    def _ensure_fts_index(self, table_name: str) -> None:
        """Create FTS index after first row is added (requires non-empty table)."""
        if table_name in self._fts_created:
            return
        table = self._tables[table_name]
        if table.count_rows() == 0:
            return
        try:
            table.create_fts_index("title", replace=True)
            self._fts_created.add(table_name)
        except Exception:
            logger.warning("Failed to create FTS index for %s", table_name, exc_info=True)

    def add_observation(
        self,
        obs_id: str,
        session_id: str,
        title: str,
        summary: str,
        files_touched: str,
        functions_changed: str,
        created_at: str,
    ) -> None:
        """Embed title and store observation in LanceDB."""
        vector = self.embedder.embed_single(title, query_type="document")
        self._tables["observations_vec"].add(
            [
                {
                    "observation_id": obs_id,
                    "session_id": session_id,
                    "title": title,
                    "summary": summary,
                    "files_touched": files_touched,
                    "functions_changed": functions_changed,
                    "created_at": created_at,
                    "vector": vector,
                }
            ]
        )
        self._ensure_fts_index("observations_vec")

    def add_summary(
        self,
        session_id: str,
        summary_text: str,
        project_path: str,
        created_at: str,
    ) -> None:
        """Embed and store a session summary in LanceDB."""
        vector = self.embedder.embed_single(summary_text, query_type="document")
        self._tables["summaries_vec"].add(
            [
                {
                    "session_id": session_id,
                    "summary_text": summary_text,
                    "project_path": project_path,
                    "created_at": created_at,
                    "vector": vector,
                }
            ]
        )

    def search_observations(
        self,
        query: str,
        limit: int = 5,
        query_type: str = "observation",
    ) -> list[dict]:
        """Hybrid search: vector + FTS merged with RRF reranking."""
        table = self._tables["observations_vec"]
        if table.count_rows() == 0:
            return []

        fetch = limit * 2  # over-fetch for RRF merge

        if self.embedder.available:
            vector = self.embedder.embed_single(query, query_type=query_type)
            # Vector search
            try:
                vec_results = table.search(vector).limit(fetch).to_list()
            except Exception:
                logger.warning("Vector search failed", exc_info=True)
                vec_results = []

            # FTS search
            fts_results = self._fts_search(table, query, fetch)

            if vec_results and fts_results:
                return _rrf_merge(vec_results, fts_results, limit)
            if vec_results:
                return list(vec_results[:limit])
            if fts_results:
                return list(fts_results[:limit])
            return []

        return self.search_fts_only(query, limit)

    def search_fts_only(self, query: str, limit: int = 5) -> list[dict]:
        """FTS-only search (fallback when embeddings unavailable)."""
        table = self._tables["observations_vec"]
        if table.count_rows() == 0:
            return []
        return self._fts_search(table, query, limit)

    def _fts_search(self, table: Any, query: str, limit: int) -> list[dict]:
        """Run FTS query, returning empty list on failure."""
        self._ensure_fts_index("observations_vec")
        try:
            results: list[dict] = table.search(query, query_type="fts").limit(limit).to_list()
        except Exception:
            logger.warning("FTS search failed for query=%r", query, exc_info=True)
            return []
        else:
            return results
