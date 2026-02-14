"""Hybrid search orchestration with graceful degradation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_mem_lite.search.embedder import Embedder
    from claude_mem_lite.search.lance_store import LanceStore


class HybridSearcher:
    """Orchestrates hybrid search with graceful fallback to FTS-only."""

    def __init__(self, lance_store: LanceStore, embedder: Embedder) -> None:
        self.lance_store = lance_store
        self.embedder = embedder

    def search(
        self,
        query: str,
        limit: int = 5,
        query_type: str = "observation",
    ) -> tuple[list[dict], str]:
        """Search with best available method.

        Returns:
            Tuple of (results, search_type) where search_type is "hybrid" or "fts".
        """
        if self.embedder.available:
            results = self.lance_store.search_observations(query, limit, query_type)
            return results, "hybrid"

        results = self.lance_store.search_fts_only(query, limit)
        return results, "fts"
