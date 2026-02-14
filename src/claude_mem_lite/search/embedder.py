"""Qwen3-Embedding-0.6B wrapper for local embedding generation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import ClassVar

    from claude_mem_lite.config import Config


logger = logging.getLogger(__name__)


class Embedder:
    """Manages local embedding model for vector search."""

    INSTRUCTIONS: ClassVar[dict[str, str]] = {
        "observation": (
            "Instruct: Find development observations about code changes and decisions\nQuery: "
        ),
        "code": ("Instruct: Find code snippets, functions, and implementation details\nQuery: "),
        "learning": ("Instruct: Find project learnings, patterns, and best practices\nQuery: "),
        "document": "",
    }

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model: Any = None
        self._available = False

    def load(self) -> bool:
        """Load embedding model. Returns True on success, False on failure."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.config.embedding_model,
                device=self.config.embedding_device,
                truncate_dim=self.config.embedding_dim,
            )
            self._available = True
        except Exception:
            logger.warning(
                "Failed to load %s — search will use FTS-only fallback.",
                self.config.embedding_model,
                exc_info=True,
            )
            self._available = False
        return self._available

    @property
    def available(self) -> bool:
        """Whether the embedding model is loaded and ready."""
        return self._available

    @property
    def ndims(self) -> int:
        """Embedding dimension."""
        return self.config.embedding_dim

    def embed_texts(self, texts: list[str], query_type: str = "document") -> list[list[float]]:
        """Embed multiple texts with instruction prefix based on query_type."""
        if not self._available or self._model is None:
            msg = "Embedding model not loaded"
            raise RuntimeError(msg)

        prefix = self.INSTRUCTIONS.get(query_type, "")
        prefixed = [f"{prefix}{t}" for t in texts] if prefix else texts

        embeddings = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()  # type: ignore[no-any-return]

    def embed_single(self, text: str, query_type: str = "document") -> list[float]:
        """Embed a single text. Convenience wrapper around embed_texts."""
        return self.embed_texts([text], query_type)[0]
