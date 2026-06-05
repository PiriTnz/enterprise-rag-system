"""Embedding model wrapper. Uses sentence-transformers by default.

The wrapper exists so we can swap in a remote embedding API (OpenAI, Cohere)
without touching the rest of the system.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers (BGE / E5 / etc)."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size or settings.embedding_batch_size

        logger.info(f"Loading embedder model={self.model_name} device={self.device}")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self.dim = self._model.get_sentence_embedding_dimension()

        # BGE and E5 use different query prefixes for asymmetric retrieval.
        name = self.model_name.lower()
        if "bge" in name:
            self._query_prefix = "Represent this sentence for searching relevant passages: "
            self._doc_prefix = ""
        elif "e5" in name:
            self._query_prefix = "query: "
            self._doc_prefix = "passage: "
        else:
            self._query_prefix = ""
            self._doc_prefix = ""

    def embed(self, texts: list[str]) -> np.ndarray:
        prepared = [self._doc_prefix + t for t in texts] if self._doc_prefix else texts
        vecs = self._model.encode(
            prepared,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs

    def embed_query(self, text: str) -> np.ndarray:
        prepared = (self._query_prefix + text) if self._query_prefix else text
        vec = self._model.encode(
            [prepared],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return vec


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder()
    return _embedder
