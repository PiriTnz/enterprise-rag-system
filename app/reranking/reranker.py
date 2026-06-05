"""Reranking layer.

Cross-encoder rerankers score (query, passage) jointly, which is far more
accurate than the bi-encoder embeddings used in first-stage retrieval — but
also slower, so we only rerank the top-N candidates from hybrid search.

This is a single, focused responsibility module so it's easy to swap models
(BGE-reranker, ms-marco, Cohere Rerank API, etc.) without touching anything
else.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.models import RetrievedChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or settings.reranker_model
        self.device = device or settings.embedding_device
        logger.info(f"Loading reranker model={self.model_name} device={self.device}")
        self._model: CrossEncoder = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return candidates
        top_n = top_n or settings.rerank_top_n

        import math
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)

        # Use sigmoid for an always-meaningful [0,1] score, even with 1 candidate.
        # Cross-encoder raw scores are logits; sigmoid maps them to probabilities.
        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        scored: list[RetrievedChunk] = []
        for cand, raw in zip(candidates, scores):
            normalized = _sigmoid(float(raw))
            scored.append(RetrievedChunk(
                chunk=cand.chunk,
                score=normalized,
                retriever="reranked",
                rank=cand.rank,  # will be overwritten below
                # store raw score in metadata if downstream wants it
            ))
        scored.sort(key=lambda c: c.score, reverse=True)
        for new_rank, c in enumerate(scored):
            c.rank = new_rank
        return scored[:top_n]


_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
