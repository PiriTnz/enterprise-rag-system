"""Hybrid retriever — combines vector + BM25 with Reciprocal Rank Fusion.

RRF is the standard, well-behaved fusion technique. It avoids the need to
calibrate disparate score scales (cosine similarity vs BM25 scores) and is
used by major search systems precisely because it's robust to that.

If you want to bias toward one side, raise the other's `k`. We expose
`alpha` too for weighted RRF.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import get_logger, log_event
from app.ingestion.embedder import get_embedder
from app.retrieval.bm25 import BM25Hit, get_bm25
from app.retrieval.vector_store import VectorHit, get_vector_store
from app.schemas.models import Chunk, ChunkMetadata, RetrievedChunk

logger = get_logger(__name__)


@dataclass
class HybridConfig:
    top_k_vector: int = settings.top_k_vector
    top_k_bm25: int = settings.top_k_bm25
    top_k_final: int = settings.top_k_final
    rrf_k: int = 60  # standard RRF constant
    alpha: float = settings.hybrid_alpha  # vector weight ; (1-alpha) for bm25


class HybridRetriever:
    def __init__(self, config: HybridConfig | None = None) -> None:
        self.config = config or HybridConfig()
        self.embedder = get_embedder()
        self.vector_store = get_vector_store()
        self.bm25 = get_bm25()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.config.top_k_final

        # Vector branch
        q_vec = self.embedder.embed_query(query)
        vector_hits: list[VectorHit] = self.vector_store.query(
            q_vec, top_k=self.config.top_k_vector, filters=filters,
        )

        # BM25 branch
        bm25_hits: list[BM25Hit] = self.bm25.search(query, top_k=self.config.top_k_bm25)

        fused = self._rrf(vector_hits, bm25_hits)

        # Take top_k after fusion (rerank will happen as a separate layer).
        log_event(
            logger, "retrieval",
            query=query[:80].replace(" ", "_"),
            vector_n=len(vector_hits),
            bm25_n=len(bm25_hits),
            fused_n=len(fused),
        )
        return fused[:top_k]

    # ---- fusion ----

    def _rrf(
        self,
        vector_hits: list[VectorHit],
        bm25_hits: list[BM25Hit],
    ) -> list[RetrievedChunk]:
        scores: dict[str, float] = defaultdict(float)
        meta_pool: dict[str, tuple[str, dict]] = {}

        k = self.config.rrf_k
        alpha = self.config.alpha

        for rank, h in enumerate(vector_hits):
            scores[h.chunk_id] += alpha * (1.0 / (k + rank + 1))
            meta_pool[h.chunk_id] = (h.text, h.metadata)

        for rank, h in enumerate(bm25_hits):
            scores[h.chunk_id] += (1 - alpha) * (1.0 / (k + rank + 1))
            meta_pool.setdefault(h.chunk_id, (h.text, h.metadata))

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        out: list[RetrievedChunk] = []
        for new_rank, (cid, score) in enumerate(ordered):
            text, meta = meta_pool[cid]
            chunk = Chunk(
                chunk_id=cid,
                text=text,
                metadata=self._reconstruct_metadata(meta),
            )
            out.append(RetrievedChunk(
                chunk=chunk,
                score=float(score),
                retriever="hybrid",
                rank=new_rank,
            ))
        return out

    @staticmethod
    def _reconstruct_metadata(meta: dict) -> ChunkMetadata:
        # Filter to known + arbitrary fields; ChunkMetadata allows extras.
        known = {
            "source": meta.get("source", "unknown"),
            "doc_id": meta.get("doc_id", "unknown"),
            "doc_type": meta.get("doc_type", "txt"),
            "page": meta.get("page"),
            "section": meta.get("section"),
            "chunk_index": int(meta.get("chunk_index", 0)),
            "char_start": meta.get("char_start"),
            "char_end": meta.get("char_end"),
        }
        if "ingested_at" in meta:
            known["ingested_at"] = meta["ingested_at"]
        return ChunkMetadata(**known)
