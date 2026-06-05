"""BM25 keyword index.

Why BM25 even though we have embeddings? Because semantic search misses exact
identifiers, codes, model numbers, names of obscure entities, and the like.
The hybrid retriever combines both — that's where the precision wins come from.

This implementation persists to disk so restarts don't lose the index.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.models import Chunk

logger = get_logger(__name__)


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Hit:
    __slots__ = ("chunk_id", "text", "metadata", "score")

    def __init__(self, chunk_id: str, text: str, metadata: dict, score: float) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.score = score


class BM25Retriever:
    """In-memory BM25 with periodic persistence.

    For very large corpora swap this for OpenSearch / Elasticsearch — same
    interface, drop-in replacement.
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        base = Path(settings.data_dir) / "bm25"
        base.mkdir(parents=True, exist_ok=True)
        self.persist_path = Path(persist_path) if persist_path else base / "bm25.pkl"

        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._load()

    # ---- persistence ----

    def _load(self) -> None:
        if self.persist_path.exists():
            try:
                with self.persist_path.open("rb") as f:
                    blob = pickle.load(f)
                self._bm25 = blob["bm25"]
                self._chunk_ids = blob["chunk_ids"]
                self._texts = blob["texts"]
                self._metadatas = blob["metadatas"]
                logger.info(f"BM25 loaded n={len(self._chunk_ids)}")
            except Exception as e:
                logger.warning(f"BM25 load failed: {e}")

    def save(self) -> None:
        with self.persist_path.open("wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "chunk_ids": self._chunk_ids,
                "texts": self._texts,
                "metadatas": self._metadatas,
            }, f)

    # ---- index ops ----

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        existing = set(self._chunk_ids)
        new_chunks = [c for c in chunks if c.chunk_id not in existing]
        for c in new_chunks:
            self._chunk_ids.append(c.chunk_id)
            self._texts.append(c.text)
            self._metadatas.append(c.metadata.model_dump(mode="json"))
        self._rebuild()
        self.save()

    def clear(self) -> None:
        self._bm25 = None
        self._chunk_ids.clear()
        self._texts.clear()
        self._metadatas.clear()
        if self.persist_path.exists():
            self.persist_path.unlink()

    def _rebuild(self) -> None:
        tokenized = [tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    # ---- query ----

    def search(self, query: str, top_k: int = 20) -> list[BM25Hit]:
        if not self._bm25 or not self._texts:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        # Top-k indices (descending)
        if len(scores) <= top_k:
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        else:
            import numpy as np
            idx_part = np.argpartition(-scores, top_k)[:top_k]
            order = sorted(idx_part.tolist(), key=lambda i: scores[i], reverse=True)
        return [
            BM25Hit(
                chunk_id=self._chunk_ids[i],
                text=self._texts[i],
                metadata=self._metadatas[i],
                score=float(scores[i]),
            )
            for i in order
            if scores[i] > 0
        ]

    def count(self) -> int:
        return len(self._chunk_ids)


_bm25: BM25Retriever | None = None


def get_bm25() -> BM25Retriever:
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Retriever()
    return _bm25
