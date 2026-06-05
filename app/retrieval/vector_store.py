"""Vector store adapters.

A `VectorStore` is the minimal interface the retrieval layer needs: upsert,
query, delete, count. Two concrete backends ship with the system:

  * `ChromaVectorStore`  — easy local persistence, great for dev / small data
  * `QdrantVectorStore`  — production-grade, scales out, gRPC, payload filtering

Use `get_vector_store()` to pick by config.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.models import Chunk

logger = get_logger(__name__)


class VectorHit:
    __slots__ = ("chunk_id", "text", "metadata", "score")

    def __init__(self, chunk_id: str, text: str, metadata: dict, score: float) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.score = score


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: dict | None = None,
    ) -> list[VectorHit]: ...

    @abstractmethod
    def delete_all(self) -> None: ...

    @abstractmethod
    def count(self) -> int: ...


# ---------- Chroma ----------

class ChromaVectorStore(VectorStore):
    def __init__(self, collection_name: str | None = None, persist_dir: str | None = None) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.collection_name
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Chroma store ready collection={self.collection_name} "
            f"persist_dir={self.persist_dir} count={self.count()}"
        )

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        # Chroma metadata must be flat & JSON-serializable primitives.
        metas = [self._flatten_metadata(c.metadata.model_dump(mode="json")) for c in chunks]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=docs,
            metadatas=metas,
        )

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: dict | None = None,
    ) -> list[VectorHit]:
        res = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters or None,
        )
        hits: list[VectorHit] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # cosine distance -> similarity
            score = 1.0 - float(dist) if dist is not None else 0.0
            hits.append(VectorHit(cid, doc, meta or {}, score))
        return hits

    def delete_all(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0

    @staticmethod
    def _flatten_metadata(meta: dict) -> dict:
        flat: dict[str, Any] = {}
        for k, v in meta.items():
            if v is None:
                continue  # ChromaDB does not accept None values
            if isinstance(v, (str, int, float, bool)):
                flat[k] = v
            else:
                flat[k] = json.dumps(v, default=str)
        return flat


# ---------- Qdrant ----------

class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        dim: int | None = None,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm

        self._qm = qm
        self.collection_name = collection_name or settings.collection_name
        self.dim = dim or settings.embedding_dim

        self._client = QdrantClient(
            url=url or settings.qdrant_url,
            api_key=api_key or settings.qdrant_api_key,
            prefer_grpc=False,
        )

        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            )
        logger.info(
            f"Qdrant store ready collection={self.collection_name} url={url or settings.qdrant_url}"
        )

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        points = []
        for c, vec in zip(chunks, embeddings):
            payload = c.metadata.model_dump(mode="json")
            payload["text"] = c.text
            payload["chunk_id"] = c.chunk_id
            points.append(self._qm.PointStruct(
                id=self._uuid_from_chunk_id(c.chunk_id),
                vector=vec.tolist(),
                payload=payload,
            ))
        self._client.upsert(collection_name=self.collection_name, points=points)

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: dict | None = None,
    ) -> list[VectorHit]:
        q_filter = self._build_filter(filters) if filters else None
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=q_filter,
            with_payload=True,
        )
        hits = []
        for r in results:
            payload = r.payload or {}
            hits.append(VectorHit(
                chunk_id=payload.get("chunk_id", str(r.id)),
                text=payload.get("text", ""),
                metadata=payload,
                score=float(r.score),
            ))
        return hits

    def delete_all(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._qm.VectorParams(
                size=self.dim, distance=self._qm.Distance.COSINE,
            ),
        )

    def count(self) -> int:
        try:
            return self._client.count(self.collection_name, exact=True).count
        except Exception:
            return 0

    def _build_filter(self, filters: dict):
        must = [
            self._qm.FieldCondition(key=k, match=self._qm.MatchValue(value=v))
            for k, v in filters.items()
        ]
        return self._qm.Filter(must=must)

    @staticmethod
    def _uuid_from_chunk_id(chunk_id: str) -> str:
        import uuid
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


# ---------- Factory ----------

_store: VectorStore | None = None


def get_vector_store(backend: str | None = None) -> VectorStore:
    """Pick a vector backend based on config. Caches an instance per process."""
    global _store
    if _store is not None and backend is None:
        return _store

    chosen = (backend or settings.vector_backend).lower()
    if chosen == "chroma":
        _store = ChromaVectorStore()
    elif chosen == "qdrant":
        _store = QdrantVectorStore()
    else:
        raise ValueError(f"Unknown vector backend: {chosen}")
    return _store
