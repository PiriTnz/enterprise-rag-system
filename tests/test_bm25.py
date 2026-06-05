"""Tests for the BM25 retriever. No external models needed."""
import tempfile
from pathlib import Path

from app.retrieval.bm25 import BM25Retriever
from app.schemas.models import Chunk, ChunkMetadata, DocumentType


def _make_chunk(idx: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc1::c{idx}::abcd",
        text=text,
        metadata=ChunkMetadata(
            source="report.pdf",
            doc_id="doc1",
            doc_type=DocumentType.PDF,
            chunk_index=idx,
            page=idx + 1,
        ),
    )


def test_bm25_add_and_search():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bm25.pkl"
        bm = BM25Retriever(persist_path=path)
        bm.add([
            _make_chunk(0, "Quarterly revenue exceeded projections by 12 percent."),
            _make_chunk(1, "The team migrated the data pipeline to Kafka last month."),
            _make_chunk(2, "Capacity constraints in Q3 were caused by RGPT throttling."),
        ])
        hits = bm.search("Q3 capacity constraints", top_k=3)
        assert hits, "expected at least one hit"
        # The most relevant chunk should be the one mentioning Q3 + capacity.
        assert hits[0].chunk_id.endswith("c2::abcd")


def test_bm25_persists():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bm25.pkl"
        bm = BM25Retriever(persist_path=path)
        bm.add([_make_chunk(0, "alpha beta gamma")])
        # Reload
        bm2 = BM25Retriever(persist_path=path)
        assert bm2.count() == 1


def test_bm25_empty_query_returns_nothing_when_empty_index():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bm25.pkl"
        bm = BM25Retriever(persist_path=path)
        assert bm.search("anything", top_k=5) == []
