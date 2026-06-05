"""Unit tests for the chunker. These don't touch any model — pure logic."""
from app.ingestion.chunker import Chunker, ChunkingConfig
from app.ingestion.loaders import LoadedSegment
from app.schemas.models import DocumentType


def test_short_text_single_chunk():
    chunker = Chunker(ChunkingConfig(chunk_size=200, chunk_overlap=20))
    segments = [LoadedSegment(text="This is a short test document. " * 5, page=1)]
    chunks = chunker.chunk_segments(
        segments,
        source="test.txt",
        doc_id="doc1",
        doc_type=DocumentType.TXT,
    )
    assert len(chunks) >= 1
    assert all(c.metadata.doc_id == "doc1" for c in chunks)
    assert all(c.metadata.page == 1 for c in chunks)


def test_large_text_splits():
    chunker = Chunker(ChunkingConfig(chunk_size=30, chunk_overlap=4))
    # Force a real split
    long_text = " ".join([f"sentence{i}." for i in range(200)])
    segments = [LoadedSegment(text=long_text)]
    chunks = chunker.chunk_segments(
        segments, source="big.txt", doc_id="d", doc_type=DocumentType.TXT,
    )
    assert len(chunks) > 1
    # Each chunk has a unique chunk_id
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)


def test_min_chunk_chars_filter():
    chunker = Chunker(ChunkingConfig(chunk_size=200, chunk_overlap=0, min_chunk_chars=200))
    segments = [LoadedSegment(text="too short")]
    chunks = chunker.chunk_segments(
        segments, source="x.txt", doc_id="d", doc_type=DocumentType.TXT,
    )
    assert chunks == []


def test_chunk_indices_monotonic():
    chunker = Chunker(ChunkingConfig(chunk_size=40, chunk_overlap=4))
    segments = [
        LoadedSegment(text="alpha. " * 20, page=1),
        LoadedSegment(text="beta. " * 20, page=2),
    ]
    chunks = chunker.chunk_segments(
        segments, source="multi.txt", doc_id="d", doc_type=DocumentType.TXT,
    )
    indices = [c.metadata.chunk_index for c in chunks]
    assert indices == sorted(indices)
    assert indices[0] == 0
