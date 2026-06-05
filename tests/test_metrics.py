"""Tests for evaluation metrics."""
from app.evaluation.metrics import (
    citation_correctness,
    grounding_score,
    hallucination_flag,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
)
from app.schemas.models import (
    Chunk,
    ChunkMetadata,
    Citation,
    DocumentType,
    QueryResponse,
    RetrievedChunk,
    StructuredOutputType,
)


def _retrieved(chunk_id: str, doc_id: str = "d1", rank: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            text="some text",
            metadata=ChunkMetadata(
                source="x.pdf", doc_id=doc_id,
                doc_type=DocumentType.PDF, chunk_index=rank,
            ),
        ),
        score=0.9 - rank * 0.1,
        retriever="hybrid",
        rank=rank,
    )


def test_precision_at_k_exact_chunks():
    retrieved = [_retrieved("c1"), _retrieved("c2"), _retrieved("c3")]
    p = retrieval_precision_at_k(retrieved, relevant_chunk_ids={"c1", "c2"})
    assert abs(p - 2 / 3) < 1e-6


def test_recall_at_k_doc_level():
    retrieved = [
        _retrieved("c1", doc_id="dA"),
        _retrieved("c2", doc_id="dB"),
    ]
    r = retrieval_recall_at_k(retrieved, relevant_doc_ids={"dA", "dB", "dC"})
    assert abs(r - 2 / 3) < 1e-6


def test_grounding_score_with_citations():
    cid = "abc123::c0::dead"
    resp = QueryResponse(
        answer=f"This is the first claim [{cid}]. This is the second claim.",
        output_type=StructuredOutputType.ANSWER,
        citations=[Citation(source="x.pdf", chunk_id=cid, snippet="...", score=0.9)],
        items=[],
        confidence=0.8,
        chunks=[_retrieved(cid)],
        latency_ms=10,
        provider_used="test",
    )
    # 1 of 2 sentences cited
    g = grounding_score(resp)
    assert 0.49 < g < 0.51


def test_hallucination_when_no_citations_but_chunks_present():
    resp = QueryResponse(
        answer="Some confident claim with no citations.",
        output_type=StructuredOutputType.ANSWER,
        citations=[],
        items=[],
        confidence=0.5,
        chunks=[_retrieved("c1")],
        latency_ms=10,
        provider_used="test",
    )
    assert hallucination_flag(resp) is True


def test_citation_correctness():
    resp = QueryResponse(
        answer="x",
        output_type=StructuredOutputType.ANSWER,
        citations=[
            Citation(source="x.pdf", chunk_id="c1", snippet="...", score=0.9),
            Citation(source="x.pdf", chunk_id="ghost", snippet="...", score=0.9),
        ],
        items=[],
        confidence=0.5,
        chunks=[_retrieved("c1")],
        latency_ms=10,
        provider_used="test",
    )
    c = citation_correctness(resp)
    assert abs(c - 0.5) < 1e-6
