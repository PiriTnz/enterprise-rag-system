"""Metric implementations for the eval harness.

These are deliberately lightweight and explainable. The point of having them
in-repo (rather than depending on ragas / langfuse) is that recruiters and
reviewers can actually read the math and trust the numbers.
"""
from __future__ import annotations

import re

from app.schemas.models import Citation, QueryResponse, RetrievedChunk


# ---- Retrieval ----

def retrieval_precision_at_k(
    retrieved: list[RetrievedChunk],
    relevant_doc_ids: set[str] | None = None,
    relevant_chunk_ids: set[str] | None = None,
) -> float:
    """Precision@k: fraction of retrieved that are relevant."""
    if not retrieved:
        return 0.0
    relevant = 0
    for r in retrieved:
        if relevant_chunk_ids and r.chunk.chunk_id in relevant_chunk_ids:
            relevant += 1
        elif relevant_doc_ids and r.chunk.metadata.doc_id in relevant_doc_ids:
            relevant += 1
    return relevant / len(retrieved)


def retrieval_recall_at_k(
    retrieved: list[RetrievedChunk],
    relevant_doc_ids: set[str] | None = None,
    relevant_chunk_ids: set[str] | None = None,
) -> float:
    """Recall@k: fraction of relevant items that were retrieved.

    If chunk-level ground truth is given we use that; otherwise we fall
    back to doc-level (was any chunk of a relevant doc retrieved?).
    """
    if relevant_chunk_ids:
        if not relevant_chunk_ids:
            return 0.0
        found = {r.chunk.chunk_id for r in retrieved} & relevant_chunk_ids
        return len(found) / len(relevant_chunk_ids)
    if relevant_doc_ids:
        retrieved_docs = {r.chunk.metadata.doc_id for r in retrieved}
        found = retrieved_docs & relevant_doc_ids
        return len(found) / len(relevant_doc_ids)
    return 0.0


# ---- Grounding ----

def grounding_score(response: QueryResponse) -> float:
    """How well-grounded the answer is.

    Defined as: fraction of sentences in the answer that contain at least
    one citation marker referencing a retrieved chunk.

    A perfect score (1.0) means every sentence was backed by a citation.
    """
    if not response.answer:
        return 0.0
    # If structured items exist, score over them instead
    if response.items:
        if not response.items:
            return 0.0
        cited = sum(1 for it in response.items if it.citations)
        return cited / len(response.items)

    sentences = _split_sentences(response.answer)
    if not sentences:
        return 0.0
    cited_ids = {c.chunk_id for c in response.citations}
    if not cited_ids:
        return 0.0
    citation_re = re.compile(r"\[?([a-f0-9]{6,}::c\d+::[a-f0-9]{4,})\]?")
    grounded = 0
    for sent in sentences:
        for m in citation_re.finditer(sent):
            if m.group(1) in cited_ids:
                grounded += 1
                break
    return grounded / len(sentences)


def hallucination_flag(
    response: QueryResponse,
    must_contain: list[str] | None = None,
    must_not_contain: list[str] | None = None,
) -> bool:
    """Flag obvious hallucinations.

    Heuristics:
      - any `must_not_contain` substring appears
      - `must_contain` strings are missing entirely
      - answer makes claims with no citations at all even though chunks were retrieved
    """
    ans = response.answer.lower()
    if must_not_contain:
        for forbidden in must_not_contain:
            if forbidden.lower() in ans:
                return True
    if must_contain:
        for needed in must_contain:
            if needed.lower() not in ans:
                return True
    if response.chunks and not response.citations and not response.items:
        # answered something but cited nothing -> suspicious
        if "do not contain enough information" not in ans:
            return True
    return False


def citation_correctness(response: QueryResponse) -> float:
    """Fraction of citations that point to chunks actually retrieved."""
    if not response.citations:
        return 0.0 if response.items or response.answer else 1.0
    if not response.chunks:
        return 0.0
    retrieved_ids = {c.chunk.chunk_id for c in response.chunks}
    valid = sum(1 for c in response.citations if c.chunk_id in retrieved_ids)
    return valid / len(response.citations)


# ---- helpers ----

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u0600-\u06FF\u00C0-\u017F])")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]
