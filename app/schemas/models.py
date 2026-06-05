"""Pydantic schemas. These are the contract between every layer of the system.

Keeping them in one place makes the API surface easy to audit and lets the
frontend/codegen stay in sync.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------- Documents & chunks ----------

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")  # allow custom domain fields

    source: str  # filename or path
    doc_id: str
    doc_type: DocumentType
    page: int | None = None
    section: str | None = None
    chunk_index: int
    char_start: int | None = None
    char_end: int | None = None
    ingested_at: datetime = Field(default_factory=_utcnow)


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    retriever: str  # "vector", "bm25", "hybrid", "reranked"
    rank: int


# ---------- Ingestion ----------

class IngestionRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)
    rebuild_index: bool = False


class IngestionResult(BaseModel):
    documents_processed: int
    chunks_created: int
    chunks_indexed: int
    failures: list[dict[str, str]] = Field(default_factory=list)
    duration_seconds: float


# ---------- Query ----------

class StructuredOutputType(str, Enum):
    ANSWER = "answer"          # default Q&A
    SUMMARY = "summary"        # executive summary of retrieved docs
    DECISIONS = "decisions"    # extract decisions
    RISKS = "risks"            # extract risks
    ACTIONS = "actions"        # extract recommended actions


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = None
    output_type: StructuredOutputType = StructuredOutputType.ANSWER
    filters: dict[str, Any] | None = None
    enable_reranking: bool | None = None  # override config
    llm_provider: str | None = None  # override config
    include_chunks: bool = True


class Citation(BaseModel):
    source: str       # e.g. "report_2025.pdf"
    page: int | None = None
    section: str | None = None
    chunk_id: str
    snippet: str      # short snippet for UI display
    score: float


class StructuredItem(BaseModel):
    """A single extracted item for summary/decisions/risks/actions outputs."""
    title: str
    detail: str
    citations: list[Citation] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    output_type: StructuredOutputType
    citations: list[Citation] = Field(default_factory=list)
    items: list[StructuredItem] = Field(default_factory=list)  # for structured outputs
    confidence: float = Field(ge=0.0, le=1.0)
    chunks: list[RetrievedChunk] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float
    provider_used: str


# ---------- Evaluation ----------

class EvalCase(BaseModel):
    """One ground-truth example for the eval harness."""
    query: str
    expected_answer: str | None = None
    relevant_doc_ids: list[str] = Field(default_factory=list)  # for retrieval metrics
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    case: EvalCase
    response: QueryResponse
    retrieval_precision: float | None = None
    retrieval_recall: float | None = None
    grounding_score: float | None = None  # how well the answer cites retrieved chunks
    hallucination_flag: bool = False
    citation_correctness: float | None = None
    passed: bool = False


class EvalSummary(BaseModel):
    total_cases: int
    passed: int
    avg_retrieval_precision: float
    avg_retrieval_recall: float
    avg_grounding: float
    hallucination_rate: float
    avg_citation_correctness: float
    avg_latency_ms: float
    by_case: list[EvalResult]
