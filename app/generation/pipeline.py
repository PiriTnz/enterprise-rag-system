"""Top-level RAG orchestrator.

retrieve -> (optionally) rerank -> build prompt -> generate -> parse + cite -> score

This is the function the API and the eval harness both call.
"""
from __future__ import annotations

import json
import re
import time

from app.core.config import settings
from app.core.logging import get_logger, log_event
from app.generation.prompts import (
    SYSTEM_GROUNDED,
    build_answer_prompt,
    build_structured_prompt,
)
from app.generation.providers import get_llm
from app.reranking.reranker import get_reranker
from app.retrieval.hybrid import HybridRetriever
from app.schemas.models import (
    Citation,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    StructuredItem,
    StructuredOutputType,
)

logger = get_logger(__name__)


# Match either [chunk_id], (chunk_id), or bare chunk_id mentions of the form
# <doc_id>::c<idx>::<hex>. We keep the bare form too because small models
# sometimes drop the brackets.
_CITATION_RE = re.compile(r"[\[\(]?([a-f0-9]{6,}::c\d+::[a-f0-9]{4,})[\]\)]?")


class RAGPipeline:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self._reranker = None  # lazy
        self.llm = get_llm()

    @property
    def reranker(self):
        if self._reranker is None and settings.enable_reranking:
            self._reranker = get_reranker()
        return self._reranker

    # ---- public entrypoint ----

    def run(self, req: QueryRequest) -> QueryResponse:
        t0 = time.perf_counter()

        # 1) Retrieve
        # Pull more than `top_k_final` so reranker has candidates to work with.
        retrieval_k = max(
            settings.top_k_final * 4,
            settings.rerank_top_n * 4,
        )
        candidates = self.retriever.search(
            req.query,
            top_k=retrieval_k,
            filters=req.filters,
        )

        # 2) Rerank
        enable_rr = (req.enable_reranking
                     if req.enable_reranking is not None
                     else settings.enable_reranking)
        if enable_rr and self.reranker and candidates:
            top_n = req.top_k or settings.top_k_final
            top_chunks = self.reranker.rerank(req.query, candidates, top_n=top_n)
        else:
            top_chunks = candidates[: (req.top_k or settings.top_k_final)]

        if not top_chunks:
            latency = (time.perf_counter() - t0) * 1000
            return QueryResponse(
                answer="The available documents do not contain enough information to answer this.",
                output_type=req.output_type,
                citations=[],
                items=[],
                confidence=0.0,
                chunks=[] if req.include_chunks else None,
                metadata={"reason": "no_retrieval_hits"},
                latency_ms=latency,
                provider_used="none",
            )

        # 3) Generate
        if req.output_type == StructuredOutputType.ANSWER:
            prompt = build_answer_prompt(req.query, top_chunks)
            json_mode = False
        else:
            prompt = build_structured_prompt(req.query, top_chunks, req.output_type)
            json_mode = True

        llm_resp = self.llm.complete(
            prompt=prompt,
            system=SYSTEM_GROUNDED,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            json_mode=json_mode,
            preferred=req.llm_provider,
        )

        # 4) Parse + extract citations
        if req.output_type == StructuredOutputType.ANSWER:
            answer_text = llm_resp.text.strip()
            citations = self._extract_citations(answer_text, top_chunks)
            items: list[StructuredItem] = []
        else:
            answer_text, items, citations = self._parse_structured(llm_resp.text, top_chunks)

        # 5) Confidence
        confidence = self._confidence(top_chunks, citations, answer_text)

        latency_ms = (time.perf_counter() - t0) * 1000
        log_event(
            logger, "rag_query",
            output_type=req.output_type.value,
            n_chunks=len(top_chunks),
            n_citations=len(citations),
            confidence=f"{confidence:.2f}",
            latency_ms=f"{latency_ms:.0f}",
            provider=llm_resp.provider,
        )

        return QueryResponse(
            answer=answer_text,
            output_type=req.output_type,
            citations=citations,
            items=items,
            confidence=confidence,
            chunks=top_chunks if req.include_chunks else None,
            metadata={
                "model": llm_resp.model,
                "llm_latency_ms": round(llm_resp.latency_ms, 1),
                "reranked": bool(enable_rr and self.reranker),
            },
            latency_ms=round(latency_ms, 1),
            provider_used=llm_resp.provider,
        )

    # ---- helpers ----

    def _extract_citations(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> list[Citation]:
        chunk_by_id = {c.chunk.chunk_id: c for c in chunks}
        cited_ids: list[str] = []
        for m in _CITATION_RE.finditer(answer):
            cid = m.group(1)
            if cid in chunk_by_id and cid not in cited_ids:
                cited_ids.append(cid)

        citations: list[Citation] = []
        for cid in cited_ids:
            rc = chunk_by_id[cid]
            citations.append(self._citation_from_chunk(rc))
        return citations

    def _citation_from_chunk(self, rc: RetrievedChunk) -> Citation:
        text = rc.chunk.text.strip().replace("\n", " ")
        snippet = (text[:200] + "…") if len(text) > 200 else text
        return Citation(
            source=rc.chunk.metadata.source,
            page=rc.chunk.metadata.page,
            section=rc.chunk.metadata.section,
            chunk_id=rc.chunk.chunk_id,
            snippet=snippet,
            score=rc.score,
        )

    def _parse_structured(
        self,
        raw: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, list[StructuredItem], list[Citation]]:
        chunk_by_id = {c.chunk.chunk_id: c for c in chunks}
        data = self._safe_json(raw)
        if not data:
            return raw.strip(), [], []

        summary = str(data.get("summary", "")).strip()
        items_raw = data.get("items", []) or []

        items: list[StructuredItem] = []
        all_citation_ids: list[str] = []
        for item in items_raw:
            cids = item.get("chunk_ids") or []
            item_cits: list[Citation] = []
            for cid in cids:
                if cid in chunk_by_id:
                    item_cits.append(self._citation_from_chunk(chunk_by_id[cid]))
                    if cid not in all_citation_ids:
                        all_citation_ids.append(cid)
            items.append(StructuredItem(
                title=str(item.get("title", "")).strip(),
                detail=str(item.get("detail", "")).strip(),
                citations=item_cits,
            ))

        flat_citations = [
            self._citation_from_chunk(chunk_by_id[cid])
            for cid in all_citation_ids
        ]
        return summary, items, flat_citations

    @staticmethod
    def _safe_json(raw: str) -> dict | None:
        raw = raw.strip()
        # Strip code fences if the model wrapped them.
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n?|\n?```$", "", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # try to find the first {...} block
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
            return None

    def _confidence(
        self,
        chunks: list[RetrievedChunk],
        citations: list[Citation],
        answer: str,
    ) -> float:
        """Confidence heuristic combining:
          - top retrieval score
          - average top-3 retrieval score
          - whether citations are present
          - whether the model explicitly admitted insufficient info
        """
        if not chunks:
            return 0.0

        admitted_unknown = "do not contain enough information" in answer.lower()
        if admitted_unknown:
            return 0.05

        top_scores = [c.score for c in chunks[: min(3, len(chunks))]]
        avg_top = sum(top_scores) / len(top_scores)
        top1 = chunks[0].score

        citation_bonus = 0.0
        if citations:
            citation_bonus = min(0.2, 0.05 * len(citations))

        base = 0.4 * top1 + 0.4 * avg_top
        confidence = base + citation_bonus
        return max(0.0, min(1.0, confidence))


_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
