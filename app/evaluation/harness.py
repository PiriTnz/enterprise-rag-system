"""End-to-end evaluation harness.

Reads a YAML/JSON file of EvalCases, runs each through the RAG pipeline, and
produces an EvalSummary with retrieval, grounding, hallucination, citation,
and latency metrics — plus a markdown report.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.core.logging import get_logger
from app.evaluation.metrics import (
    citation_correctness,
    grounding_score,
    hallucination_flag,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
)
from app.generation.pipeline import get_pipeline
from app.schemas.models import (
    EvalCase,
    EvalResult,
    EvalSummary,
    QueryRequest,
    StructuredOutputType,
)

logger = get_logger(__name__)


def load_cases(path: str | Path) -> list[EvalCase]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return [EvalCase(**c) for c in raw["cases"]]


def evaluate(
    cases: list[EvalCase],
    output_type: StructuredOutputType = StructuredOutputType.ANSWER,
) -> EvalSummary:
    pipeline = get_pipeline()
    results: list[EvalResult] = []

    for i, case in enumerate(cases, 1):
        logger.info(f"[{i}/{len(cases)}] {case.query[:80]}")
        req = QueryRequest(
            query=case.query,
            output_type=output_type,
            include_chunks=True,
        )
        resp = pipeline.run(req)

        relevant_doc_ids = set(case.relevant_doc_ids) or None
        relevant_chunk_ids = set(case.relevant_chunk_ids) or None

        precision = retrieval_precision_at_k(
            resp.chunks or [], relevant_doc_ids, relevant_chunk_ids,
        )
        recall = retrieval_recall_at_k(
            resp.chunks or [], relevant_doc_ids, relevant_chunk_ids,
        )
        grounding = grounding_score(resp)
        halluc = hallucination_flag(resp, case.must_contain, case.must_not_contain)
        cit_correct = citation_correctness(resp)

        passed = (
            (precision >= 0.5 or not relevant_doc_ids and not relevant_chunk_ids)
            and grounding >= 0.5
            and not halluc
            and cit_correct >= 0.8
        )

        results.append(EvalResult(
            case=case,
            response=resp,
            retrieval_precision=precision,
            retrieval_recall=recall,
            grounding_score=grounding,
            hallucination_flag=halluc,
            citation_correctness=cit_correct,
            passed=passed,
        ))

    return _summarize(results)


def _summarize(results: list[EvalResult]) -> EvalSummary:
    n = len(results) or 1
    return EvalSummary(
        total_cases=len(results),
        passed=sum(r.passed for r in results),
        avg_retrieval_precision=sum(r.retrieval_precision or 0 for r in results) / n,
        avg_retrieval_recall=sum(r.retrieval_recall or 0 for r in results) / n,
        avg_grounding=sum(r.grounding_score or 0 for r in results) / n,
        hallucination_rate=sum(r.hallucination_flag for r in results) / n,
        avg_citation_correctness=sum(r.citation_correctness or 0 for r in results) / n,
        avg_latency_ms=sum(r.response.latency_ms for r in results) / n,
        by_case=results,
    )


def save_report(summary: EvalSummary, out_dir: str | Path = "benchmark_results") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"eval_{ts}.json"
    md_path = out_dir / f"eval_{ts}.md"

    json_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(summary), encoding="utf-8")
    logger.info(f"Saved eval report -> {json_path}")
    return json_path


def _render_markdown(summary: EvalSummary) -> str:
    lines = [
        f"# RAG Evaluation Report",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}Z_",
        "",
        f"- **Total cases:** {summary.total_cases}",
        f"- **Passed:** {summary.passed} / {summary.total_cases} "
        f"({100 * summary.passed / max(1, summary.total_cases):.1f}%)",
        f"- **Avg retrieval precision:** {summary.avg_retrieval_precision:.3f}",
        f"- **Avg retrieval recall:** {summary.avg_retrieval_recall:.3f}",
        f"- **Avg grounding score:** {summary.avg_grounding:.3f}",
        f"- **Hallucination rate:** {summary.hallucination_rate:.3f}",
        f"- **Avg citation correctness:** {summary.avg_citation_correctness:.3f}",
        f"- **Avg latency:** {summary.avg_latency_ms:.0f} ms",
        "",
        "## Per-case results",
        "",
        "| # | Query | Pass | P | R | Grounding | Hallu | Cite ✓ | Latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(summary.by_case, 1):
        q = r.case.query.replace("|", "\\|")
        if len(q) > 60:
            q = q[:57] + "…"
        lines.append(
            f"| {i} | {q} | {'✅' if r.passed else '❌'} | "
            f"{r.retrieval_precision:.2f} | {r.retrieval_recall:.2f} | "
            f"{r.grounding_score:.2f} | {'🚨' if r.hallucination_flag else '—'} | "
            f"{r.citation_correctness:.2f} | {r.response.latency_ms:.0f}ms |"
        )
    return "\n".join(lines)
