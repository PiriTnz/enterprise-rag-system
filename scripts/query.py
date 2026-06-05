#!/usr/bin/env python3
"""Ask a question from the command line.

Usage:
    python scripts/query.py "What caused the Q3 capacity issues?"
    python scripts/query.py --type summary "key points from planning report"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import setup_logging
from app.generation.pipeline import get_pipeline
from app.schemas.models import QueryRequest, StructuredOutputType


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--type", default="answer",
                        choices=[t.value for t in StructuredOutputType])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--json", action="store_true",
                        help="emit full JSON response")
    args = parser.parse_args()

    setup_logging()
    pipeline = get_pipeline()
    resp = pipeline.run(QueryRequest(
        query=args.query,
        output_type=StructuredOutputType(args.type),
        top_k=args.top_k,
        llm_provider=args.provider,
        enable_reranking=not args.no_rerank,
        include_chunks=False,
    ))

    if args.json:
        print(resp.model_dump_json(indent=2))
        return 0

    print(f"\n>> {resp.answer}\n")
    print(f"  provider: {resp.provider_used}  confidence: {resp.confidence:.2f}  "
          f"latency: {resp.latency_ms:.0f}ms")
    if resp.citations:
        print("\nCitations:")
        for c in resp.citations:
            page = f", p.{c.page}" if c.page else ""
            print(f"  - {c.source}{page}  (score {c.score:.2f})")
            print(f"      \"{c.snippet}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
