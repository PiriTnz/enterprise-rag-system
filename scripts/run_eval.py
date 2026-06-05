#!/usr/bin/env python3
"""Run the evaluation harness from the command line.

Usage:
    python scripts/run_eval.py tests/eval_cases.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import setup_logging
from app.evaluation.harness import evaluate, load_cases, save_report


def main(cases_file: str) -> int:
    setup_logging()
    cases = load_cases(cases_file)
    print(f"Loaded {len(cases)} cases from {cases_file}")
    summary = evaluate(cases)
    out = save_report(summary)
    print(f"\n--- Eval Summary ---")
    print(f"Passed:                 {summary.passed}/{summary.total_cases}")
    print(f"Avg retrieval precision: {summary.avg_retrieval_precision:.3f}")
    print(f"Avg retrieval recall:    {summary.avg_retrieval_recall:.3f}")
    print(f"Avg grounding:           {summary.avg_grounding:.3f}")
    print(f"Hallucination rate:      {summary.hallucination_rate:.3f}")
    print(f"Avg citation correctness:{summary.avg_citation_correctness:.3f}")
    print(f"Avg latency:             {summary.avg_latency_ms:.0f} ms")
    print(f"Report saved -> {out}")
    return 0 if summary.passed == summary.total_cases else 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/eval_cases.yaml"
    sys.exit(main(path))
