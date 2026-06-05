#!/usr/bin/env python3
"""Ingest documents from a path (file or directory) into the indices.

Usage:
    python scripts/ingest.py                       # ingests `documents/`
    python scripts/ingest.py path/to/dir
    python scripts/ingest.py --rebuild path/to/dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import setup_logging
from app.ingestion.pipeline import IngestionPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="files or directories to ingest")
    parser.add_argument("--rebuild", action="store_true",
                        help="wipe indices before ingesting")
    args = parser.parse_args()

    setup_logging()
    pipeline = IngestionPipeline()
    paths = args.paths or ["documents"]
    res = pipeline.ingest(paths=paths, rebuild=args.rebuild)
    print(res.model_dump_json(indent=2))
    return 0 if not res.failures else 1


if __name__ == "__main__":
    sys.exit(main())
