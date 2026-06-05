"""End-to-end ingestion pipeline.

Walks documents, loads them, chunks, embeds, and indexes into BOTH the vector
store and the BM25 index — keeping them synchronized.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Iterable

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.chunker import Chunker
from app.ingestion.embedder import get_embedder
from app.ingestion.loaders import get_loader
from app.retrieval.bm25 import get_bm25
from app.retrieval.vector_store import get_vector_store
from app.schemas.models import (
    Chunk,
    DocumentType,
    IngestionResult,
)

logger = get_logger(__name__)


_SUFFIX_TO_TYPE = {
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.DOCX,
    ".txt": DocumentType.TXT,
    ".md": DocumentType.TXT,
    ".csv": DocumentType.CSV,
}


def _doc_id_for(path: Path) -> str:
    return hashlib.md5(str(path.resolve()).encode()).hexdigest()[:12]


class IngestionPipeline:
    def __init__(self) -> None:
        self.chunker = Chunker()
        self.embedder = get_embedder()
        self.vector_store = get_vector_store()
        self.bm25 = get_bm25()

    def _iter_files(self, paths: Iterable[str | Path]) -> Iterable[Path]:
        for p in paths:
            path = Path(p)
            if path.is_dir():
                for sub in path.rglob("*"):
                    if sub.is_file() and sub.suffix.lower() in _SUFFIX_TO_TYPE:
                        yield sub
            elif path.is_file() and path.suffix.lower() in _SUFFIX_TO_TYPE:
                yield path
            else:
                logger.warning(f"Skipping unsupported or missing path: {p}")

    def ingest(
        self,
        paths: Iterable[str | Path] | None = None,
        rebuild: bool = False,
    ) -> IngestionResult:
        t0 = time.perf_counter()
        if paths is None:
            paths = [Path(settings.documents_dir)]

        if rebuild:
            logger.warning("Rebuild requested — clearing existing indices")
            self.vector_store.delete_all()
            self.bm25.clear()

        all_chunks: list[Chunk] = []
        failures: list[dict[str, str]] = []
        docs_processed = 0

        for file in self._iter_files(paths):
            try:
                loader = get_loader(file)
                segments = list(loader.load(file))
                if not segments:
                    failures.append({"path": str(file), "reason": "no extractable text"})
                    continue
                doc_id = _doc_id_for(file)
                chunks = self.chunker.chunk_segments(
                    segments,
                    source=file.name,
                    doc_id=doc_id,
                    doc_type=_SUFFIX_TO_TYPE[file.suffix.lower()],
                )
                all_chunks.extend(chunks)
                docs_processed += 1
                logger.info(f"Loaded {file.name}: {len(segments)} segments -> {len(chunks)} chunks")
            except Exception as e:
                logger.exception(f"Failed processing {file}")
                failures.append({"path": str(file), "reason": str(e)})

        # Embed + index
        indexed = 0
        if all_chunks:
            batch = settings.embedding_batch_size * 8
            for i in range(0, len(all_chunks), batch):
                batch_chunks = all_chunks[i : i + batch]
                embeddings = self.embedder.embed([c.text for c in batch_chunks])
                self.vector_store.upsert(batch_chunks, embeddings)
                indexed += len(batch_chunks)
            self.bm25.add(all_chunks)

        duration = time.perf_counter() - t0
        logger.info(
            f"Ingestion complete docs={docs_processed} chunks={len(all_chunks)} "
            f"indexed={indexed} failures={len(failures)} duration={duration:.2f}s"
        )
        return IngestionResult(
            documents_processed=docs_processed,
            chunks_created=len(all_chunks),
            chunks_indexed=indexed,
            failures=failures,
            duration_seconds=duration,
        )
