"""Chunking. Recursive token-aware splitter with overlap.

Why not just naive char-split? Because retrieval quality is sensitive to
chunk boundaries — splitting mid-sentence wrecks semantic embeddings. So we
split by paragraph -> sentence -> word, keeping chunks under a token budget.

We approximate tokens with a fast tiktoken encoding if available, otherwise
fall back to ~4 chars/token.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.loaders import LoadedSegment
from app.schemas.models import Chunk, ChunkMetadata, DocumentType

logger = get_logger(__name__)


# ---- token counter ----
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text, disallowed_special=()))
except Exception:  # pragma: no cover
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


_PARA_SPLIT = re.compile(r"\n{2,}")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u0600-\u06FF])")


def _split_recursive(text: str, max_tokens: int) -> list[str]:
    """Split text into pieces each <= max_tokens, respecting paragraph then
    sentence boundaries before resorting to word-level."""
    if count_tokens(text) <= max_tokens:
        return [text]

    parts = _PARA_SPLIT.split(text)
    if len(parts) > 1:
        out: list[str] = []
        for p in parts:
            out.extend(_split_recursive(p, max_tokens))
        return out

    parts = _SENT_SPLIT.split(text)
    if len(parts) > 1:
        # Greedily group sentences up to budget
        out, buf, buf_tok = [], [], 0
        for s in parts:
            t = count_tokens(s)
            if buf and buf_tok + t > max_tokens:
                out.append(" ".join(buf))
                buf, buf_tok = [], 0
            buf.append(s)
            buf_tok += t
        if buf:
            out.append(" ".join(buf))
        return out

    # last resort: words
    words = text.split()
    out, buf, buf_tok = [], [], 0
    for w in words:
        t = count_tokens(w)
        if buf and buf_tok + t > max_tokens:
            out.append(" ".join(buf))
            buf, buf_tok = [], 0
        buf.append(w)
        buf_tok += t
    if buf:
        out.append(" ".join(buf))
    return out


@dataclass
class ChunkingConfig:
    chunk_size: int = settings.chunk_size
    chunk_overlap: int = settings.chunk_overlap
    min_chunk_chars: int = settings.min_chunk_chars


class Chunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk_segments(
        self,
        segments: list[LoadedSegment],
        *,
        source: str,
        doc_id: str,
        doc_type: DocumentType,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_idx = 0

        for seg in segments:
            pieces = _split_recursive(seg.text, self.config.chunk_size)
            pieces = self._add_overlap(pieces)

            for piece in pieces:
                piece = piece.strip()
                if len(piece) < self.config.min_chunk_chars:
                    continue
                meta = ChunkMetadata(
                    source=source,
                    doc_id=doc_id,
                    doc_type=doc_type,
                    page=seg.page,
                    section=seg.section,
                    chunk_index=chunk_idx,
                    **seg.extra,
                )
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}::c{chunk_idx}::{uuid.uuid4().hex[:8]}",
                    text=piece,
                    metadata=meta,
                ))
                chunk_idx += 1

        logger.debug(f"Chunked {source} into {len(chunks)} chunks")
        return chunks

    def _add_overlap(self, pieces: list[str]) -> list[str]:
        if len(pieces) < 2 or self.config.chunk_overlap <= 0:
            return pieces
        out = [pieces[0]]
        for prev, cur in zip(pieces, pieces[1:]):
            tail_tokens = self.config.chunk_overlap
            prev_tokens = prev.split()
            tail = " ".join(prev_tokens[-tail_tokens:])
            out.append((tail + " " + cur).strip())
        return out
