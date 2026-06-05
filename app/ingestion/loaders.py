"""Document loaders. Each loader returns a list of (text, metadata_extras) pairs.

We keep the loader interface narrow so adding a new format (HTML, MD, EML, etc.)
is just one new class. The pipeline downstream is format-agnostic.
"""
from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from app.core.logging import get_logger
from app.schemas.models import DocumentType

logger = get_logger(__name__)


class LoadedSegment:
    """A piece of text plus structural metadata (page, section, row...)."""
    __slots__ = ("text", "page", "section", "extra")

    def __init__(
        self,
        text: str,
        page: int | None = None,
        section: str | None = None,
        extra: dict | None = None,
    ) -> None:
        self.text = text
        self.page = page
        self.section = section
        self.extra = extra or {}


class BaseLoader(ABC):
    doc_type: DocumentType

    @abstractmethod
    def load(self, path: Path) -> Iterator[LoadedSegment]:
        ...


class PDFLoader(BaseLoader):
    doc_type = DocumentType.PDF

    def load(self, path: Path) -> Iterator[LoadedSegment]:
        # Lazy import so the package isn't required when only TXT is used.
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("pypdf is required for PDF support") from e

        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                yield LoadedSegment(text=text, page=i)


class DOCXLoader(BaseLoader):
    doc_type = DocumentType.DOCX

    def load(self, path: Path) -> Iterator[LoadedSegment]:
        try:
            from docx import Document
        except ImportError as e:
            raise RuntimeError("python-docx is required for DOCX support") from e

        doc = Document(str(path))
        current_section: str | None = None
        buffer: list[str] = []

        def flush() -> Iterator[LoadedSegment]:
            if buffer:
                yield LoadedSegment(text="\n".join(buffer), section=current_section)

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "").lower() if para.style else ""
            if "heading" in style:
                # flush previous section
                yield from flush()
                buffer.clear()
                current_section = text
            else:
                buffer.append(text)

        yield from flush()


class TXTLoader(BaseLoader):
    doc_type = DocumentType.TXT

    def load(self, path: Path) -> Iterator[LoadedSegment]:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            yield LoadedSegment(text=text)


class CSVLoader(BaseLoader):
    """Treats each row as a small segment. Header row provides field names so
    the LLM sees structured data, not raw commas."""
    doc_type = DocumentType.CSV

    def load(self, path: Path) -> Iterator[LoadedSegment]:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=1):
                parts = [f"{k}: {v}" for k, v in row.items() if v]
                if parts:
                    yield LoadedSegment(
                        text=" | ".join(parts),
                        section=f"row_{row_idx}",
                        extra={"row_index": row_idx},
                    )


_LOADERS: dict[str, type[BaseLoader]] = {
    ".pdf": PDFLoader,
    ".docx": DOCXLoader,
    ".txt": TXTLoader,
    ".md": TXTLoader,
    ".csv": CSVLoader,
}


def get_loader(path: Path) -> BaseLoader:
    suffix = path.suffix.lower()
    if suffix not in _LOADERS:
        raise ValueError(f"Unsupported file type: {suffix} ({path.name})")
    return _LOADERS[suffix]()
