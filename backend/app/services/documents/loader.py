"""Lazy PDF loading and parsing.

PDFs are downloaded *only* when a paper survives hybrid retrieval +
reranking (the report's lazy-loading objective). Parsing uses PyMuPDF and
must never crash the pipeline: on any failure the paper falls back to
abstract-only mode and the error is recorded on the Paper row.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

from backend.app.core.config import settings
from backend.app.db.models import Paper

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF

    _FITZ_OK = True
except Exception:  # pragma: no cover
    _FITZ_OK = False


@dataclass
class ParsedDocument:
    arxiv_id: str
    # list of (heading, text) preserving document order; heading may be ""
    blocks: list[tuple[str, str]]
    n_pages: int = 0
    source: str = "pdf"  # "pdf" | "abstract"


def _pdf_path(arxiv_id: str) -> Path:
    return settings.pdf_dir / f"{arxiv_id.replace('/', '_')}.pdf"


def download_pdf(paper: Paper, timeout: int | None = None) -> Path | None:
    """Idempotent download; returns local path or None on failure."""
    path = _pdf_path(paper.arxiv_id)
    if path.exists() and path.stat().st_size > 1024:
        return path
    url = paper.pdf_url or f"https://arxiv.org/pdf/{paper.arxiv_id}"
    try:
        with httpx.Client(
            follow_redirects=True, timeout=timeout or settings.request_timeout_s
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            if not resp.content[:5].startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            path.write_bytes(resp.content)
        logger.info("Downloaded %s (%.1f KB)", paper.arxiv_id, len(resp.content) / 1024)
        return path
    except Exception as exc:
        logger.warning("PDF download failed for %s: %s", paper.arxiv_id, exc)
        paper.parse_error = f"download: {exc}"
        return None


def parse_pdf(path: Path, arxiv_id: str) -> ParsedDocument | None:
    """Extract text blocks with font-size-aware heading detection."""
    if not _FITZ_OK:
        return None
    try:
        doc = fitz.open(path)
        sizes: list[float] = []
        raw_lines: list[tuple[float, str]] = []  # (max_font_size, line_text)
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if not text:
                        continue
                    size = max(s.get("size", 0.0) for s in spans)
                    raw_lines.append((size, text))
                    sizes.append(size)
        n_pages = doc.page_count
        doc.close()
        if not raw_lines:
            return None

        body_size = _median(sizes)
        blocks: list[tuple[str, str]] = []
        current_heading = ""
        buf: list[str] = []
        for size, text in raw_lines:
            # Heading heuristic: notably larger font AND short line
            # (line-level, since headings often share blocks with body text).
            if size > body_size * 1.15 and len(text) < 120:
                if buf:
                    blocks.append((current_heading, " ".join(buf)))
                    buf = []
                current_heading = text
            else:
                buf.append(text)
        if buf:
            blocks.append((current_heading, " ".join(buf)))
        return ParsedDocument(arxiv_id=arxiv_id, blocks=blocks, n_pages=n_pages)
    except Exception as exc:
        logger.warning("PDF parse failed for %s: %s", arxiv_id, exc)
        return None


def _median(values: list[float]) -> float:
    if not values:
        return 10.0
    s = sorted(values)
    return s[len(s) // 2]


def load_document(paper: Paper) -> ParsedDocument:
    """Full lazy pipeline for one paper: download -> parse -> fallback.

    Never raises; on failure returns an abstract-only document so QA can
    still answer from metadata (reliability NFR).
    """
    path = download_pdf(paper)
    if path is not None:
        parsed = parse_pdf(path, paper.arxiv_id)
        if parsed and parsed.blocks:
            paper.pdf_status = "parsed"
            paper.pdf_path = str(path)
            paper.parse_error = ""
            return parsed
        paper.parse_error = paper.parse_error or "parse: no extractable text"
    paper.pdf_status = "failed"
    logger.info("Falling back to abstract-only for %s", paper.arxiv_id)
    return ParsedDocument(
        arxiv_id=paper.arxiv_id,
        blocks=[("Abstract", paper.abstract or paper.title)],
        source="abstract",
    )
