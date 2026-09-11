"""Real-time arXiv ingestion.

Fetches paper metadata (title, abstract, authors, categories, PDF URL) from
the arXiv API and upserts it into the metadata DB. PDFs are *not* downloaded
here — that happens lazily in ``documents.loader`` only for papers that
survive retrieval + reranking.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.models import Paper

logger = logging.getLogger(__name__)

try:  # the `arxiv` package wraps the Atom API with retry/backoff
    import arxiv as _arxiv

    _ARXIV_OK = True
except Exception:  # pragma: no cover
    _ARXIV_OK = False


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    abstract: str
    authors: str = ""
    categories: str = ""
    published: str = ""
    pdf_url: str = ""
    extra: dict = field(default_factory=dict)


def _clean_id(entry_id: str) -> str:
    # http://arxiv.org/abs/2503.23013v2 -> 2503.23013
    aid = entry_id.rsplit("/", 1)[-1]
    return aid.split("v")[0] if "v" in aid[-4:] else aid


def search_arxiv(query: str, max_results: int | None = None) -> list[PaperMeta]:
    """Query the live arXiv API; returns [] gracefully when offline."""
    if not _ARXIV_OK:
        logger.warning("arxiv package unavailable; skipping live fetch")
        return []
    max_results = max_results or settings.arxiv_max_results
    try:
        client = _arxiv.Client(page_size=settings.arxiv_page_size, delay_seconds=1)
        search = _arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=_arxiv.SortCriterion.Relevance,
        )
        results = []
        for r in client.results(search):
            results.append(
                PaperMeta(
                    arxiv_id=_clean_id(r.entry_id),
                    title=(r.title or "").replace("\n", " ").strip(),
                    abstract=(r.summary or "").replace("\n", " ").strip(),
                    authors="; ".join(a.name for a in r.authors),
                    categories=",".join(r.categories or []),
                    published=r.published.date().isoformat() if r.published else "",
                    pdf_url=r.pdf_url or "",
                )
            )
        return results
    except Exception as exc:  # network failures must never crash a query
        logger.warning("arXiv fetch failed (%s); continuing with local corpus", exc)
        return []


def upsert_papers(session: Session, metas: list[PaperMeta]) -> list[Paper]:
    """Insert new papers; skip ones already indexed (dedup on arxiv_id)."""
    papers: list[Paper] = []
    for m in metas:
        if not m.arxiv_id or not m.title:
            continue
        existing = session.execute(
            select(Paper).where(Paper.arxiv_id == m.arxiv_id)
        ).scalar_one_or_none()
        if existing:
            papers.append(existing)
            continue
        p = Paper(
            arxiv_id=m.arxiv_id,
            title=m.title,
            abstract=m.abstract,
            authors=m.authors,
            categories=m.categories,
            published=m.published,
            pdf_url=m.pdf_url,
        )
        session.add(p)
        papers.append(p)
    session.flush()
    return papers
