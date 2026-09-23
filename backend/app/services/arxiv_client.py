"""Real-time arXiv ingestion.

Fetches paper metadata (title, abstract, authors, categories, PDF URL) from
the arXiv Atom API and upserts it into the metadata DB. PDFs are *not*
downloaded here — that happens on demand in ``documents.loader`` only for
papers the user selects for QA.

Implementation note: this used to go through the ``arxiv`` PyPI package,
whose underlying HTTP request has **no timeout** and hidden retry sleeps —
a slow or blocked route to export.arxiv.org could hang a search for
80+ seconds. We now call the Atom API directly with httpx under a hard
timeout (``ARXIV_TIMEOUT_S``, default 8s): a search is never held hostage
by the live fetch, and failures surface as an explicit error string the
pipeline can report to the user.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.models import Paper

logger = logging.getLogger(__name__)

_ATOM = "{http://www.w3.org/2005/Atom}"


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


def _squash(text: str | None) -> str:
    return " ".join((text or "").split())


def parse_atom_feed(xml_text: str) -> list[PaperMeta]:
    """Parse an arXiv Atom API response into PaperMeta records."""
    root = ET.fromstring(xml_text)
    metas: list[PaperMeta] = []
    for entry in root.findall(f"{_ATOM}entry"):
        entry_id = entry.findtext(f"{_ATOM}id", default="")
        title = _squash(entry.findtext(f"{_ATOM}title"))
        if not entry_id or not title:
            continue
        pdf_url = ""
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
                break
        published = entry.findtext(f"{_ATOM}published", default="")
        metas.append(
            PaperMeta(
                arxiv_id=_clean_id(entry_id),
                title=title,
                abstract=_squash(entry.findtext(f"{_ATOM}summary")),
                authors="; ".join(
                    _squash(a.findtext(f"{_ATOM}name"))
                    for a in entry.findall(f"{_ATOM}author")
                ),
                categories=",".join(
                    c.get("term", "")
                    for c in entry.findall(f"{_ATOM}category")
                    if c.get("term")
                ),
                published=published[:10],
                pdf_url=pdf_url,
            )
        )
    return metas


_last_fetch_ts: float = 0.0
_cache: dict[tuple[str, int], tuple[float, list[PaperMeta]]] = {}


def search_arxiv(
    query: str, max_results: int | None = None
) -> tuple[list[PaperMeta], str | None]:
    """Query the live arXiv API with a hard timeout, throttling, and caching.

    Returns ``(metas, error)``: error is None on success (even with zero
    matches) and a short human-readable reason on failure. Never raises —
    a broken live fetch must never break search over the local corpus.

    Politeness (arXiv asks for at most ~1 request / 3s):
    * identical queries within ``ARXIV_CACHE_TTL_S`` are served from a
      local cache without touching the API;
    * uncached fetches within ``ARXIV_MIN_INTERVAL_S`` of the previous one
      are skipped with an explanatory reason instead of risking a 429;
    * HTTP 429/503 responses are reported explicitly as rate limiting.
    """
    global _last_fetch_ts
    import time as _time

    max_results = max_results or settings.arxiv_max_results
    key = (query, max_results)
    now = _time.monotonic()

    hit = _cache.get(key)
    if hit and now - hit[0] < settings.arxiv_cache_ttl_s:
        logger.info("arXiv fetch served from cache for %r", query)
        return hit[1], None

    if now - _last_fetch_ts < settings.arxiv_min_interval_s:
        return [], (
            "live fetch throttled to respect arXiv rate limits "
            f"(min {settings.arxiv_min_interval_s:.0f}s between requests)"
        )

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    _last_fetch_ts = now
    try:
        resp = requests.get(
            settings.arxiv_api_url,
            params=params,
            timeout=settings.arxiv_timeout_s,
            allow_redirects=True,
            headers={"User-Agent": "arxiv-semantic-search/1.0"},
        )
        if resp.status_code in (429, 503):
            msg = (
                f"arXiv API rate-limited (HTTP {resp.status_code}) — "
                "wait a minute before fetching again"
            )
            logger.warning("%s (query=%r)", msg, query)
            return [], msg
        resp.raise_for_status()
        metas = parse_atom_feed(resp.text)
        _cache[key] = (now, metas)
        logger.info("arXiv fetch: %d entries for %r", len(metas), query)
        return metas, None
    except requests.Timeout:
        msg = f"arXiv API timed out after {settings.arxiv_timeout_s:.0f}s"
        logger.warning("%s (query=%r)", msg, query)
        return [], msg
    except Exception as exc:  # network failures must never crash a query
        msg = f"arXiv API unreachable ({type(exc).__name__})"
        logger.warning("%s: %s (query=%r)", msg, exc, query)
        return [], msg


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
