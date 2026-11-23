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
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx
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
_cache: dict[tuple, tuple[float, list[PaperMeta]]] = {}
# rate-limit back-off: no arXiv request before this monotonic time
_blocked_until: float = 0.0
_strikes: int = 0


def rate_limit_remaining() -> float:
    """Seconds left in the back-off after an arXiv 429/503 (0 = not blocked)."""
    import time as _time

    return max(0.0, _blocked_until - _time.monotonic())


def _start_backoff(resp: httpx.Response) -> float:
    """Pause live fetches after a 429/503. Retrying into a block only
    extends it, so honour Retry-After, else back off exponentially."""
    global _blocked_until, _strikes
    import time as _time

    try:
        wait = float(resp.headers.get("Retry-After", ""))
    except ValueError:
        wait = 0.0
    if wait <= 0:
        wait = min(settings.arxiv_backoff_s * (2 ** _strikes), settings.arxiv_backoff_max_s)
    _strikes += 1
    _blocked_until = _time.monotonic() + wait
    return wait


def seconds_until_next_slot() -> float:
    """How long until another uncached arXiv request is polite."""
    import time as _time

    return max(
        0.0,
        settings.arxiv_min_interval_s - (_time.monotonic() - _last_fetch_ts),
    )


def wait_for_slot(max_wait_s: float = 3.5) -> bool:
    """Sleep until the throttle allows a request (bounded). Used for the
    relaxed follow-up query, which is worth a short wait."""
    import time as _time

    wait = seconds_until_next_slot()
    if wait > max_wait_s:
        return False
    if wait > 0:
        _time.sleep(wait + 0.05)
    return True


def search_arxiv(
    query: str,
    max_results: int | None = None,
    sort_by: str = "relevance",
    id_list: list[str] | None = None,
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
    global _last_fetch_ts, _strikes
    import time as _time

    max_results = max_results or settings.arxiv_max_results
    key = (query, max_results, sort_by, tuple(id_list or ()))
    now = _time.monotonic()

    hit = _cache.get(key)
    if hit and now - hit[0] < settings.arxiv_cache_ttl_s:
        logger.info("arXiv fetch served from cache for %r", query)
        return hit[1], None

    if now < _blocked_until:
        return [], (
            "arXiv API rate-limited this computer (HTTP 429); live fetch paused "
            f"for {int(_blocked_until - now) + 1}s so the block is not extended"
        )

    if now - _last_fetch_ts < settings.arxiv_min_interval_s:
        return [], (
            "live fetch throttled to respect arXiv rate limits "
            f"(min {settings.arxiv_min_interval_s:.0f}s between requests)"
        )

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    if id_list:
        params["id_list"] = ",".join(id_list)
        if not query:
            params.pop("search_query")
    _last_fetch_ts = now
    try:
        resp = httpx.get(
            settings.arxiv_api_url,
            params=params,
            timeout=settings.arxiv_timeout_s,
            follow_redirects=True,
            headers={"User-Agent": "arxiv-semantic-search/1.0"},
        )
        if resp.status_code in (429, 503):
            wait = _start_backoff(resp)
            msg = (
                f"arXiv API rate-limited this computer (HTTP {resp.status_code}); "
                f"live fetch paused for {int(wait)}s so the block is not extended"
            )
            logger.warning("%s (query=%r)", msg, query)
            return [], msg
        resp.raise_for_status()
        _strikes = 0
        metas = parse_atom_feed(resp.text)
        _cache[key] = (now, metas)
        logger.info("arXiv fetch: %d entries for %r", len(metas), query)
        return metas, None
    except httpx.TimeoutException:
        msg = f"arXiv API timed out after {settings.arxiv_timeout_s:.0f}s"
        logger.warning("%s (query=%r)", msg, query)
        return [], msg
    except Exception as exc:  # network failures must never crash a query
        msg = f"arXiv API unreachable ({type(exc).__name__})"
        logger.warning("%s: %s (query=%r)", msg, exc, query)
        return [], msg


def fetch_by_ids(ids: list[str]) -> tuple[list[PaperMeta], str | None]:
    """Fetch metadata for explicit arXiv ids (e.g. a pasted reading list)."""
    ids = [i.strip() for i in ids if i and i.strip()]
    if not ids:
        return [], None
    wait_for_slot()
    return search_arxiv("", max_results=len(ids), id_list=ids)


# ------------------------------------------------------- OpenAlex fallback
_OPENALEX_WORKS = "https://api.openalex.org/works"
_OPENALEX_ARXIV_SOURCE = "S4306400194"          # "arXiv (Cornell University)"
_ARXIV_ID_IN_URL = re.compile(r"arxiv\.(?:org/abs/)?(\d{4}\.\d{4,5})", re.I)


def _openalex_abstract(inverted: dict | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]}; rebuild the text."""
    if not inverted:
        return ""
    slots: dict[int, str] = {}
    for word, positions in inverted.items():
        for pos in positions:
            slots[pos] = word
    return " ".join(slots[i] for i in sorted(slots))


def search_openalex(
    text: str,
    max_results: int | None = None,
    newest_first: bool = False,
    year: int | None = None,
) -> tuple[list[PaperMeta], str | None]:
    """Search arXiv papers through OpenAlex's index of arXiv.

    Used only when the arXiv API itself is rate-limiting or unreachable, so
    a search still returns fresh, on-topic arXiv papers instead of falling
    back to the small local collection. OpenAlex has no arXiv categories,
    so callers with field filters should not rely on it.
    """
    if not text.strip():
        return [], None
    max_results = max_results or settings.arxiv_max_results
    filters = [f"title_and_abstract.search:{text.replace(',', ' ')}",
               f"locations.source.id:{_OPENALEX_ARXIV_SOURCE}"]
    if year:
        filters.append(f"publication_year:{year}")
    try:
        resp = httpx.get(
            _OPENALEX_WORKS,
            params={
                "filter": ",".join(filters),
                "sort": "publication_date:desc" if newest_first else "relevance_score:desc",
                "per-page": min(max_results, 50),
                "select": "doi,title,publication_date,authorships,abstract_inverted_index,locations",
            },
            timeout=settings.arxiv_timeout_s,
            headers={"User-Agent": "arxiv-semantic-search/1.0"},
        )
        resp.raise_for_status()
    except Exception as exc:  # the fallback must never break search either
        msg = f"OpenAlex fallback unavailable ({type(exc).__name__})"
        logger.warning("%s: %s", msg, exc)
        return [], msg
    metas: list[PaperMeta] = []
    seen: set[str] = set()
    for w in resp.json().get("results", []):
        urls = [loc.get("landing_page_url") or "" for loc in w.get("locations") or []]
        urls.append(w.get("doi") or "")
        ids = _ARXIV_ID_IN_URL.findall(" ".join(urls))
        abstract = _openalex_abstract(w.get("abstract_inverted_index"))
        if not ids or not w.get("title") or not abstract or ids[0] in seen:
            continue
        aid = ids[0]
        seen.add(aid)
        authors = "; ".join(
            (a.get("author") or {}).get("display_name", "") for a in w.get("authorships") or []
        )
        metas.append(PaperMeta(
            arxiv_id=aid, title=_squash(w["title"]), abstract=_squash(abstract),
            authors=authors, categories="", published=w.get("publication_date") or "",
            pdf_url=f"https://arxiv.org/pdf/{aid}",
        ))
    logger.info("OpenAlex fallback: %d arXiv papers for %r", len(metas), text)
    return metas, None


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
            # papers first stored from the OpenAlex fallback have no arXiv
            # categories; fill them in when arXiv itself returns the paper
            if m.categories and not existing.categories:
                existing.categories = m.categories
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
