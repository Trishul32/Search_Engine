"""Citation data for the research map (ResearchRabbit-style timeline).

For a set of arXiv papers this returns, per paper: citation count, year,
first author, a one-line TL;DR, and which papers of the set cite which
(edges). It also surfaces "common foundations": earlier papers that two or
more of the set cite, i.e. the shared background a reader should know.

Sources, in order:
1. Semantic Scholar batch API (one request; TL;DRs; arXiv ids on refs).
   It is rate-limited for anonymous use, so a 429 gets one short retry.
2. OpenAlex (free, generous limits) as fallback: counts + in-set edges.
3. Local metadata only (year/authors), citations unknown.
Results are cached per paper for a week in ``paper_citations``.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
from collections import Counter

import httpx
from sqlalchemy import select

from backend.app.db.models import Paper, PaperCitations
from backend.app.db.session import db_session

logger = logging.getLogger(__name__)

_S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
_S2_FIELDS = (
    "citationCount,year,authors,tldr,"
    "references.externalIds,references.title,references.year,references.citationCount"
)
_OPENALEX = "https://api.openalex.org/works"
_TTL = dt.timedelta(days=7)
_HEADERS = {"User-Agent": "arxiv-semantic-search/1.0 (research map)"}


def _first_author_surname(authors: str) -> str:
    first = (authors or "").split(";")[0].strip()
    return first.split()[-1] if first else ""


# ------------------------------------------------------------- providers
def _from_semantic_scholar(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for attempt in range(2):
        try:
            r = httpx.post(
                _S2_BATCH, params={"fields": _S2_FIELDS},
                json={"ids": [f"ARXIV:{i}" for i in ids]}, timeout=20, headers=_HEADERS,
            )
        except Exception as exc:
            logger.info("Semantic Scholar unreachable: %s", exc)
            return {}
        if r.status_code == 429 and attempt == 0:
            time.sleep(1.5)
            continue
        if r.status_code != 200:
            logger.info("Semantic Scholar HTTP %s", r.status_code)
            return {}
        for pid, p in zip(ids, r.json()):
            if not p:
                continue
            refs = []
            for ref in p.get("references") or []:
                ext = ref.get("externalIds") or {}
                refs.append({
                    "arxiv_id": ext.get("ArXiv"),
                    "key": ext.get("DOI") or ext.get("ArXiv") or ref.get("title", "").lower(),
                    "title": ref.get("title") or "",
                    "year": ref.get("year"),
                    "citations": ref.get("citationCount"),
                })
            authors = [a.get("name", "") for a in p.get("authors") or []]
            out[pid] = {
                "citations": p.get("citationCount"),
                "year": p.get("year"),
                "first_author": authors[0].split()[-1] if authors else "",
                "tldr": (p.get("tldr") or {}).get("text") or "",
                "references": refs,
                "source": "Semantic Scholar",
            }
        return out
    return out


_ARXIV_IN_URL = re.compile(r"arxiv\.(?:org/abs/)?(\d{4}\.\d{4,5})", re.I)


def _from_openalex(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for start in range(0, len(ids), 40):
        batch = ids[start:start + 40]
        try:
            r = httpx.get(
                _OPENALEX,
                params={
                    "filter": "doi:" + "|".join(f"10.48550/arxiv.{i}" for i in batch),
                    "per-page": 50,
                    "select": "id,doi,cited_by_count,publication_year,authorships,referenced_works,locations",
                },
                timeout=20, headers=_HEADERS,
            )
            r.raise_for_status()
        except Exception as exc:
            logger.info("OpenAlex unavailable: %s", exc)
            return out
        for w in r.json().get("results", []):
            urls = [l.get("landing_page_url") or "" for l in w.get("locations") or []] + [w.get("doi") or ""]
            found = {m for u in urls for m in _ARXIV_IN_URL.findall(u)} & set(batch)
            if not found:
                continue
            first = ((w.get("authorships") or [{}])[0].get("author") or {}).get("display_name", "")
            for pid in found:
                out[pid] = {
                    "citations": w.get("cited_by_count"),
                    "year": w.get("publication_year"),
                    "first_author": first.split()[-1] if first else "",
                    "tldr": "",
                    "openalex_id": w.get("id"),
                    "openalex_refs": w.get("referenced_works") or [],
                    "references": [],
                    "source": "OpenAlex",
                }
    # in-set references via OpenAlex ids
    by_oa = {v["openalex_id"]: pid for pid, v in out.items() if v.get("openalex_id")}
    for pid, v in out.items():
        v["references"] = [{"arxiv_id": by_oa[x], "key": x} for x in v.pop("openalex_refs", []) if x in by_oa]
        v.pop("openalex_id", None)
    return out


# ----------------------------------------------------------------- cache
def _cached(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    try:
        with db_session() as s:
            rows = s.execute(select(PaperCitations).where(PaperCitations.arxiv_id.in_(ids))).scalars().all()
            now = dt.datetime.now()
            for r in rows:
                fetched = r.fetched_at.replace(tzinfo=None) if r.fetched_at else now - _TTL * 2
                if now - fetched < _TTL:
                    out[r.arxiv_id] = json.loads(r.data)
    except Exception:
        logger.debug("citation cache read failed", exc_info=True)
    return out


def _store(data: dict[str, dict]) -> None:
    try:
        with db_session() as s:
            for pid, v in data.items():
                row = s.execute(select(PaperCitations).where(PaperCitations.arxiv_id == pid)).scalar_one_or_none()
                if row is None:
                    s.add(PaperCitations(arxiv_id=pid, data=json.dumps(v)))
                else:
                    row.data = json.dumps(v)
                    row.fetched_at = dt.datetime.now()
    except Exception:
        logger.debug("citation cache write failed", exc_info=True)


def citation_info(ids: list[str], online: bool = True) -> dict[str, dict]:
    ids = list(dict.fromkeys(ids))
    info = _cached(ids)
    missing = [i for i in ids if i not in info]
    if missing and online:
        fresh = _from_semantic_scholar(missing)
        still = [i for i in missing if i not in fresh]
        if still:
            fresh.update(_from_openalex(still))
        if fresh:
            _store(fresh)
            info.update(fresh)
    return info


# ------------------------------------------------------------------- map
def citation_map(paper_ids: list[str], online: bool = True, max_foundations: int = 6) -> dict:
    """Nodes (papers + shared foundations) and citation edges for a set."""
    with db_session() as s:
        rows = s.execute(select(Paper).where(Paper.arxiv_id.in_(paper_ids))).scalars().all()
        papers = {p.arxiv_id: p.to_dict() for p in rows}
    ids = [i for i in paper_ids if i in papers]
    info = citation_info(ids, online=online)
    nodes, edges = [], []
    in_set = set(ids)
    for pid in ids:
        p, c = papers[pid], info.get(pid, {})
        year = c.get("year") or (int(p["published"][:4]) if (p.get("published") or "")[:4].isdigit() else None)
        nodes.append({
            "id": pid, "kind": "paper", "title": p["title"],
            "label": f"{c.get('first_author') or _first_author_surname(p.get('authors', ''))} {year or ''}".strip(),
            "year": year, "published": p.get("published", ""),
            "citations": c.get("citations"), "tldr": c.get("tldr", ""),
            "authors": p.get("authors", ""), "categories": p.get("categories", ""),
            "abstract": (p.get("abstract") or "")[:600],
            "url": f"https://arxiv.org/abs/{pid}",
        })
        for ref in c.get("references", []):
            if ref.get("arxiv_id") in in_set and ref["arxiv_id"] != pid:
                edges.append({"source": pid, "target": ref["arxiv_id"], "kind": "cites"})

    # common foundations: outside papers cited by 2+ papers of the set
    counter: Counter = Counter()
    meta: dict[str, dict] = {}
    for pid in ids:
        seen = set()
        for ref in info.get(pid, {}).get("references", []):
            key = ref.get("key")
            if not key or ref.get("arxiv_id") in in_set or key in seen or not ref.get("title"):
                continue
            seen.add(key)
            counter[key] += 1
            meta.setdefault(key, ref)
            meta[key].setdefault("cited_by", []).append(pid)
    shared = [k for k, n in counter.most_common() if n >= 2][:max_foundations]
    for key in shared:
        ref = meta[key]
        fid = f"ref:{key}"
        nodes.append({
            "id": fid, "kind": "foundation", "title": ref["title"],
            "label": (ref["title"].split(":")[0][:28] + ("…" if len(ref["title"].split(":")[0]) > 28 else "")),
            "year": ref.get("year"), "citations": ref.get("citations"), "tldr": "",
            "url": f"https://arxiv.org/abs/{ref['arxiv_id']}" if ref.get("arxiv_id")
                   else f"https://www.semanticscholar.org/search?q={ref['title']}",
            "cited_by": ref.get("cited_by", []),
        })
        for pid in ref.get("cited_by", []):
            edges.append({"source": pid, "target": fid, "kind": "cites"})

    sources = sorted({v.get("source") for v in info.values() if v.get("source")})
    return {
        "nodes": nodes, "edges": edges,
        "source": ", ".join(sources) or "local metadata only",
        "missing_citations": [n["id"] for n in nodes if n["kind"] == "paper" and n["citations"] is None],
    }
