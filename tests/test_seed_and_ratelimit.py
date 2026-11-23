"""Tests for the offline seed corpus and arXiv rate-limit handling.

Covers the cold-start scenario from the field: live arXiv blocked/rate-
limited AND empty local corpus. The bundled seed corpus guarantees the
first search always has papers to rank, and the client now treats rate
limits as a first-class, explained condition instead of a hang.
"""
from __future__ import annotations

import json
import time

import httpx

import backend.app.services.arxiv_client as ac
from backend.app.services.arxiv_client import search_arxiv


# ------------------------------------------------------------ seed corpus
def test_seed_file_is_valid_jsonl():
    from backend.app.core.config import PROJECT_ROOT

    path = PROJECT_ROOT / "seed" / "arxiv_seed.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(recs) >= 40
    ids = [r["arxiv_id"] for r in recs]
    assert len(ids) == len(set(ids)), "seed ids must be unique"
    assert all(r["title"] and r["abstract"] and r["categories"] for r in recs)


def test_seed_ingestion_is_idempotent_and_searchable(pipeline):
    before = pipeline.bm25.size
    added = pipeline.seed_from_file()
    assert added >= 40
    assert pipeline.bm25.size == before + added
    # idempotent: re-seeding adds nothing (no-duplicate-indexing NFR)
    assert pipeline.seed_from_file() == 0

    out = pipeline.search("dense passage retrieval question answering",
                          fetch_arxiv=False)
    top = [r["arxiv_id"] for r in out["results"][:5]]
    assert "2004.04906" in top, "seeded DPR paper should rank for this query"


# ------------------------------------------------------- rate-limit client
def _reset_client_state(monkeypatch):
    monkeypatch.setattr(ac, "_last_fetch_ts", 0.0)
    monkeypatch.setattr(ac, "_cache", {})


def test_http_429_reported_as_rate_limited(monkeypatch):
    from backend.app.core.config import settings

    _reset_client_state(monkeypatch)
    monkeypatch.setattr(settings, "arxiv_min_interval_s", 0.0)
    req = httpx.Request("GET", "https://export.arxiv.org/api/query")
    monkeypatch.setattr(
        ac.httpx, "get", lambda *a, **k: httpx.Response(429, request=req)
    )
    metas, err = search_arxiv("computer science")
    assert metas == []
    assert err and "rate-limited" in err and "429" in err


def test_rapid_fetches_are_throttled(monkeypatch):
    from backend.app.core.config import settings

    _reset_client_state(monkeypatch)
    monkeypatch.setattr(settings, "arxiv_min_interval_s", 3.0)
    req = httpx.Request("GET", "https://export.arxiv.org/api/query")
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        return httpx.Response(500, request=req)

    monkeypatch.setattr(ac.httpx, "get", fake_get)
    search_arxiv("query one")            # real attempt (fails)
    metas, err = search_arxiv("query two")  # within the interval -> skipped
    assert calls["n"] == 1, "second rapid fetch must not hit the API"
    assert metas == [] and err and "throttled" in err


def test_identical_queries_served_from_cache(monkeypatch):
    from backend.app.core.config import settings

    from tests.test_arxiv_fetch import _ATOM_SAMPLE

    _reset_client_state(monkeypatch)
    monkeypatch.setattr(settings, "arxiv_min_interval_s", 0.0)
    req = httpx.Request("GET", "https://export.arxiv.org/api/query")
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        return httpx.Response(200, text=_ATOM_SAMPLE, request=req)

    monkeypatch.setattr(ac.httpx, "get", fake_get)
    t0 = time.perf_counter()
    m1, e1 = search_arxiv("dynamic alpha tuning")
    m2, e2 = search_arxiv("dynamic alpha tuning")
    assert calls["n"] == 1, "identical query within TTL must be cached"
    assert e1 is None and e2 is None
    assert [m.arxiv_id for m in m1] == [m.arxiv_id for m in m2] == ["2503.23013"]
    assert time.perf_counter() - t0 < 2


# ---------------------------------------------- 429 back-off + fallback
def test_429_pauses_fetches_instead_of_retrying_into_the_block(monkeypatch):
    from backend.app.core.config import settings

    _reset_client_state(monkeypatch)
    monkeypatch.setattr(settings, "arxiv_min_interval_s", 0.0)
    req = httpx.Request("GET", "https://export.arxiv.org/api/query")
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "120"}, request=req)

    monkeypatch.setattr(ac.httpx, "get", fake_get)
    search_arxiv("query one")
    metas, err = search_arxiv("query two")
    assert calls["n"] == 1, "no request may be sent while arXiv is blocking us"
    assert metas == [] and "paused" in err and "429" in err
    assert 100 < ac.rate_limit_remaining() <= 120, "Retry-After must be honoured"


def test_default_api_url_has_no_redirect_hops():
    from backend.app.core.config import Settings

    # arxiv.org/api/query redirects twice, tripling requests per search
    assert Settings().arxiv_api_url == "https://export.arxiv.org/api/query"


def test_openalex_results_are_parsed_into_arxiv_papers(monkeypatch):
    payload = {"results": [
        {"doi": "https://doi.org/10.48550/arxiv.2403.16971", "title": "AIOS: LLM Agent Operating System",
         "publication_date": "2024-03-25",
         "authorships": [{"author": {"display_name": "Kai Mei"}}, {"author": {"display_name": "Yongfeng Zhang"}}],
         "abstract_inverted_index": {"An": [0], "agent": [1, 4], "operating": [2], "system": [3]},
         "locations": [{"landing_page_url": "https://arxiv.org/abs/2403.16971"}]},
        {"doi": "https://doi.org/10.1000/not-arxiv", "title": "Journal only", "publication_date": "2020-01-01",
         "authorships": [], "abstract_inverted_index": {"x": [0]}, "locations": []},
    ]}
    req = httpx.Request("GET", "https://api.openalex.org/works")
    monkeypatch.setattr(ac.httpx, "get", lambda *a, **k: httpx.Response(200, json=payload, request=req))
    metas, err = ac.search_openalex("operating system")
    assert err is None and len(metas) == 1
    m = metas[0]
    assert m.arxiv_id == "2403.16971" and m.abstract == "An agent operating system agent"
    assert m.authors == "Kai Mei; Yongfeng Zhang" and m.published == "2024-03-25"


def test_rate_limited_search_falls_back_to_openalex(pipeline, monkeypatch):
    import backend.app.services.pipeline as pl
    from backend.app.core.config import settings
    from backend.app.services.arxiv_client import PaperMeta

    monkeypatch.setattr(settings, "openalex_fallback", True)
    monkeypatch.setattr(pl, "search_arxiv", lambda q, max_results=None, sort_by="relevance":
                        ([], "arXiv API rate-limited this computer (HTTP 429); live fetch paused for 60s"))
    fresh = PaperMeta(arxiv_id="2403.16971", title="AIOS: LLM Agent Operating System",
                      abstract="AIOS is an operating system that embeds large language models "
                               "into operating system kernels for agents.",
                      published="2024-03-25", pdf_url="")
    seen = {}

    def fake_openalex(text, **kw):
        seen["text"] = text
        return [fresh], None

    monkeypatch.setattr(pl, "search_openalex", fake_openalex)
    out = pipeline.search("operating system", fetch_arxiv=True)
    assert seen["text"] == "operating system"
    assert "2403.16971" in [r["arxiv_id"] for r in out["results"]]
    assert "OpenAlex" in out["notice"] and "429" in out["notice"]


def test_openalex_category_backfill_when_arxiv_returns_paper(pipeline):
    from backend.app.db.models import Paper
    from backend.app.db.session import db_session
    from backend.app.services.arxiv_client import PaperMeta
    from sqlalchemy import select

    pipeline.ingest_metadata([PaperMeta(arxiv_id="2401.00001", title="T", abstract="A", categories="")])
    pipeline.ingest_metadata([PaperMeta(arxiv_id="2401.00001", title="T", abstract="A", categories="cs.OS")])
    with db_session() as s:
        cats = s.execute(select(Paper.categories).where(Paper.arxiv_id == "2401.00001")).scalar_one()
    assert cats == "cs.OS"
