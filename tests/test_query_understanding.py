"""Query understanding: exclusions ("without AI"), research fields, the
structured arXiv query, and hard filtering in the search pipeline."""
from __future__ import annotations

import pytest

from backend.app.services.query_processor import build_arxiv_query, process_query


# ---------------------------------------------------------------- parsing
def test_field_with_exclusion_is_understood():
    pq = process_query("paper on computer science without AI")
    assert [f.categories for f in pq.fields] == [["cs.*"]]
    assert [e.label for e in pq.exclusions] == ["AI"]
    ai = pq.exclusions[0]
    assert "cs.AI" in ai.categories and "cs.LG" in ai.categories
    assert "machine learning" in ai.phrases and "deep learning" in ai.phrases
    # nothing positive left except the field -> browse the field
    assert pq.browse_only
    assert "ai" not in pq.dense_query.lower().split()


@pytest.mark.parametrize(
    "query, topic, excluded",
    [
        ("graph algorithms without machine learning or deep learning",
         "graph algorithms", ["machine learning", "deep learning"]),
        ("retrieval augmented generation but not LLMs",
         "retrieval augmented generation", ["LLMs"]),
        ("networking -blockchain", "networking", ["blockchain"]),
        ("database indexing NOT learned", "database indexing", ["learned"]),
        ("non-AI approaches to compiler optimization",
         "approaches to compiler optimization", ["AI"]),
        ("computer networking, no AI", "computer networking", ["AI"]),
        ("image segmentation excluding the use of transformers",
         "image segmentation", ["transformers"]),
        ("graph algorithms not about AI", "graph algorithms", ["AI"]),
    ],
)
def test_exclusion_phrasings(query, topic, excluded):
    pq = process_query(query)
    assert pq.topic == topic
    assert [e.label for e in pq.exclusions] == excluded


@pytest.mark.parametrize(
    "query",
    [
        "how can robots learn new skills without forgetting the old ones",
        "self-supervised learning without labels",
        "sorting without comparisons",
        "how do llms not hallucinate",
    ],
)
def test_descriptive_negation_stays_in_topic(query):
    """'without X' describing the method is part of the topic, not a filter."""
    pq = process_query(query)
    assert pq.exclusions == []


def test_paper_framed_without_is_an_exclusion():
    pq = process_query("papers on networking without labels")
    assert [e.label for e in pq.exclusions] == ["labels"]


def test_single_word_field_only_when_used_as_field():
    assert process_query("papers in physics without quantum").fields
    assert not process_query("physics-informed neural networks").fields


def test_explicit_ui_filters_merge_with_query():
    pq = process_query("image segmentation", exclude=["transformers"], field_filter="computer science")
    assert [e.label for e in pq.exclusions] == ["transformers"]
    assert [f.label for f in pq.fields] == ["computer science"]


# ------------------------------------------------------------ arXiv query
def test_arxiv_query_targets_title_and_abstract_with_filters():
    pq = process_query("graph algorithms in computer science without machine learning")
    q = build_arxiv_query(pq)
    assert 'abs:"graph algorithms"' in q and 'ti:"graph algorithms"' in q
    assert '(abs:"graph" AND abs:"algorithms")' in q      # abstract word match
    assert "cat:cs.*" in q
    assert q.count("ANDNOT") == 1 and 'abs:"machine learning"' in q.split("ANDNOT")[1]
    assert "cat:cs.LG" in q.split("ANDNOT")[1]


def test_arxiv_query_strict_vs_relaxed_and_year():
    pq = process_query("reinforcement learning for congestion control in 2024")
    strict, relaxed = build_arxiv_query(pq), build_arxiv_query(pq, relaxed=True)
    assert ") AND (" in strict and ") OR (" in relaxed
    assert "submittedDate:[202401010000 TO 202412312359]" in strict


def test_arxiv_query_empty_when_only_exclusions():
    assert build_arxiv_query(process_query("papers without AI")) == ""


# ------------------------------------------------------------ pipeline
_NON_AI = [
    ("7001.00001", "Faster Shortest Paths in Planar Graphs",
     "We give a near-linear time algorithm for shortest paths in planar graphs.", "cs.DS"),
    ("7001.00002", "Verified Compilation of Concurrent Programs",
     "A verified compiler for concurrent programs with a mechanised proof of correctness.", "cs.PL"),
]
_AI = [
    ("7001.00003", "Graph Algorithms Accelerated by Deep Learning",
     "We use deep learning to speed up classic graph algorithms such as shortest paths.", "cs.DS"),
    ("7001.00004", "Neural Program Synthesis",
     "Program synthesis with neural networks for concurrent programs.", "cs.LG,cs.PL"),
]


def _ingest(pipeline, rows, published="2031-01-01"):
    from backend.app.services.arxiv_client import PaperMeta

    pipeline.ingest_metadata([
        PaperMeta(arxiv_id=i, title=t, abstract=a, categories=c, published=published)
        for i, t, a, c in rows
    ])


def test_search_without_ai_never_returns_ai_papers(pipeline):
    _ingest(pipeline, _NON_AI + _AI)
    out = pipeline.search("paper on computer science without AI", top_k=20, fetch_arxiv=False)
    ids = {r["arxiv_id"] for r in out["results"]}
    assert {"7001.00001", "7001.00002"} <= ids
    assert not ids & {"7001.00003", "7001.00004"}
    assert out["excluded_count"] >= 2
    assert out["interpretation"]["excluded"][0]["label"] == "AI"
    from backend.app.services.relevance import exclusion_hit
    pq = process_query("x without AI")
    assert all(exclusion_hit(pq.exclusions, r) is None for r in out["results"])


def test_topic_search_with_exclusion_drops_synonyms(pipeline):
    _ingest(pipeline, _NON_AI + _AI)
    out = pipeline.search("shortest paths graph algorithms without machine learning",
                          top_k=10, fetch_arxiv=False)
    ids = [r["arxiv_id"] for r in out["results"]]
    assert "7001.00001" in ids
    assert "7001.00003" not in ids          # mentions deep learning (a synonym)


def test_only_exclusions_returns_guidance(pipeline):
    out = pipeline.search("papers without AI", fetch_arxiv=False)
    assert out["results"] == [] and "only says what to exclude" in out["notice"]


def test_live_arxiv_matches_always_reach_the_reranker(pipeline, monkeypatch):
    """A paper the structured arXiv query found (by its abstract) is scored
    even when local hybrid retrieval does not surface it."""
    import backend.app.services.pipeline as pl
    from backend.app.services.arxiv_client import PaperMeta

    live = PaperMeta(
        arxiv_id="7001.00009", title="An Unrelated-Sounding Title",
        abstract="We study tidal energy harvesting with oscillating hydrofoils.",
        categories="physics.flu-dyn",
    )
    seen: list[str] = []

    def fake_search(q, max_results=None):
        seen.append(q)
        return [live], None

    monkeypatch.setattr(pl, "search_arxiv", fake_search)
    monkeypatch.setattr(pipeline.hybrid, "retrieve", lambda _pq: [])
    out = pipeline.search("tidal energy harvesting hydrofoils", top_k=5, fetch_arxiv=True)
    # strict query first (all concepts in title/abstract); a thin result
    # (1 paper) for a long concept may add a relaxed follow-up query
    assert 'abs:"tidal energy harvesting hydrofoils"' in seen[0]
    assert out["results"] and out["results"][0]["arxiv_id"] == "7001.00009"
    assert out["results"][0]["live_match"] is True
    assert out["live_query"] == seen


def test_results_carry_best_matching_abstract_sentence(pipeline):
    _ingest(pipeline, [(
        "7001.00010", "A Study",
        "This paper has several parts. We propose dynamic alpha tuning for hybrid "
        "retrieval fusion. The rest is about something else entirely.", "cs.IR",
    )])
    out = pipeline.search("dynamic alpha tuning hybrid retrieval", top_k=20, fetch_arxiv=False)
    hit = next(r for r in out["results"] if r["arxiv_id"] == "7001.00010")
    assert "dynamic alpha tuning" in hit["evidence"]


def test_llm_rewrite_adds_vocabulary_without_overriding_user(pipeline, monkeypatch):
    import backend.app.services.pipeline as pl
    from backend.app.services.query_rewriter import Rewrite

    monkeypatch.setattr(pl, "needs_rewrite", lambda pq: True)
    monkeypatch.setattr(
        pl, "rewrite_query",
        lambda topic, exclusions=None: Rewrite(
            academic_query="hybrid retrieval fusion",
            keywords=["dynamic alpha tuning"],
            pseudo_abstract="We tune the fusion weight per query.",
        ),
    )
    out = pipeline.search("mixing keyword and meaning based search per question",
                          top_k=5, fetch_arxiv=False)
    rw = out["interpretation"]["rewrite"]
    assert rw["keywords"] == ["dynamic alpha tuning"]
    assert out["query"] == "mixing keyword and meaning based search per question"


def test_rewrite_drops_terms_hitting_exclusions(monkeypatch):
    import backend.app.services.query_rewriter as qr

    monkeypatch.setattr(qr, "_rewrite_cached", lambda topic: qr.Rewrite(
        academic_query="machine learning for routing", keywords=["deep learning", "shortest paths"],
        pseudo_abstract="",
    ))
    pq = process_query("fast route finding without AI")
    rw = qr.rewrite_query(pq.topic, pq.exclusions)
    assert rw.keywords == ["shortest paths"]
    assert rw.academic_query == ""               # mentions machine learning -> dropped


def test_embedder_change_resets_persisted_index(tmp_path):
    import numpy as np

    from backend.app.services.vector_store import FaissStore

    a = FaissStore("t", 4, tmp_path, signature="hash:4")
    a.add(np.eye(4, dtype="float32")[:1], [{"arxiv_id": "x"}])
    a.save()
    same = FaissStore("t", 4, tmp_path, signature="hash:4")
    assert same.ntotal == 1 and same.reset_reason is None
    other = FaissStore("t", 4, tmp_path, signature="st:model:4")
    assert other.ntotal == 0 and "embedder changed" in other.reset_reason


def test_field_browse_skips_reranker_and_sorts_newest_first(pipeline, monkeypatch):
    """A field with no topic has nothing to rank against: no cross-encoder
    call (it cost 10-20s), newest papers first, and no relevance score."""
    # fresh ids: earlier tests ingested _NON_AI with other dates (dedup)
    _ingest(pipeline, [("7002.00001", "Older CS Paper", "Planar graph routing.", "cs.DS")],
            published="2032-05-01")
    _ingest(pipeline, [("7002.00002", "Newest CS Paper", "Compiler verification.", "cs.PL")],
            published="2032-06-01")

    def no_rerank(*a, **k):
        raise AssertionError("reranker must not run in browse mode")

    monkeypatch.setattr(pipeline.reranker, "rerank", no_rerank)
    out = pipeline.search("give me paper on computer science", top_k=5, fetch_arxiv=False)
    assert out["interpretation"]["browse"] is True
    dates = [r["published"] for r in out["results"]]
    assert dates == sorted(dates, reverse=True)
    ids = [r["arxiv_id"] for r in out["results"]]
    assert ids[:2] == ["7002.00002", "7002.00001"]
    assert all(r["score"] is None for r in out["results"])
