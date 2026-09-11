"""End-to-end tests: search -> lazy index -> QA -> API -> evaluation."""
from __future__ import annotations

import json


def test_search_pipeline_end_to_end(pipeline):
    out = pipeline.search(
        "hybrid retrieval reranking for scientific RAG",
        fetch_arxiv=False,   # offline test
        lazy_load=False,
    )
    assert out["results"], "hybrid search returned nothing"
    top_ids = [r["arxiv_id"] for r in out["results"][:3]]
    assert set(top_ids) & {"2503.23013", "2508.08742", "2506.17277"}
    assert out["latency_ms"] > 0
    scores = [r["score"] for r in out["results"]]
    assert scores == sorted(scores, reverse=True)


def test_lazy_indexing_from_abstract_and_single_qa(pipeline):
    # 2508.17694 has no reachable PDF in tests -> abstract fallback chunks
    n = pipeline.ensure_paper_indexed("2508.17694")
    assert n >= 1
    # idempotency: second call must not duplicate chunks
    n2 = pipeline.ensure_paper_indexed("2508.17694")
    assert n2 == n

    res = pipeline.ask_single(
        "What does this paper show about transformer embeddings versus BM25?",
        "2508.17694",
    )
    assert res.mode == "single"
    assert res.sources, "single-doc QA must attribute sources"
    assert "BM25" in res.answer or "transformer" in res.answer.lower()


def test_multi_document_map_reduce_qa(pipeline):
    for pid in ("2503.23013", "2402.01767", "2506.17277"):
        pipeline.ensure_paper_indexed(pid)
    res = pipeline.ask_multi(
        "How can retrieval augmented generation systems be improved?",
        paper_ids=["2503.23013", "2402.01767", "2506.17277"],
    )
    assert res.mode == "multi"
    assert len(res.papers) >= 2, "reduce stage should synthesize >=2 papers"
    assert res.answer and "not contain" not in res.answer
    assert all(s.snippet for s in res.sources)


def test_conversation_history_is_accepted(pipeline):
    pipeline.ensure_paper_indexed("2503.23013")
    res = pipeline.ask_single(
        "And how is the weight adjusted?",
        "2503.23013",
        history=[
            {"role": "user", "content": "What is dynamic alpha tuning?"},
            {"role": "assistant", "content": "It balances BM25 and dense scores."},
        ],
    )
    assert res.answer


def test_recommendations(pipeline):
    recs = pipeline.recommender.similar_papers("2503.23013", top_k=3)
    assert recs
    ids = [r["arxiv_id"] for r in recs]
    assert "2503.23013" not in ids                 # never recommend itself
    assert all("score" in r for r in recs)


def test_stats_and_query_logging(pipeline):
    stats = pipeline.stats()
    assert stats["papers"] >= 6
    assert stats["chunk_vectors"] >= 1
    assert stats["queries"] >= 1                   # searches above were logged


def test_api_routes(pipeline):
    from fastapi.testclient import TestClient

    import backend.app.services.pipeline as pl
    from backend.app.main import app

    pl._pipeline = pipeline  # reuse the seeded pipeline
    client = TestClient(app)

    r = client.get("/api/v1/health")
    assert r.status_code == 200

    r = client.post(
        "/api/v1/search",
        json={"query": "multi document question answering",
              "fetch_arxiv": False, "lazy_load": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["results"]

    r = client.post(
        "/api/v1/qa/single",
        json={"question": "What is HiQA?", "arxiv_id": "2402.01767"},
    )
    assert r.status_code == 200
    assert r.json()["sources"]

    r = client.get("/api/v1/papers/2402.01767/recommendations")
    assert r.status_code == 200

    r = client.get("/api/v1/admin/stats")
    assert r.status_code == 200

    r = client.get("/api/v1/papers/does-not-exist")
    assert r.status_code == 404


def test_evaluation_metrics_and_harness(pipeline, tmp_path):
    from evaluation.evaluate import evaluate, mrr, ndcg_at_k, recall_at_k

    ranked = ["a", "b", "c", "d"]
    rel = {"b", "d"}
    assert recall_at_k(ranked, rel, 2) == 0.5
    assert mrr(ranked, rel) == 0.5
    assert 0 < ndcg_at_k(ranked, rel, 4) <= 1

    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"query": "dynamic alpha tuning hybrid retrieval",
                 "relevant": ["2503.23013"]},
                {"query": "benchmark reranking scientific",
                 "relevant": ["2508.08742"]},
                {"query": "multi document question answering hierarchical",
                 "relevant": ["2402.01767"]},
            ]
        )
    )
    report = evaluate(eval_file, ks=[1, 5])
    assert report["_n_queries"] == 3
    for strategy in ("bm25", "dense", "hybrid", "hybrid+rerank"):
        assert 0.0 <= report[strategy]["recall@5"] <= 1.0
    # hybrid should not be catastrophically worse than bm25 on this tiny set
    assert report["hybrid"]["recall@5"] >= report["bm25"]["recall@5"] - 0.34
